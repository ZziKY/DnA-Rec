import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GBSR import kernel_matrix, hsic


class DnARec(nn.Module):
    """
    DNA-REC: item-item graph bottleneck for long-tail recommendation.

    A popularity-aware MLP gates each item-item co-occurrence edge.
    Optional attribute graph (modes A/B/C) injects semantic signals for tail items.
    """

    def __init__(self, args, dataset, item_feats_np, item_degrees_dict, coo_i, coo_j):
        super(DnARec, self).__init__()
        self.num_user   = args.num_user
        self.num_item   = args.num_item
        self.gcn_layer  = args.gcn_layer
        self.latent_dim = args.latent_dim
        self.l2_reg     = args.l2_reg
        self.beta       = args.beta
        self.sigma      = args.sigma
        self.edge_bias  = args.edge_bias
        self.gate_temp  = args.gate_temp
        self.batch_size = args.batch_size
        self.device = torch.device(
            'cuda:' + str(args.device_id) if torch.cuda.is_available() else 'cpu'
        )

        self.user_embeddings = nn.Embedding(self.num_user, self.latent_dim)
        self.item_embeddings = nn.Embedding(self.num_item, self.latent_dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

        feat_dim = item_feats_np.shape[1]
        self.item_feats = nn.Parameter(
            torch.tensor(item_feats_np, dtype=torch.float32), requires_grad=False
        )
        self.attr_proj = nn.Linear(feat_dim, self.latent_dim, bias=False)

        # Popularity-aware MLP: [attr_proj(a), attr_proj(b), log_deg_a, log_deg_b] → logit
        self.linear_1  = nn.Linear(2 * self.latent_dim + 2, self.latent_dim, bias=True)
        self.linear_2  = nn.Linear(self.latent_dim, 1, bias=True)
        self.pop_alpha = nn.Parameter(torch.tensor(args.pop_alpha_init))
        self.activate  = nn.ReLU()

        deg_vec = torch.zeros(self.num_item, dtype=torch.float32)
        for iid, d in item_degrees_dict.items():
            if iid < self.num_item:
                deg_vec[iid] = float(d)
        self.register_buffer('deg_vec', deg_vec)

        self.lambda_cl      = args.lambda_cl
        self.cl_option      = args.cl_option
        self.cl_temp        = getattr(args, 'cl_temp', 0.2)
        self.cl_detach      = True   # anchor L(a) is detached; only attr_proj receives gradient
        self.cl_convergence = getattr(args, 'cl_convergence', 10.0)
        self.cl_ips_clip    = getattr(args, 'cl_ips_clip', 5.0)
        self.projection_head = getattr(args, 'projection_head', False)

        self.aux_hsic        = getattr(args, 'aux_hsic', False)
        self.lambda_aux_hsic = getattr(args, 'lambda_aux_hsic', 1.0)

        if self.projection_head:
            self.cl_proj = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )

        self.adj_matrix, self.item_item_index = dataset.get_ii_u_matrix(coo_i, coo_j)

        adj_idx = self.adj_matrix.indices()
        ii_idx  = torch.tensor(self.item_item_index, dtype=torch.long)
        self.ii_row = adj_idx[0][ii_idx]
        self.ii_col = adj_idx[1][ii_idx]

        # Pure bipartite adj (user-item only, item-item edges zeroed out)
        pure_weights = self.adj_matrix.values().clone()
        pure_weights[ii_idx] = 0.0
        self.pure_bipartite_adj = torch.sparse.FloatTensor(
            self.adj_matrix.indices().clone(),
            pure_weights,
            self.adj_matrix.size(),
        ).coalesce()

        self.attr_graph_mode = getattr(args, 'attr_graph_mode', 'none')
        self.k_attr          = getattr(args, 'k_attr', 10)
        self.lambda_attr     = getattr(args, 'lambda_attr', 0.1)
        self.gamma_attr      = getattr(args, 'gamma_attr', 0.5)
        self.deg_thresh      = getattr(args, 'deg_thresh', 10)

        if self.attr_graph_mode != 'none':
            print(f'[attr_graph] building cosine adj (k_attr={self.k_attr}, mode={self.attr_graph_mode})...')
            self.attr_adj_ii = self._build_attr_adj_ii(item_feats_np)
            if self.attr_graph_mode == 'B':
                self.tail_cosine_adj_full = self._build_tail_cosine_adj_full()
                print(f'[attr_graph] tail cosine edges: {self.tail_cosine_adj_full._nnz()}')

    # ── Graph learner ──────────────────────────────────────────────────────────

    def item_graph_learner(self):
        """Gumbel-soft gate over item-item edges."""
        item_a_ids = self.ii_row - self.num_user
        item_b_ids = self.ii_col - self.num_user

        attr_a    = self.attr_proj(self.item_feats[item_a_ids])
        attr_b    = self.attr_proj(self.item_feats[item_b_ids])
        log_deg_a = torch.log1p(self.deg_vec[item_a_ids]).unsqueeze(1)
        log_deg_b = torch.log1p(self.deg_vec[item_b_ids]).unsqueeze(1)

        cat_input  = torch.cat([attr_a, attr_b, log_deg_a, log_deg_b], dim=1)
        base_logit = self.linear_2(self.activate(self.linear_1(cat_input))).view(-1)

        alpha    = F.softplus(self.pop_alpha)
        pop_flow = (log_deg_b - log_deg_a).view(-1)
        logit    = base_logit + alpha * pop_flow

        if self.training:
            eps    = torch.rand_like(logit).clamp(1e-7, 1 - 1e-7)
            gumbel = torch.log(eps) - torch.log(1 - eps)
            mask   = torch.sigmoid((logit + gumbel) / self.gate_temp) + self.edge_bias
        else:
            mask = torch.sigmoid(logit / self.gate_temp) + self.edge_bias

        weights = torch.ones_like(self.adj_matrix.values())
        ii_idx  = torch.tensor(self.item_item_index, dtype=torch.long, device=self.device)
        weights[ii_idx] = mask

        masked_Graph = torch.sparse.FloatTensor(
            self.adj_matrix.indices(),
            self.adj_matrix.values() * weights,
            self.adj_matrix.size(),
        ).coalesce().to(self.device)
        return masked_Graph

    # ── Propagation ────────────────────────────────────────────────────────────

    def forward(self, adj_matrix):
        """LightGCN mean-pooled propagation."""
        ego_emb = torch.cat(
            [self.user_embeddings.weight, self.item_embeddings.weight], dim=0
        )
        all_emb = [ego_emb]
        for _ in range(self.gcn_layer):
            all_emb.append(torch.sparse.mm(adj_matrix, all_emb[-1]))
        mean_emb = torch.mean(torch.stack(all_emb, dim=1), dim=1)
        return torch.split(mean_emb, [self.num_user, self.num_item])

    def forward_last_layer(self, adj_matrix):
        """LightGCN — final layer only (used for infoNCE anchor)."""
        emb = torch.cat(
            [self.user_embeddings.weight, self.item_embeddings.weight], dim=0
        )
        for _ in range(self.gcn_layer):
            emb = torch.sparse.mm(adj_matrix, emb)
        return torch.split(emb, [self.num_user, self.num_item])

    def forward_attr(self, attr_adj_ii):
        """LightGCN on item-only cosine-sim graph."""
        if attr_adj_ii._nnz() == 0:
            return self.item_embeddings.weight
        emb = self.item_embeddings.weight
        all_emb = [emb]
        for _ in range(self.gcn_layer):
            emb = torch.sparse.mm(attr_adj_ii, all_emb[-1])
            all_emb.append(emb)
        return torch.mean(torch.stack(all_emb, dim=1), dim=1)

    # ── Attr graph builders ────────────────────────────────────────────────────

    def _build_attr_adj_ii(self, item_feats_np):
        """Top-k cosine-sim item-item adj, D^{-1/2} A D^{-1/2} normalized."""
        from sklearn.preprocessing import normalize as sk_normalize

        feats = sk_normalize(item_feats_np, norm='l2').astype(np.float32)
        n, k  = self.num_item, self.k_attr

        rows, cols, vals = [], [], []
        chunk = 256
        for start in range(0, n, chunk):
            end  = min(start + chunk, n)
            sims = (feats[start:end] @ feats.T).astype(np.float32)
            for local_i in range(end - start):
                global_i          = start + local_i
                sim_row           = sims[local_i].copy()
                sim_row[global_i] = -2.0
                top_k = np.argpartition(sim_row, -k)[-k:]
                for j in top_k:
                    w = float(max(0.0, sim_row[j]))
                    rows.append(global_i)
                    cols.append(int(j))
                    vals.append(w)

        rows_np = np.array(rows, dtype=np.int64)
        cols_np = np.array(cols, dtype=np.int64)
        vals_np = np.array(vals, dtype=np.float32)

        deg = np.zeros(n, dtype=np.float32)
        np.add.at(deg, rows_np, vals_np)
        deg[deg == 0.0] = 1.0
        d_inv_sqrt = (1.0 / np.sqrt(deg)).astype(np.float32)
        vals_norm  = vals_np * d_inv_sqrt[rows_np] * d_inv_sqrt[cols_np]

        indices = torch.LongTensor(np.stack([rows_np, cols_np]))
        values  = torch.FloatTensor(vals_norm)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size([n, n])).coalesce()
        return adj.to(self.device)

    def _build_tail_cosine_adj_full(self):
        """Cosine edges restricted to tail items, excluding existing co-occurrence edges (Mode B)."""
        ii_row_items  = (self.ii_row - self.num_user).cpu().tolist()
        ii_col_items  = (self.ii_col - self.num_user).cpu().tolist()
        existing_edges = set()
        for i, j in zip(ii_row_items, ii_col_items):
            existing_edges.add((i, j));  existing_edges.add((j, i))

        deg_np   = self.deg_vec.cpu().numpy()
        tail_set = set(int(i) for i in np.where(deg_np < self.deg_thresh)[0])

        attr_idx = self.attr_adj_ii.cpu().indices().numpy()
        attr_val = self.attr_adj_ii.cpu().values().numpy()

        new_rows, new_cols, new_vals = [], [], []
        for e in range(attr_idx.shape[1]):
            i, j = int(attr_idx[0, e]), int(attr_idx[1, e])
            if (i in tail_set or j in tail_set) and (i, j) not in existing_edges:
                new_rows.append(i + self.num_user)
                new_cols.append(j + self.num_user)
                new_vals.append(float(attr_val[e]))

        size = self.num_user + self.num_item
        if not new_rows:
            return torch.sparse.FloatTensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0),
                torch.Size([size, size])
            ).coalesce().to(self.device)

        indices = torch.LongTensor([new_rows, new_cols])
        values  = torch.FloatTensor(new_vals)
        return torch.sparse.FloatTensor(
            indices, values, torch.Size([size, size])
        ).coalesce().to(self.device)

    def _add_sparse_adj(self, a, b, scale=1.0):
        idx = torch.cat([a.indices(), b.indices()], dim=1)
        val = torch.cat([a.values(), b.values() * scale])
        return torch.sparse_coo_tensor(idx, val, a.size()).coalesce()

    # ── Loss functions ─────────────────────────────────────────────────────────

    def bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb):
        u_emb   = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]

        u0   = self.user_embeddings(users)
        pos0 = self.item_embeddings(pos_items)
        neg0 = self.item_embeddings(neg_items)
        reg_loss = (u0.norm(2).pow(2) + pos0.norm(2).pow(2) + neg0.norm(2).pow(2)) / (2 * len(users))

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        auc      = (pos_scores > neg_scores).float().mean()
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9).mean()
        return auc, bpr_loss, reg_loss * self.l2_reg

    def hsic_loss(self, users, pos_items, user_emb_old, item_emb_old, user_emb, item_emb):
        """HSIC between unmasked and masked embeddings (users + items)."""
        unique_u = torch.unique(users)
        unique_i = torch.unique(pos_items)
        m = self.batch_size

        ix = F.normalize(item_emb_old[unique_i], p=2, dim=1)
        iy = F.normalize(item_emb[unique_i],     p=2, dim=1)
        loss_item = hsic(kernel_matrix(ix, self.sigma), kernel_matrix(iy, self.sigma), m)

        ux = F.normalize(user_emb_old[unique_u], p=2, dim=1)
        uy = F.normalize(user_emb[unique_u],     p=2, dim=1)
        loss_user = hsic(kernel_matrix(ux, self.sigma), kernel_matrix(uy, self.sigma), m)

        return loss_item + loss_user

    def _aux_hsic_loss(self, pos_items, item_emb_coo):
        """HSIC(E^mask_item, log(1+deg)) — penalises popularity bias."""
        unique_i = torch.unique(pos_items)
        m = self.batch_size
        ix = F.normalize(item_emb_coo[unique_i], p=2, dim=1)

        deg_feat = torch.log1p(self.deg_vec[unique_i]).unsqueeze(1)
        d_min, d_max = deg_feat.min(), deg_feat.max()
        if d_max > d_min:
            deg_feat = (deg_feat - d_min) / (d_max - d_min)

        return hsic(kernel_matrix(ix, self.sigma), kernel_matrix(deg_feat, self.sigma), m)

    def _apply_hsic(self, users, pos_items, user_emb_old, item_emb_old, user_emb_coo, item_emb_coo):
        ib = self.hsic_loss(users, pos_items, user_emb_old, item_emb_old, user_emb_coo, item_emb_coo)
        if self.aux_hsic:
            ib = ib + self._aux_hsic_loss(pos_items, item_emb_coo) * self.lambda_aux_hsic
        return ib * self.beta

    def _pop_weight(self, unique_items):
        """Pop(i) = 1 - r / (r + exp(deg_i / r))."""
        r   = self.cl_convergence
        deg = self.deg_vec[unique_items]
        return 1.0 - r / (r + torch.exp(deg / r))

    def _ips_weight(self, unique_items):
        """IPS normalised to [0, 1]."""
        deg  = self.deg_vec[unique_items]
        prob = (deg + 1.0) / (self.num_user + 1.0)
        ips  = torch.sqrt(1.0 / prob).clamp(max=self.cl_ips_clip)
        return (ips - 1.0) / (self.cl_ips_clip - 1.0 + 1e-8)

    def infonce_loss(self, pos_items, item_emb_last):
        """InfoNCE between pure-bipartite LightGCN embeddings and attr_proj embeddings.

        Options:
          1 – uniform  2 – deg/max_deg  3 – 1-deg/max_deg
          4 – Pop(i)   5 – 1-Pop(i)     6 – IPS   7 – 1-IPS
        """
        unique_items = torch.unique(pos_items)

        z_raw = item_emb_last[unique_items]
        z_a   = F.normalize(z_raw.detach() if self.cl_detach else z_raw, p=2, dim=1)

        h_raw = self.attr_proj(self.item_feats[unique_items])
        if self.projection_head:
            h_raw = self.cl_proj(h_raw)
        h_a = F.normalize(h_raw, p=2, dim=1)

        sim    = z_a @ h_a.T / self.cl_temp
        labels = torch.arange(sim.size(0), device=self.device)

        if self.cl_option == 1:
            return F.cross_entropy(sim, labels)

        per_loss = F.cross_entropy(sim, labels, reduction='none')

        if self.cl_option == 2:
            max_deg  = self.deg_vec.max().clamp(min=1.0)
            weights  = self.deg_vec[unique_items] / max_deg
        elif self.cl_option == 3:
            max_deg  = self.deg_vec.max().clamp(min=1.0)
            weights  = 1.0 - self.deg_vec[unique_items] / max_deg
        elif self.cl_option == 4:
            weights = self._pop_weight(unique_items)
        elif self.cl_option == 5:
            weights = 1.0 - self._pop_weight(unique_items)
        elif self.cl_option == 6:
            weights = self._ips_weight(unique_items)
        else:
            weights = 1.0 - self._ips_weight(unique_items)

        return (weights * per_loss).mean()

    def _attr_cl_loss(self, anchor, positive, pos_items, symmetric=False):
        """InfoNCE between two item embedding views over the current batch."""
        unique_items = torch.unique(pos_items)
        z = F.normalize(anchor[unique_items],   p=2, dim=1)
        h = F.normalize(positive[unique_items], p=2, dim=1)
        labels = torch.arange(z.size(0), device=self.device)

        loss = F.cross_entropy(z @ h.T / self.cl_temp, labels)
        if symmetric:
            loss = (loss + F.cross_entropy(h @ z.T / self.cl_temp, labels)) / 2.0
        return loss

    # ── Training & inference ───────────────────────────────────────────────────

    def calculate_all_loss(self, users, pos_items, neg_items):
        masked_adj = self.item_graph_learner()

        if self.attr_graph_mode == 'none':
            user_emb_old, item_emb_old = self.forward(self.adj_matrix)
            user_emb, item_emb         = self.forward(masked_adj)
            user_emb_coo, item_emb_coo = user_emb, item_emb
            auc, bpr_loss, reg_loss    = self.bpr_loss(users, pos_items, neg_items, user_emb, item_emb)
            ib_loss = self._apply_hsic(users, pos_items,
                                       user_emb_old, item_emb_old,
                                       user_emb_coo, item_emb_coo)

        elif self.attr_graph_mode == 'A':
            user_emb_old, item_emb_old = self.forward(self.adj_matrix)
            user_emb, item_emb_coo     = self.forward(masked_adj)
            item_emb_attr              = self.forward_attr(self.attr_adj_ii)
            item_emb = (item_emb_coo + item_emb_attr) / 2.0
            user_emb_coo = user_emb
            auc, bpr_loss, reg_loss = self.bpr_loss(users, pos_items, neg_items, user_emb, item_emb)
            ib_loss = self._apply_hsic(users, pos_items,
                                       user_emb_old, item_emb_old,
                                       user_emb_coo, item_emb_coo)

        elif self.attr_graph_mode == 'B':
            user_emb_old, item_emb_old = self.forward(self.adj_matrix)
            user_emb_coo, item_emb_coo = self.forward(masked_adj)
            augmented_adj              = self._add_sparse_adj(
                masked_adj, self.tail_cosine_adj_full, self.gamma_attr)
            user_emb, item_emb = self.forward(augmented_adj)
            auc, bpr_loss, reg_loss = self.bpr_loss(users, pos_items, neg_items, user_emb, item_emb)
            ib_loss = self._apply_hsic(users, pos_items,
                                       user_emb_old, item_emb_old,
                                       user_emb_coo, item_emb_coo)

        elif self.attr_graph_mode == 'C':
            user_emb_old, item_emb_old = self.forward(self.adj_matrix)
            user_emb, item_emb_coo     = self.forward(masked_adj)
            item_emb_attr = self.forward_attr(self.attr_adj_ii)
            item_emb = item_emb_coo
            user_emb_coo = user_emb
            auc, bpr_loss, reg_loss = self.bpr_loss(users, pos_items, neg_items, user_emb, item_emb)
            ib_loss = self._apply_hsic(users, pos_items,
                                       user_emb_old, item_emb_old,
                                       user_emb_coo, item_emb_coo)

        # InfoNCE: align pure-bipartite embedding with attr_proj
        if self.lambda_cl > 0.0:
            _, item_emb_last = self.forward_last_layer(self.pure_bipartite_adj.to(self.device))
            cl_loss = self.infonce_loss(pos_items, item_emb_last) * self.lambda_cl
        else:
            cl_loss = torch.tensor(0.0, device=self.device)

        # Attr-graph CL: mode A (one-way) / mode C (symmetric)
        if self.attr_graph_mode in ('A', 'C') and self.lambda_attr > 0.0:
            symmetric = (self.attr_graph_mode == 'C')
            anchor    = item_emb_coo.detach() if self.attr_graph_mode == 'A' else item_emb_coo
            attr_cl   = self._attr_cl_loss(anchor, item_emb_attr, pos_items, symmetric=symmetric)
            cl_loss   = cl_loss + attr_cl * self.lambda_attr

        total = bpr_loss + reg_loss + ib_loss + cl_loss
        return auc, bpr_loss, reg_loss, ib_loss, cl_loss, total

    def get_embeddings(self):
        self.eval()
        with torch.no_grad():
            masked_adj = self.item_graph_learner()
            user_emb, item_emb = self.forward(masked_adj)
        return user_emb.cpu().numpy(), item_emb.cpu().numpy()
