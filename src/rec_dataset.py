import torch
import numpy as np
import os
from time import time
import numba as nb
from collections import defaultdict
import random
import pandas as pd


@nb.njit()
def negative_sampling(training_user, training_item, traindata, num_item, num_negative):
    trainingData = []
    for k in range(len(training_user)):
        u = training_user[k]
        pos_i = training_item[k]
        for _ in range(num_negative):
            neg_j = random.randint(0, num_item - 1)
            while neg_j in traindata[u]:
                neg_j = random.randint(0, num_item - 1)
            trainingData.append([u, pos_i, neg_j])
    return np.array(trainingData)


PKL_DATASETS = {'book_crossing', 'movielens_1m', 'amazon_fashion'}


class Dataset(object):
    def __init__(self, args):
        self.device = torch.device('cuda:' + str(args.device_id) if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.data_path = args.data_path
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.num_node = self.num_user + self.num_item
        self.batch_size = args.batch_size
        self.split = getattr(args, 'split', 'default')
        self.dataset_name = getattr(args, 'dataset', '')

        self.load_data()
        self.val_maskdata = self._copy_rating_dict(self.traindata)
        self.test_maskdata = self._copy_rating_dict(self.traindata)
        self.data_to_numba_dict()
        self.training_user, self.training_item = [], []
        for u, items in self.traindata.items():
            self.training_user.extend([u] * len(items))
            self.training_item.extend(items)

    @property
    def itemDegrees(self):
        deg = defaultdict(int)
        for items in self.traindata.values():
            for it in items:
                deg[it] += 1
        return deg

    @staticmethod
    def _copy_rating_dict(ratings):
        return {int(user): [int(item) for item in items]
                for user, items in ratings.items()}

    def load_data(self):
        if self.dataset_name in PKL_DATASETS:
            self._load_pkl_data()
        else:
            self._load_npy_data()

    def _load_pkl_data(self):
        suffix = '_split8' if self.split == 'split8' else ''

        def _read(base):
            parquet = self.data_path + base + '.parquet'
            pkl = self.data_path + base + '.pkl'
            if os.path.exists(parquet):
                try:
                    return pd.read_parquet(parquet)
                except Exception:
                    pass
            return pd.read_pickle(pkl)

        train_df = _read(f'interact_train{suffix}')
        val_df   = _read(f'interact_val{suffix}')
        test_df  = _read(f'interact_test{suffix}')
        self.traindata = train_df.groupby('userid')['itemid'].apply(list).to_dict()
        self.valdata   = val_df.groupby('userid')['itemid'].apply(list).to_dict() if len(val_df) > 0 else {}
        self.testdata  = test_df.groupby('userid')['itemid'].apply(list).to_dict()
        print(f'Loaded {self.dataset_name} ({suffix or "default"}): '
              f'train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

    def _load_npy_data(self):
        self.traindata = np.load(self.data_path + 'traindata.npy', allow_pickle=True).tolist()
        val_path = self.data_path + 'valdata.npy'
        self.valdata = np.load(val_path, allow_pickle=True).tolist() if os.path.exists(val_path) else {}
        self.testdata = np.load(self.data_path + 'testdata.npy', allow_pickle=True).tolist()

    def data_to_numba_dict(self):
        self.traindict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:])
        for key, values in self.traindata.items():
            if len(values) > 0:
                self.traindict[key] = np.asarray(list(values), dtype=np.int64)

        self.valdict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:])
        for key, values in self.valdata.items():
            if len(values) > 0:
                self.valdict[key] = np.asarray(list(values), dtype=np.int64)

        self.testdict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:])
        for key, values in self.testdata.items():
            if len(values) > 0:
                self.testdict[key] = np.asarray(list(values), dtype=np.int64)

    def load_item_features(self):
        """Load pre-computed item attribute features (num_item, feat_dim)."""
        path = os.path.join(self.data_path, 'item_attr_feats.npy')
        feats = np.load(path).astype(np.float32)
        assert feats.shape[0] == self.num_item, (
            f'Feature rows ({feats.shape[0]}) != num_item ({self.num_item})')
        return feats

    def build_item_coo_v1(self, k):
        """Random-k item co-occurrence neighbors per item."""
        user_items = defaultdict(list)
        for u, items in self.traindata.items():
            for it in items:
                user_items[u].append(it)
        item_users = defaultdict(list)
        for u, items in self.traindata.items():
            for it in items:
                item_users[it].append(u)

        coo_i, coo_j = [], []
        for item_a in range(self.num_item):
            co_pool = set()
            for u in item_users.get(item_a, []):
                co_pool.update(user_items[u])
            co_pool.discard(item_a)
            if not co_pool:
                continue
            neighbors = random.sample(sorted(co_pool), min(k, len(co_pool)))
            for item_b in neighbors:
                coo_i.append(item_a);  coo_j.append(item_b)
                coo_i.append(item_b);  coo_j.append(item_a)
        print(f'build_item_coo_v1(k={k}): {len(coo_i)//2} edges (one-way)')
        return coo_i, coo_j

    def build_item_coo_v2(self, item_feats_np, k):
        """Cosine-similarity top-k item co-occurrence neighbors per item."""
        from sklearn.preprocessing import normalize as sk_normalize

        user_items = defaultdict(list)
        for u, items in self.traindata.items():
            for it in items:
                user_items[u].append(it)
        item_users = defaultdict(list)
        for u, items in self.traindata.items():
            for it in items:
                item_users[it].append(u)

        feats_normed = sk_normalize(item_feats_np, norm='l2')

        coo_i, coo_j = [], []
        for item_a in range(self.num_item):
            co_pool = set()
            for u in item_users.get(item_a, []):
                co_pool.update(user_items[u])
            co_pool.discard(item_a)
            if not co_pool:
                continue
            co_list = list(co_pool)
            if len(co_list) <= k:
                neighbors = co_list
            else:
                sims = feats_normed[item_a] @ feats_normed[np.array(co_list)].T
                top_idx = np.argpartition(sims, -k)[-k:]
                neighbors = [co_list[i] for i in top_idx]
            for item_b in neighbors:
                coo_i.append(item_a);  coo_j.append(item_b)
                coo_i.append(item_b);  coo_j.append(item_a)
        print(f'build_item_coo_v2(k={k}): {len(coo_i)//2} edges (one-way)')
        return coo_i, coo_j

    def get_ii_u_matrix(self, coo_i, coo_j):
        """Build normalized sparse adj: user-item interactions + item-item co-occurrences.

        Item nodes are offset by num_user (binary edge weights).
        Returns (Graph [torch sparse float], item_item_edge_indices [list[int]]).
        """
        user_dim = torch.LongTensor(self.training_user)
        item_dim = torch.LongTensor(self.training_item) + self.num_user
        ii_src   = torch.LongTensor(coo_i) + self.num_user
        ii_dst   = torch.LongTensor(coo_j) + self.num_user

        index = torch.cat([
            torch.stack([user_dim, item_dim]),
            torch.stack([item_dim, user_dim]),
            torch.stack([ii_src,   ii_dst]),
        ], dim=1)
        size = torch.Size([self.num_user + self.num_item, self.num_user + self.num_item])

        data  = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data, size)
        dense = Graph.to_dense().float()

        D = torch.sum(dense, dim=1)
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()

        index = dense.nonzero()
        data  = dense[dense >= 1e-9]
        Graph = torch.sparse.FloatTensor(index.t(), data, size)
        Graph = Graph.coalesce().to(self.device)

        item_item_edge_indices = [
            i for i in range(index.shape[0])
            if index[i][0] >= self.num_user and index[i][1] >= self.num_user
        ]
        print(f'get_ii_u_matrix: total edges={index.shape[0]}, '
              f'item-item edges={len(item_item_edge_indices)}')
        return Graph, item_item_edge_indices

    def _batch_sampling(self, num_negative):
        t1 = time()
        triplet_data = negative_sampling(
            nb.typed.List(self.training_user), nb.typed.List(self.training_item),
            self.traindict, self.num_item, num_negative)
        print('prepare training data cost time:{:.4f}'.format(time() - t1))
        batch_num = int(len(triplet_data) / self.batch_size) + 1
        indexs = np.arange(triplet_data.shape[0])
        np.random.shuffle(indexs)
        for k in range(batch_num):
            index_start = k * self.batch_size
            index_end = min((k + 1) * self.batch_size, len(indexs))
            if index_end == len(indexs):
                index_start = len(indexs) - self.batch_size
            batch_data = triplet_data[indexs[index_start:index_end]]
            yield batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]
