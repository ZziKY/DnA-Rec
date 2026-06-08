import os
import sys
import argparse
import json
from copy import deepcopy
from collections import defaultdict
from shutil import copyfile
from time import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.append('./')
sys.path.append('../')
from log import Logger
from set import seed_everything, set_color
from dna_rec import DnARec
from rec_dataset import Dataset
from evaluate import (
    num_faiss_evaluate,
    num_faiss_evaluate_head_tail,
    num_faiss_evaluate_hsbt,
    build_5group_segments,
)

DATASET_INFO = {
    'book_crossing':  {'num_user': 6330, 'num_item': 5836},
    'amazon_fashion': {'num_user': 1908, 'num_item': 2065},
}

BAR  = '-' * 80
DBAR = '=' * 80


# ── Hyperparameter config helpers ──────────────────────────────────────────────

def _resolve_hparam_config_path(dataset_name, config_dir, explicit_path, no_autoload):
    script_root = os.path.dirname(os.path.abspath(__file__))
    if explicit_path:
        if os.path.isabs(explicit_path) or os.path.isfile(explicit_path):
            return explicit_path
        return os.path.join(script_root, explicit_path)
    if no_autoload:
        return None
    if not os.path.isabs(config_dir):
        config_dir = os.path.join(script_root, config_dir)
    candidate = os.path.join(config_dir, f'{dataset_name}.json')
    return candidate if os.path.isfile(candidate) else None


def _apply_hparam_defaults(parser, path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    valid_dests = {a.dest for a in parser._actions}
    defaults = {k: v for k, v in cfg.items() if k in valid_dests}
    unknown  = [k for k in cfg if k not in valid_dests]
    if defaults:
        parser.set_defaults(**defaults)
    if unknown:
        print(f'[hparam-config] Ignoring unknown keys: {unknown}')


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='DNA-REC')
    parser.add_argument('--dataset',           type=str,   default='book_crossing',
                        choices=sorted(DATASET_INFO.keys()))
    parser.add_argument('--split',             type=str,   default='split8',
                        choices=['default', 'split8'])
    parser.add_argument('--co_occur_version',  type=str,   default='v2',
                        choices=['v1', 'v2'],
                        help='v1=random-k neighbors, v2=cosine top-k neighbors')
    parser.add_argument('--k_neighbors',       type=int,   default=10,
                        help='Max item-item co-occurrence neighbors per item')
    parser.add_argument('--runid',             type=str,   default='0')
    parser.add_argument('--device_id',         type=str,   default='0')
    parser.add_argument('--epochs',            type=int,   default=300)
    parser.add_argument('--batch_size',        type=int,   default=512)
    parser.add_argument('--lr',                type=float, default=1e-3)
    parser.add_argument('--early_stops',       type=int,   default=20)
    parser.add_argument('--num_neg',           type=int,   default=1)
    parser.add_argument('--gcn_layer',         type=int,   default=2)
    parser.add_argument('--latent_dim',        type=int,   default=128)
    parser.add_argument('--l2_reg',            type=float, default=1e-4)
    parser.add_argument('--beta',              type=float, default=2.0,
                        help='Information bottleneck (HSIC) weight')
    parser.add_argument('--sigma',             type=float, default=0.5,
                        help='Kernel bandwidth for HSIC')
    parser.add_argument('--edge_bias',         type=float, default=0.25,
                        help='Gumbel gate bias')
    parser.add_argument('--gate_temp',         type=float, default=0.2,
                        help='Temperature for sigmoid gate over item-item edges')
    parser.add_argument('--pop_alpha_init',    type=float, default=0.1,
                        help='Initial value for popularity-flow strength')
    parser.add_argument('--lambda_cl',         type=float, default=0.0,
                        help='Weight for infoNCE contrastive loss')
    parser.add_argument('--cl_option',         type=int,   default=1,
                        choices=[1, 2, 3, 4, 5, 6, 7],
                        help='infoNCE sample weighting: 1=none, 2=pop, 3=1-pop, '
                             '4=Pop(i), 5=1-Pop(i), 6=IPS, 7=1-IPS')
    parser.add_argument('--cl_temp',           type=float, default=0.2,
                        help='Temperature for infoNCE')
    parser.add_argument('--cl_convergence',    type=float, default=10.0,
                        help='r in Pop(i) = 1 - r/(r+exp(deg/r))')
    parser.add_argument('--cl_ips_clip',       type=float, default=5.0,
                        help='Max clamp value for IPS weights')
    parser.add_argument('--projection_head',    action='store_true',  default=False)
    parser.add_argument('--no_projection_head', action='store_false', dest='projection_head')
    parser.add_argument('--attr_graph_mode',    type=str,   default='none',
                        choices=['none', 'A', 'B'],
                        help='Semantic augmentation: none, A=two-branch BPR+CL, '
                             'B=tail edge injection')
    parser.add_argument('--k_attr',             type=int,   default=10,
                        help='Top-k cosine neighbors in attr graph')
    parser.add_argument('--lambda_attr',        type=float, default=0.1,
                        help='Weight for attr-graph alignment loss (mode A)')
    parser.add_argument('--gamma_attr',         type=float, default=0.5,
                        help='Scale for tail cosine edges added to BPR graph (mode B)')
    parser.add_argument('--deg_thresh',         type=int,   default=10,
                        help='Degree threshold for tail items (modes B)')
    parser.add_argument('--top_rate',           type=float, default=0.2,
                        help='Head item ratio for head/tail splits')
    parser.add_argument('--K_list',             type=int,   nargs='+', default=[10, 20, 50])
    parser.add_argument('--seeds',              type=int,   nargs='+', default=[1, 2, 3])
    parser.add_argument('--result_path',        type=str,   default=None,
                        help='CSV path for final evaluation results')
    parser.add_argument('--hparam_config',      type=str,   default=None,
                        help='JSON file with hyperparameter defaults (CLI args override)')
    parser.add_argument('--hparam_config_dir',  type=str,   default='configs/hyperparams',
                        help='Auto-load dir: loads <dir>/<dataset>.json when present')
    parser.add_argument('--no_hparam_autoload', action='store_true', default=False)

    pre_args, _ = parser.parse_known_args()
    cfg_path = _resolve_hparam_config_path(
        dataset_name=pre_args.dataset,
        config_dir=pre_args.hparam_config_dir,
        explicit_path=pre_args.hparam_config,
        no_autoload=pre_args.no_hparam_autoload,
    )
    if cfg_path is not None:
        _apply_hparam_defaults(parser, cfg_path)
        print(f'[hparam-config] Loaded defaults from: {cfg_path}')

    args = parser.parse_args()
    if args.gate_temp <= 0:
        parser.error('--gate_temp must be > 0')
    return args


# ── Helpers ────────────────────────────────────────────────────────────────────

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _pct(r):
    return f'{r * 100:g}%'


def print_hsbt_results(log, dataset_name, K_list, top_rate,
                       hr_all, ndcg_all, hr_h, ndcg_h, hr_t, ndcg_t,
                       hsbt_metrics, hsbt_segments):
    lines = []

    def p(s=''):
        lines.append(s); print(s)

    p(DBAR)
    p(f'  Evaluation  |  dataset: {dataset_name}  K_list: {K_list}')
    p(DBAR)
    p()
    p('[1] Overall')
    p(BAR)
    for K in K_list:
        p(f'  @K={K:2d}  NDCG={ndcg_all[K]:.6f}  HR={hr_all[K]:.6f}')
    p(BAR)
    p()
    p(f'[2] Head / Tail  (head=top {_pct(top_rate)},  tail=bottom {_pct(1-top_rate)})')
    p(BAR)
    for K in K_list:
        p(f'  @K={K:2d}  head  NDCG={ndcg_h[K]:.6f}  HR={hr_h[K]:.6f}')
        p(f'         tail  NDCG={ndcg_t[K]:.6f}  HR={hr_t[K]:.6f}')
    p(BAR)
    p()
    p('[3] 5-Group  (each ~20%,  head=most popular / tail=least popular)')
    p(f'    items: head={len(hsbt_segments["head"])}  '
      f'shoulder={len(hsbt_segments["shoulder"])}  '
      f'body={len(hsbt_segments["body"])}  '
      f'knee={len(hsbt_segments["knee"])}  '
      f'tail={len(hsbt_segments["tail"])}')
    p(BAR)
    for K in K_list:
        for seg in ('head', 'shoulder', 'body', 'knee', 'tail'):
            hr_v, ndcg_v = hsbt_metrics[seg][K]
            p(f'  @K={K:2d}  {seg:<8s} NDCG={ndcg_v:.6f}  HR={hr_v:.6f}')
    p(BAR)
    p(DBAR)

    for line in lines:
        log.write(line + '\n')
    return lines


_SEGMENTS = ['overall', 'ht_head', 'ht_tail',
             '5g_head', '5g_shoulder', '5g_body', '5g_knee', '5g_tail']


def _get_seg_metrics(metrics, segment, K):
    if segment == 'overall':
        return metrics['hr_all'][K], metrics['ndcg_all'][K]
    if segment == 'ht_head':
        return metrics['hr_h'][K], metrics['ndcg_h'][K]
    if segment == 'ht_tail':
        return metrics['hr_t'][K], metrics['ndcg_t'][K]
    seg = segment[3:]
    hr_v, ndcg_v = metrics['hsbt_metrics'][seg][K]
    return hr_v, ndcg_v


def save_results_csv(result_path, K_list, seed_results):
    import pandas as pd

    K_list = sorted(K_list)
    agg  = {seg: {K: {'HR': [], 'NDCG': []} for K in K_list} for seg in _SEGMENTS}
    rows = []

    for seed, metrics in seed_results:
        for seg in _SEGMENTS:
            row = {'seed': seed, 'stat': 'run', 'segment': seg}
            for K in K_list:
                hr, ndcg = _get_seg_metrics(metrics, seg, K)
                row[f'HR@{K}']   = f'{float(hr):.6f}'
                row[f'NDCG@{K}'] = f'{float(ndcg):.6f}'
                agg[seg][K]['HR'].append(float(hr))
                agg[seg][K]['NDCG'].append(float(ndcg))
            rows.append(row)

    for stat in ('mean', 'std'):
        for seg in _SEGMENTS:
            row = {'seed': '', 'stat': stat, 'segment': seg}
            for K in K_list:
                hr_v   = agg[seg][K]['HR']
                ndcg_v = agg[seg][K]['NDCG']
                fn = np.mean if stat == 'mean' else lambda v: np.std(v, ddof=0)
                row[f'HR@{K}']   = f'{fn(hr_v):.6f}'
                row[f'NDCG@{K}'] = f'{fn(ndcg_v):.6f}'
            rows.append(row)

    cols = ['seed', 'stat', 'segment']
    for K in K_list:
        cols.append(f'HR@{K}')
    for K in K_list:
        cols.append(f'NDCG@{K}')

    pd.DataFrame(rows, columns=cols).to_csv(result_path, index=False)
    print(f'Results saved to {result_path}')


# ── Per-seed training ──────────────────────────────────────────────────────────

def run_one_seed(base_args, seed, root_record_path):
    seed_everything(seed)
    args = deepcopy(base_args)
    args.seed = seed

    record_path    = os.path.join(root_record_path, f'seed_{seed}') + os.sep
    model_save_path = os.path.join(record_path, 'models') + os.sep
    makedir(model_save_path)

    copyfile('./dna_rec.py', os.path.join(record_path, 'dna_rec.py'))
    copyfile('./train.py',   os.path.join(record_path, 'train.py'))

    log = Logger(record_path)
    for a in vars(args):
        log.write(f'{a}={getattr(args, a)}\n')

    rec_data = Dataset(args)
    device   = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

    print('Loading item attribute features...')
    item_feats_np = rec_data.load_item_features()
    item_degrees  = rec_data.itemDegrees
    print(f'  item_feats: {item_feats_np.shape}  feat_dim={item_feats_np.shape[1]}')

    print(f'Building item co-occurrence graph ({args.co_occur_version}, k={args.k_neighbors})...')
    if args.co_occur_version == 'v1':
        coo_i, coo_j, _ = rec_data.build_item_coo_v1(args.k_neighbors)
    else:
        coo_i, coo_j, _ = rec_data.build_item_coo_v2(item_feats_np, args.k_neighbors)

    model     = DnARec(args, rec_data, item_feats_np, item_degrees, coo_i, coo_j).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    # Item popularity splits (built once from training data)
    sorted_items = [i for i, _ in sorted(item_degrees.items(), key=lambda x: x[1])]
    n_items  = len(sorted_items)
    head_len = max(1, int(n_items * args.top_rate))
    ht_head  = sorted_items[-head_len:]
    ht_tail  = sorted_items[:-head_len]
    hsbt_segs = build_5group_segments(sorted_items)

    max_ndcg, early_stop, best_epoch = 0.0, 0, 0
    best_ckpt  = None
    model_files = []
    max_to_keep = 5

    for epoch in tqdm(range(args.epochs),
                      desc=set_color('Train:', 'pink'), colour='yellow',
                      dynamic_ncols=True, position=0):
        t1 = time()
        model.train()
        all_bpr, all_reg, all_ib, all_cl, all_total, batch_num = 0., 0., 0., 0., 0., 0

        for u, i, j in tqdm(rec_data._batch_sampling(num_negative=args.num_neg),
                             desc='batches', leave=False):
            u = torch.tensor(u, dtype=torch.long, device=device)
            i = torch.tensor(i, dtype=torch.long, device=device)
            j = torch.tensor(j, dtype=torch.long, device=device)

            auc, bpr_loss, reg_loss, ib_loss, cl_loss, total_loss = model.calculate_all_loss(u, i, j)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            all_bpr   += bpr_loss.item()
            all_reg   += reg_loss.item()
            all_ib    += ib_loss.item()
            all_cl    += cl_loss.item()
            all_total += total_loss.item()
            batch_num += 1

        t2 = time()
        log.write(set_color(
            f'Epoch:{epoch:d}  bpr:{all_bpr/batch_num:.4f}  '
            f'reg:{all_reg/batch_num:.4f}  ib:{all_ib/batch_num:.4f}  '
            f'cl:{all_cl/batch_num:.4f}  total:{all_total/batch_num:.4f}\n', 'blue'))

        if rec_data.valdata:
            early_stop += 1
            user_emb, item_emb = model.get_embeddings()
            hr_v, _, ndcg_v = num_faiss_evaluate(
                rec_data.valdata, rec_data.traindata, [20],
                user_emb, item_emb, list(rec_data.valdata.keys()))

            cur_ndcg = ndcg_v[20]
            log.write(set_color(
                f'Val  Epoch:{epoch:d}  NDCG@20={cur_ndcg:.4f}  HR@20={hr_v[20]:.4f}\n', 'green'))

            if cur_ndcg >= max_ndcg:
                max_ndcg   = cur_ndcg
                best_epoch = epoch
                early_stop = 0
                best_ckpt  = f'epoch_{epoch}_ndcg_{cur_ndcg:.4f}.tar'
                filepath   = model_save_path + best_ckpt
                torch.save({'args': vars(args), 'sd': model.state_dict()}, filepath)
                model_files.append(filepath)
                if len(model_files) > max_to_keep:
                    old = model_files.pop(0)
                    try:
                        os.remove(old)
                    except FileNotFoundError:
                        pass

            log.write(set_color(f'Best  Epoch:{best_epoch:d}  NDCG@20={max_ndcg:.4f}\n', 'red'))
            t3 = time()
            log.write(f'train:{t2-t1:.2f}s  val:{t3-t2:.2f}s\n\n')

            if epoch > 30 and early_stop > args.early_stops:
                log.write(f'Early stop at epoch {epoch}\n')
                break
        else:
            best_ckpt = f'epoch_{epoch}.tar'
            filepath  = model_save_path + best_ckpt
            torch.save({'args': vars(args), 'sd': model.state_dict()}, filepath)
            model_files.append(filepath)
            if len(model_files) > max_to_keep:
                old = model_files.pop(0)
                try:
                    os.remove(old)
                except FileNotFoundError:
                    pass

    # Final test evaluation
    ckpt = torch.load(model_save_path + best_ckpt, map_location='cpu')
    model.load_state_dict(ckpt['sd'])
    user_emb, item_emb = model.get_embeddings()

    K_list     = args.K_list
    test_users = list(rec_data.testdata.keys())
    test_maskdata = rec_data.test_maskdata

    hr_all, _, ndcg_all = num_faiss_evaluate(
        rec_data.testdata, test_maskdata, K_list, user_emb, item_emb, test_users)
    hr_h, _, ndcg_h, hr_t, _, ndcg_t = num_faiss_evaluate_head_tail(
        rec_data.testdata, test_maskdata, K_list,
        user_emb, item_emb, test_users, ht_head, ht_tail)
    hsbt_raw = num_faiss_evaluate_hsbt(
        rec_data.testdata, test_maskdata, K_list,
        user_emb, item_emb, test_users,
        hsbt_segs['head'], hsbt_segs['shoulder'],
        hsbt_segs['body'],  hsbt_segs['knee'], hsbt_segs['tail'])

    print_hsbt_results(log, args.dataset, K_list, args.top_rate,
                       hr_all, ndcg_all, hr_h, ndcg_h, hr_t, ndcg_t,
                       hsbt_raw, hsbt_segs)
    log.close()
    return {
        'hr_all':       hr_all,
        'ndcg_all':     ndcg_all,
        'hr_h':         hr_h,
        'ndcg_h':       ndcg_h,
        'hr_t':         hr_t,
        'ndcg_t':       ndcg_t,
        'hsbt_metrics': hsbt_raw,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    args.num_user  = DATASET_INFO[args.dataset]['num_user']
    args.num_item  = DATASET_INFO[args.dataset]['num_item']
    args.data_path = f'../datasets/{args.dataset}/'

    ver_tag = f'v{args.co_occur_version.lstrip("v")}'
    root_record_path = f'../saved/{args.dataset}/DnARec_{ver_tag}_k{args.k_neighbors}/{args.runid}/'
    makedir(root_record_path)

    seed_results = []
    for seed in args.seeds:
        print(DBAR)
        print(f'  DNA-REC | dataset={args.dataset}  split={args.split}  seed={seed}')
        print(DBAR)
        metrics = run_one_seed(args, seed, root_record_path)
        seed_results.append((seed, metrics))

    if args.result_path:
        makedir(os.path.dirname(os.path.abspath(args.result_path)))
        save_results_csv(args.result_path, args.K_list, seed_results)

    print('Done.')


if __name__ == '__main__':
    main()
