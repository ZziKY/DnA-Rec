# DNA-REC: Denoising and Augmentation of Item Graphs for Long-tail Recommendation

## Overview

How can we accurately recommend unpopular items to users? Despite receiving few interactions, tail items are valuable as they improve recommendation diversity, facilitate item discovery, and generate substantial business value. Graph-based recommenders often alleviate data sparsity by constructing auxiliary item graphs from co-occurrence patterns, allowing information to propagate across related items. However, accurately identifying useful relations in such graphs remains difficult as raw co-occurrence is contaminated by spurious relations and popularity bias, causing message propagation to amplify noise rather than reflect genuine user preferences. Moreover, denoising alone is insufficient since many tail items remain weakly connected or isolated, preventing them from receiving useful information.

In this paper, we propose **DNA-REC** (**D**enoising and **A**ugmentation of Item Graphs for Long-tail **Rec**ommendation), a framework for accurate long-tail recommendation that jointly identifies reliable item relations and enriches sparsely interacted tail items. Specifically, DNA-REC combines popularity-aware graph denoising, information bottleneck learning, and semantic augmentation to remove noisy and popularity-driven relations while providing additional semantic signals to isolated tail items.

## Requirements

```
torch >= 1.13
faiss-gpu (or faiss-cpu)
numba
numpy
pandas
scikit-learn
pyarrow
tqdm
```

Install with:
```bash
pip install torch faiss-gpu numba numpy pandas scikit-learn pyarrow tqdm
```

## Repository Structure

```
DNA-REC/
├── src/
│   ├── run_gbsr_item.py        # Training script
│   ├── gbsr_item.py            # DNA-REC model
│   ├── GBSR.py                 # HSIC kernel utilities
│   ├── rec_dataset.py          # Dataset loading and item graph construction
│   ├── evaluate.py             # FAISS-based ranking evaluation
│   ├── set.py                  # Utilities (seed, color)
│   ├── log.py                  # Logger
│   └── configs/
│       └── hyperparams/        # Per-dataset default hyperparameters
│           ├── amazon_fashion.json
│           ├── book_crossing.json
│           └── movielens_1m.json
├── datasets/
│   ├── amazon_fashion/
│   ├── book_crossing/
│   └── movielens_1m/
└── results/
```

## Datasets

Each dataset directory should contain:
- `interact_train_split8.parquet` / `interact_val_split8.parquet` / `interact_test_split8.parquet`
- `item_attr_feats.npy` — pre-computed item attribute embeddings `(num_item, feat_dim)`

Dataset statistics:

| Dataset | Users | Items |
|---|---|---|
| Amazon Fashion | 1,908 | 2,065 |
| Book-Crossing | 6,330 | 5,836 |
| MovieLens-1M | 6,040 | 3,493 |

## Quick Start

All commands are run from the `src/` directory.

### Amazon Fashion — Mode B (tail edge injection + Pop(i)-weighted CL)

```bash
cd src

python run_gbsr_item.py \
  --dataset amazon_fashion --device_id 0 \
  --attr_graph_mode B --cl_option 4 \
  --no_projection_head --no_aux_hsic --gcn_layer 2 \
  --k_neighbors 5 --beta 2.0 --sigma 0.5 --edge_bias 0.1 --latent_dim 256 \
  --gate_temp 0.4 --pop_alpha_init 0.5 \
  --lambda_cl 0.1 --cl_temp 0.2 --cl_convergence 100 --cl_ips_clip 10 \
  --k_attr 20 --lambda_attr 0.1 --gamma_attr 1.0 --deg_thresh 10 \
  --seeds 1 2 3 4 5 \
  --result_path ../results/amazon_fashion_B4.csv
```

### Amazon Fashion — Mode A (two-branch BPR + (1-IPS)-weighted CL)

```bash
python run_gbsr_item.py \
  --dataset amazon_fashion --device_id 0 \
  --attr_graph_mode A --cl_option 7 \
  --no_projection_head --no_aux_hsic --gcn_layer 2 \
  --k_neighbors 5 --beta 3.0 --sigma 0.5 --edge_bias 0.25 --latent_dim 256 \
  --gate_temp 0.2 --pop_alpha_init 0.5 \
  --lambda_cl 0.5 --cl_temp 0.05 --cl_convergence 200 --cl_ips_clip 20 \
  --k_attr 5 --lambda_attr 0.05 --gamma_attr 1.0 --deg_thresh 5 \
  --seeds 1 2 3 4 5 \
  --result_path ../results/amazon_fashion_A7.csv
```

### Other datasets (using per-dataset config defaults)

```bash
# Book-Crossing
python run_gbsr_item.py --dataset book_crossing --device_id 0 \
  --seeds 1 2 3 4 5 --result_path ../results/book_crossing.csv

# MovieLens-1M
python run_gbsr_item.py --dataset movielens_1m --device_id 0 \
  --seeds 1 2 3 4 5 --result_path ../results/movielens_1m.csv
```

## Hyperparameter Config System

Each dataset auto-loads `src/configs/hyperparams/<dataset>.json` as default values. Any CLI argument overrides the config. To disable auto-loading:

```bash
python run_gbsr_item.py --dataset amazon_fashion --no_hparam_autoload [other args...]
```

## Key Hyperparameters

| Argument | Description |
|---|---|
| `--attr_graph_mode` | Semantic augmentation: `none`, `A` (two-branch), `B` (tail injection), `C` (symmetric InfoNCE) |
| `--k_attr` | Top-k cosine neighbors in attribute graph |
| `--lambda_attr` | Weight for attribute graph alignment loss |
| `--gamma_attr` | Scale for tail cosine edges added to BPR graph (mode B) |
| `--deg_thresh` | Degree threshold below which items are considered tail |
| `--beta` | Information bottleneck (HSIC) weight |
| `--edge_bias` | Gumbel gate observation bias |
| `--gate_temp` | Temperature for sigmoid gate |
| `--pop_alpha_init` | Initial strength of popularity flow in gate MLP |
| `--lambda_cl` | Weight for infoNCE contrastive loss |
| `--cl_option` | infoNCE weighting: 1=none, 4=Pop(i), 7=(1-IPS) |

## Evaluation

Results are reported as **NDCG@K** and **HR@K** across:
1. **Overall** — all test items
2. **Head / Tail** — top-20% (head) vs. bottom-80% (tail) items by interaction count
3. **5-Group** — five equal popularity bins from tail to head

The CSV at `--result_path` contains per-seed results plus mean/std aggregates.
