# CADM

Causal-aware Disentanglement and Modulation for social recommendation.

CADM separates intrinsic interest from social influence and modulates the contribution of social information for each interaction. This repository contains PyTorch implementations for **rating prediction** and **Top-K recommendation**, together with the supplied datasets.

## Architecture

- **SDSI** separates interest and social representations using complementary masks, an orthogonality constraint, and stochastic backdoor adjustment.
- **MGSM** estimates a social-gain proxy and uses user-side and item-side gates to modulate social information, with an alignment objective.
- User–item and social graph encoders provide the input representations.

## Repository structure

```text
CADM/
├── datasets/   # Ciao, Epinions, Dianping, and Douban data
├── model/      # Rating prediction: RMSE and MAE
├── ranking/    # Top-K recommendation: HR@K and NDCG@K
├── requirements.txt
└── README.md
```

Each task has its own `run.py`, `para_parser.py`, `cadm.py`, `sdsi.py`, `mgsm.py`, and graph encoders. Data folders include raw files, processed data, and `mini_data` where provided. The Douban folder contains book-related data.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are listed from source imports; exact package versions have not been pinned. PyTorch selects CUDA when available and otherwise uses CPU.

## Restore large data files

Two large raw/processed data files are stored losslessly as `.gz` files in their original folders to fit the upload interface. Before training or accessing the full processed data, run:

```bash
python prepare_data.py
```

The script restores their original filenames and verifies SHA-256 checksums. All other data files are stored directly.

## Rating prediction

Run from the repository root:

```bash
python model/run.py --dataset ciao \
  --dir_data ./datasets/ciao/mini_data/ \
  --save_model_path ./outputs/rating/ciao/best_model \
  --train_model_weights_path ./outputs/rating/ciao/weights \
  --log_path ./outputs/rating/ciao/train.log
```

## Top-K recommendation

```bash
python ranking/run.py --dataset ciao \
  --dir_data ./datasets/ciao/mini_data/ \
  --save_model_path ./outputs/ranking/ciao/best_model \
  --train_model_weights_path ./outputs/ranking/ciao/weights \
  --log_path ./outputs/ranking/ciao/train.log \
  --top_k 10 --train_neg 1 --test_neg 99
```

Supported dataset arguments are `ciao`, `epinions`, `dianping`, and `douban`. Change both `--dataset` and `--dir_data` when selecting another dataset. Keep the trailing slash in `--dir_data`, since the current loaders concatenate paths directly.

The original default paths point to Google Colab/Google Drive. Override all four path arguments as shown for local execution. The uploaded source files are preserved as supplied.

## Configuration and ablations

Use `python model/run.py --help` or `python ranking/run.py --help` to inspect available options. Key parameters include `--embed_dim`, `--K`, `--lambda1`, `--lambda2`, and `--beta_align`.

- Default: complete CADM model.
- `--ablation wo_mgsm`: disable MGSM.
- `--ablation wo_sdsi`: disable SDSI.

Both entry points support `--resume` for checkpoint loading.

## Data and validation

The repository preserves the supplied data files and splits. Python cache files and macOS metadata are excluded. Source files have been checked for Python syntax; end-to-end training and numerical reproduction have not been verified for this upload.

No software or dataset license is assigned in this repository. Dataset usage remains subject to the original providers' terms. Publication details and a citation entry can be added when finalized.
