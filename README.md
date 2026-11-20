# AI651 – Deep Learning for Space, Time and Graphs · Assignment 3

**Danish Farhan (25280066)** · LUMS, Fall 2026

Spatio-temporal traffic-flow forecasting on **LargeST – San Diego (716 sensors, 2020–2021)** with a
lightweight STGNN (temporal CNN → graph convolution → MLP head), comparing no graph, the fixed
road-distance graph, a learned adaptive graph and a hybrid graph. Questions 1–2 are small NumPy studies.

## Repository layout

| Path | Contents |
|---|---|
| `data/largest/` | LargeST – San Diego (716 sensors, 2020–2021) => ca_meta.csv, ca_rn_adj.npy, ca_his_raw_2020.h5, ca_his_raw_2021.h5 |
| `figs/` | output figures for questions 3, 5 and 6 |
| `Q1_&_Q2.py` | Q1 & Q2 in one python executable |
| `DL4STG_PA3_Q3-Q6.ipynb` | Q3–Q6 as one notebook, to be run top to bottom |

## Reproducing the experiments

1. `pip install -r requirements.txt` (install a CUDA build of PyTorch for GPU training).
2. Download LargeST from Kaggle (`liuxu77/largest`) and place `ca_meta.csv`, `ca_rn_adj.npy`,
   `ca_his_raw_2020.h5`, `ca_his_raw_2021.h5` in one folder; set `DATA_DIR` in the first Q3 cell.
3. Q1/Q2: run `Q1_&_Q2.py`: Q1(b): base-(K+1) message encoding, exhaustive check for K = 8 + Q1(c): brute-force check of the sum / mean / max table + Q2: least-squares fit, 20-step roll-out heatmaps, directional conv, history model
4. Q3–Q6: run `DL4STG_PA3_Q3-Q6.ipynb` top to bottom. Q3 writes `outputs/sd_15min.npz`; Q4/Q5 train six
   models and write `outputs/res_*.json`, checkpoints and predictions; Q6 builds tables and figures in `figs/`.

## Configuration

| Setting | Value |
|---|---|
| Subset | PeMS District 11 (San Diego), 716 sensors, adjacency rows/cols taken in `ca_meta.csv` order |
| Resampling | 5-min vehicle counts → 15-min counts (3 × mean of available 5-min values = sum) |
| Missing values | causal forward-fill (+ back-fill of leading gaps, training mean for sensors never observed); loss and metrics only on observed targets |
| Splits | train 2020, validation Jan–Mar 2021, test Apr–Dec 2021; windows never cross a boundary |
| Windows | 8 inputs (2 h) → 4 outputs (1 h); z-score with training-set statistics |
| Model | temporal Conv1d(1→32,k3) → Conv1d(32→32,k3,dil 2) → Linear; GraphConv H = ReLU(ÂZW); residual Z+H; MLP head 32→32→4 |
| Fixed graph | Â = D_out⁻¹(A + I) (row-normalised, directed) |
| Adaptive graph | softmax_row(ReLU(E₁E₂ᵀ)), E₁,E₂ ∈ ℝ^{716×10}, dense (no top-k) |
| Hybrid graph | α·Â_fixed + (1−α)·A_adaptive, α = sigmoid(θ) learned, init 0.5 |
| Training | masked MAE, Adam lr 1e-3, weight decay 1e-4 (not on E₁, E₂, θ), batch 64, grad-clip 5, ≤ 40 epochs, early stop patience 6 on val MAE, seed 42 |
| Hardware | NVIDIA GeForce RTX 2060, Intel Core i7-10870H, 16 GB RAM |

## Results (test, Apr–Dec 2021, vehicles / 15 min)

| Model | Adjacency | MAE | RMSE | Training time |
|---|---|---|---|---|
| Temporal-only | None | 64.94 | 102.76 | 298 s |
| STGNN | Fixed | 61.18 | 96.95 | 317 s |
| STGNN | Adaptive | 53.14 | 84.54 | 324 s |
| STGNN | Hybrid | 52.62 | 83.45 | 362 s |
| Ablation | Collapsed adaptive (uniform) | 55.41 | 88.02 | 348 s |
| Ablation | Collapsed hybrid | 55.70 | 88.44 | 361 s |
