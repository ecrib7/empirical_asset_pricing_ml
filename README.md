# GKX (2019) — Empirical Asset Pricing via Machine Learning
### IEOR 4733: Algorithmic Trading — Course Project

Replication of **Gu, Kelly & Xiu (2019)**, NBER Working Paper 25398.

---

## Project Structure

```
ml_asset_pricing/
├── main.py                       # Entry-point: CLI pipeline runner
├── requirements.txt
├── src/
│   ├── config.py                 # All hyper-parameters & sample dates
│   ├── data/
│   │   ├── wrds_loader.py        # WRDS → CRSP + Compustat + Macro
│   │   └── characteristics.py   # 94 firm characteristics + feature matrix
│   ├── models/
│   │   └── all_models.py         # OLS-3, ENet, PCR, PLS, GLM, RF, GBRT, NN1-NN5
│   ├── backtest/
│   │   └── engine.py             # Recursive backtest + TC model + portfolio builder
│   ├── evaluation/
│   │   └── metrics.py            # OOS R², DM tests, Sharpe ratios
│   └── dashboard/
│       └── app.py                # Streamlit interactive dashboard
├── data/cache/                   # Auto-generated Parquet caches
├── outputs/                      # Results: CSVs, pickles, metrics JSON
└── logs/                         # Pipeline logs
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. WRDS credentials

```bash
# One-time setup — saves credentials to ~/.pgpass
python -c "import wrds; wrds.Connection()"

# Or set environment variable
export WRDS_USERNAME=your_wrds_username
```

### 3. Goyal & Welch macro data

Download `PredictorData2023.xlsx` from:
https://sites.google.com/view/agoyal145

---

## Running the Pipeline

### Quick test (synthetic data, no WRDS needed)
```bash
python main.py --mode test --models OLS-3 ENet+H RF NN3
```

### Full replication (requires WRDS)
```bash
python main.py \
  --mode full \
  --wrds-username your_username \
  --goyal-csv data/PredictorData2023.xlsx \
  --tc-bps 10
```

### Use cached features (after first full run)
```bash
python main.py --mode cache --models OLS-3 ENet+H RF GBRT+H NN1 NN2 NN3 NN4 NN5
```

### Launch interactive dashboard
```bash
streamlit run src/dashboard/app.py
# or:
python main.py --mode dashboard
```

---

## Model Overview

| Model     | Type              | GKX Name   |
|-----------|-------------------|------------|
| OLS-3     | Linear            | OLS-3      |
| ENet+H    | Penalised linear  | ENet+H     |
| PCR       | Dim. reduction    | PCR        |
| PLS       | Dim. reduction    | PLS        |
| GLM+H     | Semi-parametric   | GLM+H      |
| RF        | Tree ensemble     | RF         |
| GBRT+H    | Tree ensemble     | GBRT+H     |
| NN1–NN5   | Neural network    | NN1–NN5    |

---

## Key Results (Paper)

| Model | OOS R² (%, monthly) | H-L Sharpe (VW) |
|-------|---------------------|-----------------|
| OLS-3 | 0.16                | 0.61            |
| ENet+H| 0.11                | 0.39            |
| PCR   | 0.26                | 0.88            |
| PLS   | 0.27                | 0.72            |
| RF    | 0.33                | 0.98            |
| GBRT+H| 0.34                | 0.81            |
| NN3   | **0.40**            | **1.20**        |

---

## Features

- ✅ Full recursive backtest with no lookahead bias
- ✅ Training / Validation / Test split (1957–1974 / 1975–1986 / 1987–2016)
- ✅ 94+ firm characteristics (Green et al. 2017)
- ✅ Kronecker feature expansion (chars × macro = 920 signals)
- ✅ All ML models from the paper
- ✅ Huber loss for robust estimation
- ✅ Neural networks with BatchNorm, early stopping, ensemble
- ✅ Diebold-Mariano pairwise model comparison tests
- ✅ Long-short decile portfolios (value & equal weighted)
- ✅ Campbell-Thompson market timing Sharpe improvement
- ✅ Transaction cost sensitivity analysis
- ✅ Interactive Streamlit dashboard
- ✅ WRDS caching (fast re-runs)

---

## Citation

```bibtex
@article{gu2020empirical,
  title={Empirical Asset Pricing via Machine Learning},
  author={Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  journal={The Review of Financial Studies},
  volume={33},
  number={5},
  pages={2223--2273},
  year={2020},
  publisher={Oxford University Press}
}
```
