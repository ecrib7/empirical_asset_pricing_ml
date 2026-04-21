# GKX (2019) — Empirical Asset Pricing via Machine Learning
### IEOR 4733: Algorithmic Trading — Course Project

This repository implements a **reproduction and extension** of **Gu, Kelly & Xiu (2019)** (*Empirical Asset Pricing via Machine Learning*, RFS): monthly panels, machine-learning return forecasts, recursive out-of-sample evaluation, long–short decile portfolios, and economic significance metrics. It also includes **stub modules** for a modular trading-system layout (YAML-driven experiments, walk-forward engine scaffold, and split `models` / `portfolio` packages) alongside the original GKX pipeline.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with experiment config (stub)

Loads `configs/experiment.yaml` and returns a placeholder result dict (full wiring TBD):

```bash
python main.py --config configs/experiment.yaml
```

---

## Project structure

```
empirical_asset_pricing_ml/
├── main.py                    # CLI: GKX modes + --config YAML stub
├── requirements.txt
├── configs/
│   └── experiment.yaml        # Universe, splits, models, costs, portfolio defaults
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── wrds_loader.py
│   │   └── characteristics.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py            # ModelBase (ABC)
│   │   ├── ols.py             # OLS-3 stub
│   │   ├── elastic_net.py
│   │   ├── pls.py
│   │   ├── random_forest.py
│   │   ├── gbrt.py
│   │   ├── mlp.py
│   │   └── all_models.py      # Legacy GKX estimators (production)
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── construction.py    # Decile / L–S weights (stub)
│   │   ├── costs.py           # Commission + spread + impact (stub)
│   │   └── turnover.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py          # GKX decile backtest + TC (production)
│   │   ├── walkforward_engine.py  # Walk-forward scaffold (deliverable)
│   │   └── simulator.py       # YAML end-to-end driver (stub)
│   ├── evaluation/
│   │   └── metrics.py
│   ├── reporting/
│   └── dashboard/
│       └── app.py
├── data/cache/
├── outputs/
└── logs/
```

---

## WRDS and legacy GKX CLI

### 1. WRDS credentials

```bash
# One-time setup — saves credentials to ~/.pgpass
python -c "import wrds; wrds.Connection()"

# Or set environment variable
export WRDS_USERNAME=your_wrds_username
```

### 2. Goyal & Welch macro data

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
