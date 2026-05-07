# GKX (2019) — Empirical Asset Pricing via Machine Learning
### IEOR 4733: Algorithmic Trading — Course Project

This repository implements a **reproduction and extension** of **Gu, Kelly & Xiu (2019)** (*Empirical Asset Pricing via Machine Learning*, RFS): monthly panels, machine-learning return forecasts, recursive out-of-sample evaluation, long–short decile portfolios, and economic significance metrics.

## Two pipelines, one CLI flag

| Variant      | Sample period | Macro × char | Industry dummies | Transaction costs | Forecast combination | Regime analysis |
|--------------|---------------|:-:|:-:|---|:-:|:-:|
| `paper`      | 1957 – 2016   | ✅ | ✅ | **0 bps** (gross, matches paper headline) | ✅ | ✅ |
| `improved`   | 1957 – 2024   | ✅ | ✅ | **Impact-aware** (FIM-style) | ✅ | ✅ |

Each variant writes to its own `outputs/<variant>/` directory and uses its own cached feature matrix at `data/cache/feature_matrix_<variant>.parquet` — they don't overwrite each other.

## What's in the improved pipeline

**Impact-aware transaction costs** (Frazzini-Israel-Moskowitz 2018-style). Per-stock per-month cost rate:
```
cost_bps_i = half_spread_bps(log_mcap_i) + λ × √(trade$ / ADV_i)
```
The half-spread is log-linearly interpolated between 25 bps for the smallest-cap decile and 5 bps for the largest, computed against each month's cross-sectional distribution of log market equity. The impact term scales with √(trade dollar / average daily $-volume). ADV is computed as monthly $-volume / 21 (trading days). Implementation: `src/backtest/engine.py::ImpactAwareTransactionCostModel`. The paper variant keeps the legacy flat 0-bps cost.

**Forecast combination.** After per-model training, `--mode evaluate` automatically constructs two ensembles:
- `ENS-AVG` — equal-weighted average of all per-model predictions
- `ENS-MSE` — weighted average with weights ∝ 1 / validation MSE (validation slice = earliest 10% of test dates)

Each ensemble is then routed through the same decile portfolio construction and gets its own row in the comprehensive table, DM matrix, and the rest of the metrics. Skip with `--no-ensembles`.

**Regime-conditional evaluation.** `--mode regimes` slices each model's H-L return series by:
- NBER recession vs expansion (hardcoded recession dates through COVID 2020)
- VIX terciles (low / mid / high implied vol — embedded offline VIX series, override with `--vix-csv`)
- Calendar decade

Outputs `regimes.csv` per variant. Reveals which strategies post similar Sharpes across regimes (more likely true alpha) vs which collapse in recessions (cyclical exposure).

## Other v3 features

- **Comprehensive metrics table** — Sharpe (net), Sharpe (gross), SR\* (Campbell-Thompson), Max DD, Skew, Kurtosis, OOS R², Mean Turnover, Alpha, t(α). One row per model + ensemble. Saved as `outputs/<variant>/comprehensive.csv`.
- **DM with p-values** — `dm_table.csv` (statistic) **and** `dm_pvalues.csv` (two-sided).
- **Variable importance** — `--mode importance` fits each model on train+val, computes GKX-style zero-set importance, aggregates 920 Kronecker features back to 94 base characteristics. `outputs/<variant>/var_importance.csv`.
- **Streamlit dashboard** — variant selector, comprehensive metrics, DM heatmaps (stat + p-value), portfolio returns, transaction-cost sensitivity, **Forecast Combination** tab, **Regimes** tab, variable importance, paper-vs-improved comparison.

## Quick start

```bash
# Reproduce the paper (1957-2016, no TC)
python main.py --mode data-only --variant paper --wrds-username YOUR_USER
python main.py --mode train     --variant paper --models OLS-3 ENet+H PCR PLS GLM+H
python main.py --mode train     --variant paper --models RF GBRT+H
python main.py --mode train     --variant paper --models NN1 NN2 NN3 NN4 NN5
python main.py --mode evaluate  --variant paper        # builds ENS-AVG, ENS-MSE
python main.py --mode regimes   --variant paper        # NBER, VIX, decades
python main.py --mode importance --variant paper --models OLS-3 ENet+H PCR PLS GBRT+H

# Improved pipeline (1957-2024, impact-aware TC)
python main.py --mode data-only  --variant improved --wrds-username YOUR_USER
python main.py --mode train      --variant improved --models OLS-3 ENet+H PCR PLS GBRT+H RF NN1 NN2 NN3 NN4 NN5
python main.py --mode evaluate   --variant improved
python main.py --mode regimes    --variant improved
python main.py --mode importance --variant improved --models OLS-3 ENet+H PCR PLS GBRT+H

# Launch the dashboard (variant selector in sidebar)
streamlit run src/dashboard/app.py
```

The Colab notebook `notebooks/empirical_asset_pricing_ml.ipynb` drives the same flow with cells for runtime restarts and Drive backups.

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
