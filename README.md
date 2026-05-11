# GKX (2019) — Empirical Asset Pricing via Machine Learning
### IEOR 4733: Algorithmic Trading — Course Project

This repository implements a **reproduction and extension** of **Gu, Kelly & Xiu (2019)** (*Empirical Asset Pricing via Machine Learning*, RFS): monthly panels, machine-learning return forecasts, recursive out-of-sample evaluation, long–short decile portfolios, and economic significance metrics.

## Two pipelines, one CLI flag

| Variant              | Sample period                  | Macro × char | Industry dummies | Transaction costs | Forecast combination | Regime analysis |
|----------------------|--------------------------------|:-:|:-:|---|:-:|:-:|
| `paper`              | 1957 – 2016                    | ✅ | ✅ | **0 bps** (gross, matches paper headline) | ✅ | ✅ |
| `improved`           | 1957 – 2024                    | ✅ | ✅ | **Impact-aware** (FIM-style) | ✅ | ✅ |
| `extended_2024`      | 1957 – 2024 (test 2017–2024)   | ✅ | ✅ | **Impact-aware** | ✅ | ✅ |
| `extended_ciz_2026`  | 1957 – 2026Q1 (test 2017–2026Q1) | ✅ | ✅ | **Impact-aware** | ✅ | ✅ |

### Data timeline

* **1957 – 2016** — paper reproduction window.
* **2017 – 2026Q1** — real-data extension via CRSP CIZ/v2. Verified WRDS
  coverage on 2026-05-10:
  - Legacy `crsp.msf.max_date = 2024-12-31` (the user's WRDS subscription
    no longer extends the legacy monthly stock file beyond year-end 2024).
  - CIZ/v2 monthly stock tables reach further:
    `crsp.msf_v2.max(mthcaldt) = 2025-12-31`,
    `crsp.stkmthsecuritydata.max(mthcaldt) = 2025-12-31`,
    `crsp_q_stock.msf_v2.max(mthcaldt) = 2026-03-31`,
    `crsp_q_stock.stkmthsecuritydata.max(mthcaldt) = 2026-03-31`.
  - `comp.funda/fundq.max_datadate = 2026-04-30`,
    `crsp.ccmxpf_linktable.max_linkenddt = 2026-01-30`.

  The CIZ-aware project constant `REAL_DATA_END = 2026-03-31` is the
  boundary the `extended_ciz_2026` variant treats as "real". The legacy
  constant `LEGACY_REAL_DATA_END = 2024-12-31` is preserved for
  `extended_2024` and any caller that must remain reproducible against
  the legacy `crsp.msf` endpoint. The `paper` variant is unaffected.

  Refresh coverage with
  `python scripts/check_wrds_coverage.py --wrds-username <user>`
  (writes `outputs/data_coverage/coverage_latest.json`); add `--dry-run`
  for a credentials-free dump of the last verified coverage. The
  manifest now reports both `legacy_real_data_end` and
  `ciz_real_data_end`, plus the chosen `real_data_end` and
  next-month-end `synthetic_start`.
* **2026Q2+** — *synthetic stress tests only*. Anything strictly after
  `REAL_DATA_END` is generated under explicit regimes
  (`src/synthetic/regimes.py`, CIZ-aware default start
  `SYNTHETIC_START = 2026-04-30`) and is never fed to training.

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

# CIZ-aware extension (1957-2026Q1, impact-aware TC, CRSP CIZ/v2 tables)
# Uses crsp_q_stock.stkmthsecuritydata / msf_v2 (with fallback to
# crsp.stkmthsecuritydata / msf_v2) and maps the CIZ columns
# (mthcaldt → date, mthret → ret, …) back to the legacy schema, so the
# rest of the pipeline is unchanged.
python main.py --mode data-only  --variant extended_ciz_2026 --wrds-username YOUR_USER
python main.py --mode train      --variant extended_ciz_2026 --models OLS-3 ENet+H PCR PLS GBRT+H RF NN1 NN2 NN3 NN4 NN5
python main.py --mode evaluate   --variant extended_ciz_2026
python main.py --mode regimes    --variant extended_ciz_2026

# Launch the dashboard (variant selector in sidebar)
streamlit run src/dashboard/app.py
```

The Colab notebook `notebooks/empirical_asset_pricing_ml.ipynb` drives the same flow with cells for runtime restarts and Drive backups.

### Post-2016 CIZ scoring (Colab, no retrain)

`post2016_ciz` is a lightweight scoring variant. It reuses the per-model
prediction pickles already produced under `outputs/paper/models/` or
`outputs/improved/models/` and slices them to the 2017-01 → 2026-03 window.
Building the full 1957→2026 feature matrix is OOM on a Mac (~3.1M × 432
allocation), but slicing pickles is cheap, and `--mode data-only` for this
variant only spans 2015-01-01 → 2026-03-31 (≈11 years), which is tractable
on a Colab Pro runtime.

```bash
# 1) Mount Drive and restore your existing pickles from
#    `/content/drive/MyDrive/Algo Trading Project/outputs_backup`
#    into ./outputs/ (the notebook does this; or do it manually).

# 2) (Optional but recommended) build a post2016_ciz feature matrix
#    so the predict step can sanity-check (date, permno) coverage:
python main.py --mode data-only --variant post2016_ciz --wrds-username YOUR_USER

# 3) Slice pickles from a source variant into outputs/post2016_ciz/models/.
#    Pick `improved` (covers 1987–2024) for the widest post-2016 overlap.
python main.py --mode predict --variant post2016_ciz \
    --source-model-variant improved \
    --models OLS-3 ENet+H PCR PLS GLM+H GBRT+H NN1 NN2 NN3 NN4

# 4) Run the standard evaluate / regimes / dashboard on the sliced output.
python main.py --mode evaluate --variant post2016_ciz
python main.py --mode regimes  --variant post2016_ciz
```

Caveats:

* Per-model pickles store **predictions**, not fitted model objects. So
  `--mode predict` cannot generate predictions for dates beyond the source
  variant's `test_end` (paper → 2016-11, improved → 2024-11). For
  2025-01 → 2026-03 you must retrain with `--variant extended_ciz_2026`.
* Feature-column compatibility is therefore a non-issue here: we are
  slicing existing predictions, not re-scoring rows against a new feature
  matrix.
* If you pass a `--models` subset, the slicing only emits those names.

---

### Synthetic future scenarios (post-WRDS, 2026-04 → 2036-03)

`generate_synthetic_results.py` produces fully synthetic backtest
artifacts that don't touch WRDS and don't train any real model. It's
used to (a) populate the dashboard when no real data is available and
(b) stress-test the pipeline's downstream stages under hand-crafted
regimes inspired by the [anticor-trader](https://github.com/cvxgrp/anticor-trader)
scenario taxonomy.

For every future-2026 variant the source of truth is a **stock-level
synthetic panel** written to
`data/cache/synthetic_panels/<variant>.parquet`. Each panel covers
exactly 120 month-ends (`2026-04-30 .. 2036-03-31`) and 800 synthetic
permnos (`900000..900799`), for 96,000 rows per scenario. Columns
include the realised return, the latent expected return, a synthetic
market / common factor, market beta, six style characteristics
(`size`, `value`, `momentum`, `quality`, `volatility`, `liquidity`),
and three `model_signal_{strong,medium,weak}` columns that emulate
model predictions at different "skill" levels. The decile returns,
H-L portfolios, turnover, gross/net Sharpes and per-model pickles
that flow into `outputs/<variant>/` are then derived from the panel
by sorting stocks each month on a per-model signal — not from a
decile-only shortcut.

These artifacts are deliberately labelled
`synthetic_training`/`synthetic_evaluation` in the metrics JSON and
per-model pickles. They emulate model output shapes for dashboard
compatibility but **are not real WRDS training results and must not
be presented as forecasts**.

Available scenarios:

| Variant                             | Regime                                          |
|-------------------------------------|-------------------------------------------------|
| `future2026_base`                   | Calibrated baseline continuation                |
| `future2026_trending`               | Persistent leadership / strong momentum         |
| `future2026_mean_reversion`         | Negative autocorrelation / contrarian winners   |
| `future2026_rotating_leaders`       | Decile leadership permutes every 12 months      |
| `future2026_choppy`                 | High noise, low signal                          |
| `future2026_crisis`                 | Correlated drawdown shock + gradual recovery    |
| `future2026_factor_rotation`        | Dominant style/factor sign flips every 18 mo    |

```bash
# Generate one scenario (also writes the stock-level parquet panel)
python generate_synthetic_results.py --variant future2026_base

# Generate all seven future scenarios in one shot
python generate_synthetic_results.py --variant future2026_all

# (Re)generate panel parquets only — skip outputs/
python generate_synthetic_results.py --variant future2026_all --panels-only

# Validate the parquet panels match their scenario before consuming them
python scripts/diagnose_synthetic_panels.py \
    --panel-root data/cache/synthetic_panels \
    --output outputs/synthetic_panel_diagnostics.csv \
    --summary-md outputs/synthetic_panel_diagnostics.md

# Build outputs from an existing panel parquet (fail if missing)
python generate_synthetic_results.py --variant future2026_base --from-panel

# Override panel directory
python generate_synthetic_results.py --variant future2026_all \
    --panel-root data/cache/synthetic_panels

# Also re-generate the post-2016 CIZ-window synthetic baseline
python generate_synthetic_results.py --variant post2016_ciz

# Browse all variants together (dashboard sidebar auto-discovers any
# outputs/<sub>/ directory with a metrics.json).
streamlit run src/dashboard/app.py
```

#### Methodology — stock-level synthetic panels

Each future-2026 panel is drawn from a cross-sectional factor + idio
model with regime-specific knobs (`src/synthetic/panels.py`):

```
ret_{i,t} = β_i · mkt_t + Σ_k char_{i,k,t} · style_return_{k,t} + idio_{i,t}
```

Style returns follow scenario-specific AR(1) paths and the
characteristics drift slowly so the panel is a realistic, regime-aware
fixture rather than independent noise. Each scenario tunes the dials:

* **trending** — high AR(1) on style returns (`ρ=+0.75`) + sticky chars,
  so leadership persists for years.
* **mean_reversion** — negative AR(1) on style returns (`ρ=-0.55`),
  faster char turnover; consecutive months flip winners and losers.
* **rotating_leaders** — every 12 months the per-style premia are
  permuted, so the "winning" mix of styles rotates.
* **choppy** — high idiosyncratic vol, near-zero style premia,
  near-zero persistence.
* **crisis** — broad correlated drawdown of ~22% at month 30 with a
  ~9-month decay back to normal; idio dispersion shrinks during the
  shock month.
* **factor_rotation** — value + momentum premia flip sign every 18
  months; quality / size stay flat.
* **base** — calibrated continuation: modest premia, modest persistence.

The decile portfolios in `outputs/<variant>/portfolio_returns.pkl` and
each `models/<MODEL>.pkl` are produced by sorting the 800 synthetic
permnos each month on a per-model signal (`latent + noise`, with the
noise scale set by a per-model "skill" coefficient). Ensembles and
deep NNs have a higher skill coefficient (≈0.80) and therefore earn
wider H-L spreads — exactly the leaderboard pattern observed on real
data, but produced entirely from synthetic data.

#### Panel diagnostics

Before consuming the generated parquets, validate that each scenario panel
actually exhibits the dynamics implied by its label. The diagnostics
script reads parquets only, never WRDS, and writes a CSV (one row per
scenario) with market moments, drawdown, cross-sectional dispersion,
rank persistence, momentum/reversal spreads, factor correlations, and
scenario-specific warnings (e.g. crisis must have a deep negative
market month and elevated drawdown; choppy must show higher vol than
base; trending must show positive rank persistence and a positive
1-month momentum spread; mean_reversion must show non-positive rank
persistence and a positive reversal spread; rotating_leaders must
show high month-over-month rank churn; factor_rotation must show
sign changes in the value/momentum cross-sectional correlation).

```bash
python generate_synthetic_results.py --variant future2026_all --panels-only
python scripts/diagnose_synthetic_panels.py \
    --panel-root data/cache/synthetic_panels \
    --output outputs/synthetic_panel_diagnostics.csv
# Optional markdown summary
python scripts/diagnose_synthetic_panels.py \
    --panel-root data/cache/synthetic_panels \
    --output outputs/synthetic_panel_diagnostics.csv \
    --summary-md outputs/synthetic_panel_diagnostics.md
# Fail the run (exit 1) if any panel violates its scenario expectations
python scripts/diagnose_synthetic_panels.py \
    --panel-root data/cache/synthetic_panels \
    --output outputs/synthetic_panel_diagnostics.csv --strict
```

Treat the diagnostic CSV as a precondition. If a panel's warning list
is non-empty (`--strict` exits non-zero), do not consume the
downstream `outputs/<variant>/` artifacts — regenerate the panel and
re-run the diagnostic until it passes. Thresholds are conservative
and documented at the top of `scripts/diagnose_synthetic_panels.py`.

Caveats:

* No real returns are produced; each variant is a deterministic synthetic
  draw (seeded by the variant name).
* Scenarios differ qualitatively (drift / momentum / leadership / vol
  regime), not just by RNG seed — `future2026_trending` will show
  persistently positive H-L runs while `future2026_choppy` will not.
* These outputs are for sanity-checking the dashboard and downstream
  reporting under stress regimes; they must NOT be interpreted as
  forecasts or used to evaluate any real model.

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
