"""
main.py
-------
Entry-point for the GKX (2019) replication pipeline.

Usage
-----
    # Full pipeline (requires WRDS credentials):
    python main.py --mode full --wrds-username your_username

    # Use cached data / synthetic data for testing:
    python main.py --mode test

    # Dashboard only (after running backtest):
    python main.py --mode dashboard
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from src.config import FREQ_MONTH_END, FREQ_YEAR_START

# ── Create required directories before anything else ──────────────────────────
for _d in ("logs", "data/cache", "outputs", "outputs/models"):
    Path(_d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic data generator  (for testing without WRDS access)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(
    n_stocks: int = 500,
    start: str = "1957-03-01",
    end:   str = "2016-12-31",
    n_chars: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a synthetic monthly panel resembling the GKX dataset.
    Useful for unit testing and CI without WRDS access.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq=FREQ_MONTH_END)

    rows = []
    for permno in range(1, n_stocks + 1):
        # Persistent characteristics
        chars = rng.standard_normal((n_chars,)) * 0.5
        betas = rng.standard_normal((n_chars,)) * 0.02

        for t in dates:
            chars = 0.95 * chars + rng.standard_normal((n_chars,)) * 0.31
            chars = np.clip(chars, -1, 1)
            # Return has signal from characteristics + noise
            signal = chars @ betas + rng.standard_normal() * 0.05
            row = {
                "permno": permno,
                "date":   t,
                "ret":    signal,
                "me":     np.exp(rng.uniform(3, 12)),
                "siccd":  str(rng.integers(10, 99)).zfill(2) + "00",
            }
            for j, c in enumerate(chars):
                row[f"char_{j:02d}_const"] = c
                row[f"char_{j:02d}_dp"]    = c * rng.standard_normal()
                row["mvel1_const"] = chars[0]
                row["bm_const"] = chars[1]
                row["mom12m_const"] = chars[2]
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Synthetic dataset: {len(df):,} obs × {len(df.columns)} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Full WRDS pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(args) -> dict:
    from src.data.wrds_loader import WRDSLoader, merge_crsp_compustat
    from src.data.characteristics import CharacteristicsBuilder, build_feature_matrix
    from src.models.all_models import get_all_models
    from src.backtest.engine import BacktestEngine

    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw data from WRDS ──────────────────────────────────────────
    logger.info("=== Step 1: Loading WRDS data ===")
    loader = WRDSLoader(
        wrds_username=args.wrds_username,
        cache_dir="data/cache/",
    )
    crsp   = loader.get_crsp_monthly()
    comp_a = loader.get_compustat_annual()
    comp_q = loader.get_compustat_quarterly()
    link   = loader.get_crsp_compustat_link()
    macro  = loader.get_macro_predictors(
        goyal_csv_path=args.goyal_csv if hasattr(args, "goyal_csv") else None
    )
    loader.close()

    # ── 2. Merge Compustat onto CRSP ────────────────────────────────────────
    logger.info("=== Step 2: Merging CRSP + Compustat ===")
    panel = merge_crsp_compustat(crsp, comp_a, link, lag_months=6)

    # ── 3. Build characteristics ────────────────────────────────────────────
    logger.info("=== Step 3: Building characteristics ===")
    # Market return for beta / idiovol
    mkt_ret = (
        crsp.assign(wret=lambda x: x["ret"] * x["me"].shift(1))
            .groupby("date")
            .apply(lambda g: g["wret"].sum() / g["me"].shift(1).sum())
            .rename("mkt_ret")
    )
    builder = CharacteristicsBuilder(panel, mkt_ret)
    char_panel = builder.build()

    # ── 4. Build feature matrix (Kronecker product) ────────────────────────
    logger.info("=== Step 4: Building feature matrix ===")
    char_cols = builder._get_char_cols(char_panel)
    feature_matrix = build_feature_matrix(char_panel, macro, char_cols)
    feature_matrix.to_parquet("data/cache/feature_matrix.parquet", index=False)
    logger.info(f"Feature matrix: {feature_matrix.shape}")

    return _run_backtest(feature_matrix, args)


def run_test_pipeline(args) -> dict:
    """Run pipeline with synthetic data (no WRDS needed)."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    logger.info("=== Running TEST PIPELINE with synthetic data ===")
    feature_matrix = generate_synthetic_data(n_stocks=200)
    feature_matrix.to_parquet("data/cache/feature_matrix.parquet", index=False)
    return _run_backtest(feature_matrix, args)


def run_from_cache(args) -> dict:
    """Load cached feature matrix and run backtest."""
    cache = Path("data/cache/feature_matrix.parquet")
    if not cache.exists():
        raise FileNotFoundError(
            "No cached feature matrix found. Run --mode full or --mode test first."
        )
    feature_matrix = pd.read_parquet(cache)
    return _run_backtest(feature_matrix, args)


def _run_backtest(feature_matrix: pd.DataFrame, args) -> dict:
    from src.models.all_models import get_all_models
    from src.backtest.engine import BacktestEngine
    from src.evaluation.metrics import ModelEvaluator

    # ── 5. Initialise models ────────────────────────────────────────────────
    logger.info("=== Step 5: Initialising models ===")
    nn_kwargs = {"batch_size": 10000, "max_epochs": 100, "patience": 5, "n_ensemble": 10}
    models = get_all_models(nn_kwargs=nn_kwargs)
    if hasattr(args, "models") and args.models:
        models = {k: v for k, v in models.items() if k in args.models}
    logger.info(f"Models: {list(models.keys())}")

    # ── 6. Backtest ─────────────────────────────────────────────────────────
    logger.info("=== Step 6: Running recursive backtest ===")
    engine = BacktestEngine(
        train_start=args.train_start if hasattr(args, "train_start") else "1957-03-01",
        val_start  =args.val_start   if hasattr(args, "val_start")   else "1975-01-01",
        val_end    =args.val_end     if hasattr(args, "val_end")     else "1986-12-31",
        test_start =args.test_start  if hasattr(args, "test_start")  else "1987-01-01",
        test_end   =args.test_end    if hasattr(args, "test_end")    else "2016-12-31",
        n_deciles=10,
        weighting="value",
        tc_bps=float(args.tc_bps) if hasattr(args, "tc_bps") else 10.0,
    )

    results = engine.run(feature_matrix, models)

    # ── 7. Evaluate ─────────────────────────────────────────────────────────
    logger.info("=== Step 7: Evaluating ===")
    evaluator = ModelEvaluator(
        y_true=results["true_returns"],
        predictions=results["predictions"],
        dates=results["test_dates"],
        portfolio_returns=results["portfolio_returns"],
    )

    r2_table  = evaluator.oos_r2_table()
    sr_table  = evaluator.sharpe_table()
    dm_matrix = evaluator.dm_table()

    logger.info("\n" + "="*60)
    logger.info("OOS R² Table (%):")
    logger.info("\n" + r2_table.to_string())
    logger.info("\nH-L Sharpe Ratios:")
    logger.info("\n" + sr_table.to_string())

    # ── 8. Save outputs ─────────────────────────────────────────────────────
    logger.info("=== Step 8: Saving outputs ===")
    Path("outputs").mkdir(parents=True, exist_ok=True)

    r2_table.to_csv("outputs/oos_r2.csv")
    sr_table.to_csv("outputs/sharpe_table.csv")
    dm_matrix.to_csv("outputs/dm_table.csv")

    # Save metrics as JSON
    with open("outputs/metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=2, default=str)

    # Save portfolio returns
    port_data = {}
    for model, deciles in results["portfolio_returns"].items():
        port_data[model] = {k: v.to_dict() for k, v in deciles.items()}

    with open("outputs/portfolio_returns.pkl", "wb") as f:
        pickle.dump(results["portfolio_returns"], f)

    logger.info("Outputs saved to outputs/")
    results["evaluator"]  = evaluator
    results["r2_table"]   = r2_table
    results["sr_table"]   = sr_table
    results["dm_matrix"]  = dm_matrix
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="GKX (2019) Empirical Asset Pricing via Machine Learning"
    )
    parser.add_argument(
        "--mode", choices=["full", "test", "cache", "dashboard"],
        default="test",
        help="'full' requires WRDS; 'test' uses synthetic data; "
             "'cache' reuses saved feature matrix; 'dashboard' launches Streamlit",
    )
    parser.add_argument("--wrds-username", default=os.environ.get("WRDS_USERNAME", ""))
    parser.add_argument("--goyal-csv", default=None,
                        help="Path to Welch & Goyal PredictorData CSV/XLSX")
    parser.add_argument("--train-start", default="1957-03-01")
    parser.add_argument("--val-start",   default="1975-01-01")
    parser.add_argument("--val-end",     default="1986-12-31")
    parser.add_argument("--test-start",  default="1987-01-01")
    parser.add_argument("--test-end",    default="2016-12-31")
    parser.add_argument("--tc-bps",      default=10.0, type=float,
                        help="Transaction cost in bps (one-way)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models to run (e.g. OLS-3 ENet+H RF NN3)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "full":
        results = run_full_pipeline(args)
    elif args.mode == "test":
        results = run_test_pipeline(args)
    elif args.mode == "cache":
        results = run_from_cache(args)
    elif args.mode == "dashboard":
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "streamlit", "run",
                        "src/dashboard/app.py"])
        return

    logger.info("Pipeline complete.")
    return results


if __name__ == "__main__":
    main()
