"""
main.py
-------
Entry-point for the GKX (2019) replication pipeline.

Usage
-----
    # YAML experiment scaffold (no data pull; returns stub dict):
    python main.py --config configs/experiment.yaml

    # Full pipeline (requires WRDS credentials):
    python main.py --mode full --wrds-username your_username

    # Stage 1: data only (build feature matrix, then stop)
    python main.py --mode data-only --wrds-username brice77 --goyal-csv /content/PredictorData2023.xlsx

    # Stage 2: train models incrementally (restart runtime between groups)
    python main.py --mode train --models OLS-3 ENet+H PCR PLS GLM+H
    python main.py --mode train --models RF GBRT+H
    python main.py --mode train --models NN1 NN2 NN3 NN4 NN5

    # Stage 3: merge all per-model results and produce final tables
    python main.py --mode evaluate

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
#  Data-only pipeline (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def run_data_only(args) -> None:
    """Run all data steps (1→4) in one shot, or a single --data-step."""
    step = getattr(args, "data_step", "all")

    if step == "all" or step == "fetch":
        _data_step_fetch(args)
    if step == "all" or step == "merge":
        _data_step_merge()
    if step == "all" or step == "chars":
        _data_step_chars()
    if step == "all" or step == "features":
        _data_step_features()

    if step == "all":
        logger.info("=== Data-only stage complete (all steps). Run --mode train next. ===")


def _data_step_fetch(args) -> None:
    """Step 1: Fetch raw tables from WRDS and cache as parquet."""
    from src.data.wrds_loader import WRDSLoader

    logger.info("=== Data Step 1/4: Fetching WRDS data ===")
    loader = WRDSLoader(
        wrds_username=args.wrds_username,
        cache_dir="data/cache/",
    )
    loader.get_crsp_monthly()
    loader.get_compustat_annual()
    loader.get_compustat_quarterly()
    loader.get_crsp_compustat_link()
    allow_macro_stub = os.environ.get("GKX_ALLOW_MACRO_STUB", "").strip().lower() in (
        "1", "true", "yes",
    )
    macro = loader.get_macro_predictors(
        goyal_csv_path=args.goyal_csv if hasattr(args, "goyal_csv") else None,
        allow_macro_stub=allow_macro_stub,
    )
    macro.to_parquet("data/cache/macro.parquet", index=False)
    loader.close()
    logger.info("=== Step 1 complete. Cached: crsp, compustat, link, macro ===")


def _data_step_merge() -> None:
    """Step 2: Load cached CRSP + Compustat + link → merged panel."""
    from src.data.wrds_loader import merge_crsp_compustat

    logger.info("=== Data Step 2/4: Merging CRSP + Compustat ===")
    crsp   = pd.read_parquet("data/cache/crsp_monthly_1957_2016.parquet")
    comp_a = pd.read_parquet("data/cache/compustat_annual_1957_2016.parquet")
    link   = pd.read_parquet("data/cache/ccm_link.parquet")

    panel = merge_crsp_compustat(crsp, comp_a, link, lag_months=6)
    panel.to_parquet("data/cache/merged_panel.parquet", index=False)
    logger.info(f"=== Step 2 complete. Merged panel: {panel.shape} ===")


def _data_step_chars() -> None:
    """Step 3: Load merged panel → build characteristics."""
    from src.data.characteristics import CharacteristicsBuilder

    logger.info("=== Data Step 3/4: Building characteristics ===")
    panel = pd.read_parquet("data/cache/merged_panel.parquet")
    crsp  = pd.read_parquet("data/cache/crsp_monthly_1957_2016.parquet")

    mkt_ret = (
        crsp.assign(wret=lambda x: x["ret"] * x["me"].shift(1))
            .groupby("date")
            .apply(lambda g: g["wret"].sum() / g["me"].shift(1).sum())
            .rename("mkt_ret")
    )
    builder = CharacteristicsBuilder(panel, mkt_ret)
    char_panel = builder.build()

    # Save char_panel + the char_cols list so step 4 can reload
    char_panel.to_parquet("data/cache/char_panel.parquet", index=False)
    char_cols = builder._get_char_cols(char_panel)
    with open("data/cache/char_cols.json", "w") as f:
        json.dump(char_cols, f)
    logger.info(f"=== Step 3 complete. Char panel: {char_panel.shape}, {len(char_cols)} chars ===")


def _data_step_features() -> None:
    """Step 4: Load char_panel + macro → Kronecker feature matrix."""
    from src.data.characteristics import build_feature_matrix

    logger.info("=== Data Step 4/4: Building feature matrix ===")
    char_panel = pd.read_parquet("data/cache/char_panel.parquet")
    macro      = pd.read_parquet("data/cache/macro.parquet")
    with open("data/cache/char_cols.json") as f:
        char_cols = json.load(f)

    feature_matrix = build_feature_matrix(char_panel, macro, char_cols)
    feature_matrix.to_parquet("data/cache/feature_matrix.parquet", index=False)
    logger.info(f"=== Step 4 complete. Feature matrix: {feature_matrix.shape} ===")


# ─────────────────────────────────────────────────────────────────────────────
#  Train mode (Stage 2) — incremental per-model saving
# ─────────────────────────────────────────────────────────────────────────────

def run_train(args) -> dict:
    """
    Load cached feature matrix, run backtest for --models subset,
    save each model's results to outputs/models/<name>.pkl so that
    results accumulate across runtime restarts.
    """
    cache = Path("data/cache/feature_matrix.parquet")
    if not cache.exists():
        raise FileNotFoundError(
            "No cached feature matrix. Run --mode data-only or --mode full first."
        )
    feature_matrix = pd.read_parquet(cache)
    return _run_backtest(feature_matrix, args, save_per_model=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluate mode — merge all per-model results and produce tables
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluate(args) -> dict:
    """Load all per-model .pkl files from outputs/models/ and produce tables."""
    from src.evaluation.metrics import ModelEvaluator

    model_dir = Path("outputs/models")
    pkls = sorted(model_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            "No model results found in outputs/models/. Run --mode train first."
        )

    # Merge all per-model results
    predictions = {}
    portfolio_returns = {}
    portfolio_returns_gross = {}
    portfolio_turnover = {}
    metrics = {}
    true_returns = None
    test_dates = None

    for p in pkls:
        with open(p, "rb") as f:
            res = pickle.load(f)
        name = p.stem
        predictions[name] = res["predictions"]
        portfolio_returns[name] = res["portfolio_returns"]
        if "portfolio_returns_gross" in res:
            portfolio_returns_gross[name] = res["portfolio_returns_gross"]
        if "portfolio_turnover" in res:
            portfolio_turnover[name] = res["portfolio_turnover"]
        metrics[name] = res["metrics"]
        # true_returns / test_dates are identical across models
        if true_returns is None:
            true_returns = res["true_returns"]
            test_dates = res["test_dates"]

    logger.info(
        f"Loaded results for {len(predictions)} models: "
        f"{list(predictions.keys())}"
    )

    for name in portfolio_returns:
        portfolio_returns_gross.setdefault(name, portfolio_returns[name])
        portfolio_turnover.setdefault(name, {})

    evaluator = ModelEvaluator(
        y_true=true_returns,
        predictions=predictions,
        dates=test_dates,
        portfolio_returns=portfolio_returns,
    )

    r2_table  = evaluator.oos_r2_table()
    sr_table  = evaluator.sharpe_table()
    dm_matrix = evaluator.dm_table()

    logger.info("\n" + "=" * 60)
    logger.info("OOS R² Table (%):")
    logger.info("\n" + r2_table.to_string())
    logger.info("\nH-L Sharpe Ratios:")
    logger.info("\n" + sr_table.to_string())

    r2_table.to_csv("outputs/oos_r2.csv")
    sr_table.to_csv("outputs/sharpe_table.csv")
    dm_matrix.to_csv("outputs/dm_table.csv")

    from src.reporting.portfolio_io import save_portfolio_bundle

    reporting_meta = {
        "portfolio_pickle_format": "bundle_v1",
        "primary_hl_series": "net_of_engine_transaction_costs",
        "note": (
            "H-L in bundle['net'] reflects engine TC when tc_bps>0. "
            "Use bundle['gross'] and per-model hl_sharpe_gross for pre-cost performance."
        ),
    }
    if metrics:
        sample = next(iter(metrics.values()))
        reporting_meta["hl_engine_tc_bps_default"] = sample.get("hl_engine_tc_bps", 0.0)
        reporting_meta["hl_returns_are_net_of_engine_tc"] = bool(
            sample.get("hl_returns_are_net_of_tc", False)
        )
    metrics_out = dict(metrics)
    metrics_out["_reporting"] = reporting_meta

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    save_portfolio_bundle(
        "outputs/portfolio_returns.pkl",
        portfolio_returns,
        portfolio_returns_gross,
        portfolio_turnover,
    )

    logger.info("All outputs saved to outputs/")
    return {
        "predictions": predictions,
        "true_returns": true_returns,
        "test_dates": test_dates,
        "portfolio_returns": portfolio_returns,
        "metrics": metrics,
        "evaluator": evaluator,
        "r2_table": r2_table,
        "sr_table": sr_table,
        "dm_matrix": dm_matrix,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Full WRDS pipeline (original — data + train + evaluate in one shot)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(args) -> dict:
    from src.data.wrds_loader import WRDSLoader, merge_crsp_compustat
    from src.data.characteristics import CharacteristicsBuilder, build_feature_matrix

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
    allow_macro_stub = os.environ.get("GKX_ALLOW_MACRO_STUB", "").strip().lower() in (
        "1", "true", "yes",
    )
    macro = loader.get_macro_predictors(
        goyal_csv_path=args.goyal_csv if hasattr(args, "goyal_csv") else None,
        allow_macro_stub=allow_macro_stub,
    )
    loader.close()

    # ── 2. Merge Compustat onto CRSP ────────────────────────────────────────
    logger.info("=== Step 2: Merging CRSP + Compustat ===")
    panel = merge_crsp_compustat(crsp, comp_a, link, lag_months=6)

    # ── 3. Build characteristics ────────────────────────────────────────────
    logger.info("=== Step 3: Building characteristics ===")
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

    return _run_backtest(feature_matrix, args, save_per_model=False)


def run_test_pipeline(args) -> dict:
    """Run pipeline with synthetic data (no WRDS needed)."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    logger.info("=== Running TEST PIPELINE with synthetic data ===")
    feature_matrix = generate_synthetic_data(n_stocks=200)
    feature_matrix.to_parquet("data/cache/feature_matrix.parquet", index=False)
    return _run_backtest(feature_matrix, args, save_per_model=False)


def run_from_cache(args) -> dict:
    """Load cached feature matrix and run backtest."""
    cache = Path("data/cache/feature_matrix.parquet")
    if not cache.exists():
        raise FileNotFoundError(
            "No cached feature matrix found. Run --mode full or --mode test first."
        )
    feature_matrix = pd.read_parquet(cache)
    return _run_backtest(feature_matrix, args, save_per_model=False)


def _run_backtest(
    feature_matrix: pd.DataFrame,
    args,
    save_per_model: bool = False,
) -> dict:
    from src.models.all_models import get_all_models
    from src.backtest.engine import BacktestEngine, add_forward_return_target
    from src.evaluation.metrics import ModelEvaluator

    feature_matrix = add_forward_return_target(feature_matrix)

    # ── 5. Initialise models ────────────────────────────────────────────────
    logger.info("=== Step 5: Initialising models ===")
    nn_kwargs = {
        "batch_size": 10000,
        "max_epochs": 100,
        "patience": 5,
        "n_ensemble": 10,
    }
    models = get_all_models(nn_kwargs=nn_kwargs)
    if hasattr(args, "models") and args.models:
        models = {k: v for k, v in models.items() if k in args.models}
    logger.info(f"Models: {list(models.keys())}")

    # ── 6. Backtest ─────────────────────────────────────────────────────────
    logger.info("=== Step 6: Running recursive backtest ===")
    engine = BacktestEngine(
        train_start=getattr(args, "train_start", "1957-03-01"),
        val_start=getattr(args, "val_start", "1975-01-01"),
        val_end=getattr(args, "val_end", "1986-12-31"),
        test_start=getattr(args, "test_start", "1987-01-01"),
        test_end=getattr(args, "test_end", "2016-12-31"),
        n_deciles=10,
        weighting="value",
        tc_bps=float(getattr(args, "tc_bps", 10.0)),
    )

    results = engine.run(feature_matrix, models)

    # ── Save per-model results (for incremental train mode) ─────────────────
    if save_per_model:
        model_dir = Path("outputs/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in models:
            model_result = {
                "predictions":       results["predictions"][name],
                "true_returns":      results["true_returns"],
                "test_dates":        results["test_dates"],
                "test_permnos":      results["test_permnos"],
                "portfolio_returns": results["portfolio_returns"][name],
                "portfolio_returns_gross": results["portfolio_returns_gross"][name],
                "portfolio_turnover": results["portfolio_turnover"][name],
                "metrics":           results["metrics"][name],
            }
            out_path = model_dir / f"{name}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(model_result, f)
            logger.info(f"Saved {name} → {out_path}")

        logger.info("=== Train stage complete. Results saved per-model. ===")
        logger.info(
            "Existing model results in outputs/models/: "
            f"{[p.stem for p in sorted(model_dir.glob('*.pkl'))]}"
        )
        return results

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

    logger.info("\n" + "=" * 60)
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

    from src.reporting.portfolio_io import save_portfolio_bundle

    metrics_out = dict(results["metrics"])
    sample = next(iter(metrics_out.values())) if metrics_out else {}
    metrics_out["_reporting"] = {
        "portfolio_pickle_format": "bundle_v1",
        "primary_hl_series": "net_of_engine_transaction_costs",
        "hl_engine_tc_bps_default": sample.get("hl_engine_tc_bps", 0.0),
        "hl_returns_are_net_of_engine_tc": bool(sample.get("hl_returns_are_net_of_tc", False)),
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    save_portfolio_bundle(
        "outputs/portfolio_returns.pkl",
        results["portfolio_returns"],
        results["portfolio_returns_gross"],
        results["portfolio_turnover"],
    )

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
        "--config",
        default=None,
        help="Path to YAML experiment config (stub: loads via RunSimulation; see configs/experiment.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "test", "cache", "dashboard",
                 "data-only", "train", "evaluate"],
        default="test",
        help=(
            "'full' = WRDS + train + evaluate in one shot; "
            "'data-only' = build feature matrix then stop; "
            "'train' = load cached data, train --models subset, save per-model; "
            "'evaluate' = merge all per-model .pkl results into final tables; "
            "'test' = synthetic data; 'cache' = reuse feature matrix; "
            "'dashboard' = launch Streamlit"
        ),
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
    parser.add_argument("--data-step",
                        choices=["all", "fetch", "merge", "chars", "features"],
                        default="all",
                        help="Which data sub-step to run in data-only mode: "
                             "'all' = run 1→4; 'fetch' = WRDS download; "
                             "'merge' = CRSP+Compustat merge; "
                             "'chars' = build characteristics; "
                             "'features' = Kronecker feature matrix")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        from src.backtest.simulator import RunSimulation

        sim = RunSimulation(args.config)
        out = sim.run()
        logger.info("Config-driven stub result: %s", out)
        return out

    if args.mode == "full":
        results = run_full_pipeline(args)
    elif args.mode == "test":
        results = run_test_pipeline(args)
    elif args.mode == "cache":
        results = run_from_cache(args)
    elif args.mode == "data-only":
        run_data_only(args)
        return
    elif args.mode == "train":
        results = run_train(args)
    elif args.mode == "evaluate":
        results = run_evaluate(args)
    elif args.mode == "dashboard":
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "streamlit", "run",
                        "src/dashboard/app.py"])
        return

    logger.info("Pipeline complete.")
    return results


if __name__ == "__main__":
    main()
