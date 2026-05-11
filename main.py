"""
main.py
-------
Entry-point for the GKX (2019) replication pipeline.

Usage
-----
    # YAML experiment scaffold (no data pull; returns stub dict):
    python main.py --config configs/experiment.yaml

    # Full pipeline (requires WRDS credentials + data/gwz_data_csv_2024.zip):
    python main.py --mode full --wrds-username your_username

    # Stage 1: data only (build feature matrix, then stop)
    python main.py --mode data-only --wrds-username $WRDS_USERNAME

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
from typing import Optional

import numpy as np
import pandas as pd
from src.config import FREQ_MONTH_END, FREQ_YEAR_START, get_variant_config

# ── Create required directories before anything else ──────────────────────────
for _d in ("logs", "data/cache", "outputs",
           "outputs/paper", "outputs/paper/models",
           "outputs/improved", "outputs/improved/models",
           "outputs/models"):
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
#  Variant resolution helper
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_variant(args) -> dict:
    """
    Read --variant from args (default 'paper') and return a fully-resolved
    dict of pipeline settings, with any explicit CLI overrides applied.
    """
    name = getattr(args, "variant", "paper") or "paper"
    cfg = get_variant_config(name)
    cfg["name"] = name

    # Apply CLI-level overrides if the user passed them
    overrides = [
        ("data_start", "data_start"),
        ("data_end",   "data_end"),
        ("train_start","train_start"),
        ("val_start",  "val_start"),
        ("val_end",    "val_end"),
        ("test_start", "test_start"),
        ("test_end",   "test_end"),
        ("tc_bps",     "tc_bps"),
    ]
    for cli, key in overrides:
        v = getattr(args, cli, None)
        if v is not None and v != "":
            cfg[key] = v
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["model_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


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
#  Macro predictors — sourced from data/gwz_data_csv_2024.zip
# ─────────────────────────────────────────────────────────────────────────────

def _build_macro(start: str, end: str) -> pd.DataFrame:
    """Parse the GWZ csv zip into data/cache/macro.parquet and return the DataFrame."""
    from src.data.gwz_macro import build_macro_parquet
    return build_macro_parquet(
        data_dir=Path("data"),
        cache_path=Path("data/cache/macro.parquet"),
        start_date=start,
        end_date=end,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Data-only pipeline (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def run_data_only(args) -> None:
    """Run all data steps (1→4) in one shot, or a single --data-step."""
    cfg = _resolve_variant(args)
    step = getattr(args, "data_step", "all")

    if step == "all" or step == "fetch":
        _data_step_fetch(args, cfg)
    if step == "all" or step == "merge":
        _data_step_merge(cfg)
    if step == "all" or step == "chars":
        _data_step_chars(cfg)
    if step == "all" or step == "features":
        _data_step_features(cfg)

    if step == "all":
        logger.info(
            f"=== Data-only stage complete (variant='{cfg['name']}'). "
            f"Run --mode train next. ==="
        )


def _crsp_data_source(cfg: dict) -> str:
    """
    Decide which CRSP monthly schema to load for a variant.

    The legacy ``crsp.msf`` only reaches 2024-12-31 on the user's WRDS
    subscription, so any variant whose ``data_end`` is strictly after
    ``LEGACY_REAL_DATA_END`` must use the CIZ/v2 path.
    """
    from src.config import LEGACY_REAL_DATA_END
    from src.data.wrds_loader import CIZ_AWARE_VARIANTS

    if cfg["name"] in CIZ_AWARE_VARIANTS:
        return "ciz"
    if cfg["data_end"] > LEGACY_REAL_DATA_END:
        return "ciz"
    return "legacy"


def _crsp_cache_path(cfg: dict) -> Path:
    suffix = "_ciz" if _crsp_data_source(cfg) == "ciz" else ""
    return Path(
        f"data/cache/crsp_monthly{suffix}_"
        f"{cfg['data_start'][:4]}_{cfg['data_end'][:4]}.parquet"
    )


def _comp_a_cache_path(cfg: dict) -> Path:
    return Path(f"data/cache/compustat_annual_{cfg['data_start'][:4]}_{cfg['data_end'][:4]}.parquet")


def _merged_panel_path(cfg: dict) -> Path:
    return Path(f"data/cache/merged_panel_{cfg['name']}.parquet")


def _char_panel_path(cfg: dict) -> Path:
    return Path(f"data/cache/char_panel_{cfg['name']}.parquet")


def _char_cols_path(cfg: dict) -> Path:
    return Path(f"data/cache/char_cols_{cfg['name']}.json")


def _feature_matrix_path(cfg: dict) -> Path:
    return Path(cfg["feature_cache"])


def _data_step_fetch(args, cfg: dict) -> None:
    """Step 1: Fetch raw tables from WRDS and cache as parquet, plus build macro from GWZ zip."""
    from src.data.wrds_loader import WRDSLoader

    data_source = _crsp_data_source(cfg)
    logger.info(
        f"=== Data Step 1/4: Fetching WRDS data "
        f"({cfg['data_start']} → {cfg['data_end']}, source={data_source}) ==="
    )
    loader = WRDSLoader(
        wrds_username=args.wrds_username,
        cache_dir="data/cache/",
        start_date=cfg["data_start"],
        end_date=cfg["data_end"],
        data_source=data_source,
    )
    loader.get_crsp_monthly()
    loader.get_compustat_annual()
    loader.get_compustat_quarterly()
    loader.get_crsp_compustat_link()
    loader.close()

    # Macro predictors come from the GWZ csv zip in data/, not WRDS
    _build_macro(start=cfg["data_start"], end=cfg["data_end"])
    logger.info("=== Step 1 complete. Cached: crsp, compustat, link, macro ===")


def _data_step_merge(cfg: dict) -> None:
    """Step 2: Load cached CRSP + Compustat + link → merged panel."""
    from src.data.wrds_loader import merge_crsp_compustat

    logger.info("=== Data Step 2/4: Merging CRSP + Compustat ===")
    crsp_path = _crsp_cache_path(cfg)
    comp_path = _comp_a_cache_path(cfg)
    if not crsp_path.exists() or not comp_path.exists():
        raise FileNotFoundError(
            f"Cached CRSP/Compustat not found at {crsp_path} / {comp_path}. "
            "Run --data-step fetch first."
        )
    crsp   = pd.read_parquet(crsp_path)
    comp_a = pd.read_parquet(comp_path)
    link   = pd.read_parquet("data/cache/ccm_link.parquet")

    panel = merge_crsp_compustat(crsp, comp_a, link, lag_months=6)
    out = _merged_panel_path(cfg)
    panel.to_parquet(out, index=False)
    logger.info(f"=== Step 2 complete. Merged panel: {panel.shape} → {out} ===")


def _data_step_chars(cfg: dict) -> None:
    """Step 3: Load merged panel → build characteristics."""
    from src.data.characteristics import CharacteristicsBuilder

    logger.info("=== Data Step 3/4: Building characteristics ===")
    panel = pd.read_parquet(_merged_panel_path(cfg))
    crsp  = pd.read_parquet(_crsp_cache_path(cfg))

    mkt_ret = (
        crsp.assign(wret=lambda x: x["ret"] * x["me"].shift(1))
            .groupby("date")
            .apply(lambda g: g["wret"].sum() / g["me"].shift(1).sum())
            .rename("mkt_ret")
    )
    builder = CharacteristicsBuilder(panel, mkt_ret)
    char_panel = builder.build()

    char_panel.to_parquet(_char_panel_path(cfg), index=False)
    char_cols = builder._get_char_cols(char_panel)
    with open(_char_cols_path(cfg), "w") as f:
        json.dump(char_cols, f)
    logger.info(
        f"=== Step 3 complete. Char panel: {char_panel.shape}, "
        f"{len(char_cols)} chars ==="
    )


def _data_step_features(cfg: dict) -> None:
    """Step 4: Load char_panel + macro → Kronecker feature matrix."""
    from src.data.characteristics import build_feature_matrix

    logger.info("=== Data Step 4/4: Building feature matrix "
                f"(macro_interactions={cfg['use_macro_interactions']}) ===")
    char_panel = pd.read_parquet(_char_panel_path(cfg))
    macro      = pd.read_parquet("data/cache/macro.parquet")
    with open(_char_cols_path(cfg)) as f:
        char_cols = json.load(f)

    if cfg["use_macro_interactions"]:
        feature_matrix = build_feature_matrix(char_panel, macro, char_cols)
    else:
        # No macro interactions: only the constant block of features
        # (= raw characteristics) and SIC dummies if enabled.
        feature_matrix = build_feature_matrix(
            char_panel, macro, char_cols, macro_cols=[]
        )

    if not cfg["use_industry_dummies"]:
        feature_matrix = feature_matrix.loc[
            :, [c for c in feature_matrix.columns if not c.startswith("sic2_")]
        ]

    # ── Attach raw $-volume for impact-aware TC. ADV = monthly $-volume / 21.
    # We carry it on the feature matrix as ``adv_dollar`` so the engine can
    # pivot to a wide frame at evaluation time without re-loading char_panel.
    if "prc" in char_panel.columns and "vol" in char_panel.columns:
        adv = (
            char_panel[["permno", "date", "prc", "vol"]]
            .assign(adv_dollar=lambda d: (d["prc"].abs() * d["vol"] * 1000.0 / 21.0))
            [["permno", "date", "adv_dollar"]]
        )
        feature_matrix = feature_matrix.merge(adv, on=["permno", "date"], how="left")
        logger.info(f"Attached adv_dollar (monthly$/21). "
                    f"Coverage: {feature_matrix['adv_dollar'].notna().mean():.1%}")
    else:
        logger.info("prc/vol not in char_panel — adv_dollar not attached "
                    "(impact-aware TC will fall back to flat fallback bps).")

    out = _feature_matrix_path(cfg)
    feature_matrix.to_parquet(out, index=False)
    logger.info(
        f"=== Step 4 complete. Feature matrix: {feature_matrix.shape} → {out} ==="
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Train mode (Stage 2) — incremental per-model saving
# ─────────────────────────────────────────────────────────────────────────────

def run_train(args) -> dict:
    """
    Load cached feature matrix, run backtest for --models subset,
    save each model's results to outputs/<variant>/models/<name>.pkl
    so that results accumulate across runtime restarts.
    """
    cfg = _resolve_variant(args)
    cache = _feature_matrix_path(cfg)
    if not cache.exists():
        raise FileNotFoundError(
            f"No cached feature matrix at {cache}. "
            f"Run --mode data-only --variant {cfg['name']} first."
        )
    feature_matrix = pd.read_parquet(cache)
    return _run_backtest(feature_matrix, args, cfg, save_per_model=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluate mode — merge all per-model results and produce tables
# ─────────────────────────────────────────────────────────────────────────────

def _build_portfolios_for_predictions(
    predictions: dict,
    true_returns,
    test_dates,
    test_permnos,
    cfg: dict,
) -> dict:
    """
    Given a dict of {name: 1D pred array}, construct decile portfolios and
    return per-name {"net","gross","turnover","metrics"}. Used by
    ``run_evaluate`` to add ensemble portfolios after merging per-model
    pickles. Re-uses the variant's transaction cost configuration so the
    ensemble portfolios are scored on the same TC basis as the constituents.
    """
    import numpy as np
    from src.backtest.engine import (
        DecilePortfolioBuilder, TransactionCostModel,
        ImpactAwareTransactionCostModel,
        StockLevelImpactCostModel,
    )
    from src.evaluation.metrics import oos_r2, sharpe_ratio

    # Reload feature matrix to get me/adv at test rows
    fm = pd.read_parquet(_feature_matrix_path(cfg))
    fm_idx = fm.set_index(["date", "permno"])
    keys = list(zip(pd.to_datetime(test_dates), test_permnos))
    me_vals = (
        fm_idx["me"].reindex(keys).values if "me" in fm.columns
        else np.ones(len(test_dates))
    )
    adv_vals = (
        fm_idx["adv_dollar"].reindex(keys).values
        if "adv_dollar" in fm.columns else None
    )

    # Build TC model identical to the one engine.run used
    if cfg.get("tc_model") == "stock_level":
        tc_model = StockLevelImpactCostModel(
            vol_spread_bps    = float(cfg.get("tc_vol_spread_bps",   8.0)),
            vol_impact_scale  = float(cfg.get("tc_vol_impact_scale", 0.4)),
            nav_billions      = float(cfg.get("tc_nav_billions",     1.0)),
            fallback_bps      = float(cfg.get("tc_bps", 10.0)),
        )
    elif cfg.get("tc_model") in ("impact", "stock_level"):
        tc_model = ImpactAwareTransactionCostModel()
    else:
        tc_model = TransactionCostModel(cost_bps=float(cfg["tc_bps"]))

    out = {}
    for name, pred in predictions.items():
        df = pd.DataFrame({
            "date":   pd.to_datetime(test_dates),
            "permno": test_permnos,
            "pred":   pred,
            "ret":    true_returns,
            "me":     me_vals,
        })
        if adv_vals is not None:
            df["adv"] = adv_vals
        pred_wide = df.pivot(index="date", columns="permno", values="pred")
        ret_wide  = df.pivot(index="date", columns="permno", values="ret")
        me_wide   = df.pivot(index="date", columns="permno", values="me")
        adv_wide  = (df.pivot(index="date", columns="permno", values="adv")
                     if adv_vals is not None else None)
        builder = DecilePortfolioBuilder(
            n_deciles=10, weighting="value", tc_model=tc_model,
        )
        net, gross, turn = builder.build(pred_wide, ret_wide, me_wide, adv=adv_wide)

        # Quick metrics
        valid = ~np.isnan(pred) & ~np.isnan(np.asarray(true_returns, dtype=float))
        r2 = oos_r2(np.asarray(true_returns)[valid], np.asarray(pred)[valid]) * 100
        hl_n = net.get("H-L", pd.Series(dtype=float)).dropna()
        hl_g = gross.get("H-L", pd.Series(dtype=float)).dropna()
        hl_t = turn.get("H-L", pd.Series(dtype=float)).dropna()
        sr_n = sharpe_ratio(hl_n) if len(hl_n) else float("nan")
        sr_g = sharpe_ratio(hl_g) if len(hl_g) else float("nan")
        to_m = float(hl_t.mean()) if len(hl_t) else float("nan")
        out[name] = {
            "net": net, "gross": gross, "turnover": turn,
            "metrics": {
                "oos_r2_pct": round(r2, 3),
                "hl_sharpe": round(sr_n, 3) if sr_n == sr_n else float("nan"),
                "hl_sharpe_gross": round(sr_g, 3) if sr_g == sr_g else float("nan"),
                "hl_mean_turnover_one_way": round(to_m, 6) if to_m == to_m else float("nan"),
                "hl_engine_tc_bps": cfg["tc_bps"],
                "hl_returns_are_net_of_tc": cfg["tc_bps"] > 0 or cfg.get("tc_model") in ("impact", "stock_level"),
                "is_ensemble": True,
            },
        }
    return out


def run_evaluate(args) -> dict:
    """Load all per-model .pkl files from outputs/<variant>/models/ and produce tables."""
    from src.evaluation.metrics import (
        ModelEvaluator, dm_table_full, sharpe_ratio,
    )

    cfg = _resolve_variant(args)
    out_dir = Path(cfg["output_dir"])
    model_dir = Path(cfg["model_dir"])
    pkls = sorted(model_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"No model results found in {model_dir}. "
            f"Run --mode train --variant {cfg['name']} first."
        )

    # Merge all per-model results
    predictions = {}
    portfolio_returns = {}
    portfolio_returns_gross = {}
    portfolio_turnover = {}
    metrics = {}
    true_returns = None
    test_dates = None
    test_permnos = None

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
        if true_returns is None:
            true_returns = res["true_returns"]
            test_dates = res["test_dates"]
            test_permnos = res.get("test_permnos")

    logger.info(
        f"[variant={cfg['name']}] Loaded results for {len(predictions)} models: "
        f"{list(predictions.keys())}"
    )

    # ── Forecast combinations: ENS-AVG and ENS-MSE ──────────────────────────
    skip_ensembles = bool(getattr(args, "no_ensembles", False))
    ens_meta: dict = {}
    if not skip_ensembles and len(predictions) >= 2:
        from src.evaluation.combinations import build_ensembles
        predictions, ens_meta = build_ensembles(
            predictions, y_true=true_returns, dates=test_dates,
            which=("avg", "mse"),
        )
        # Build portfolios for each new ensemble using the same engine logic
        ens_names = [n for n in predictions if n in ("ENS-AVG", "ENS-MSE")]
        if ens_names:
            ens_pf = _build_portfolios_for_predictions(
                {n: predictions[n] for n in ens_names},
                true_returns=true_returns,
                test_dates=test_dates,
                test_permnos=test_permnos,
                cfg=cfg,
            )
            for n, pf in ens_pf.items():
                portfolio_returns[n]       = pf["net"]
                portfolio_returns_gross[n] = pf["gross"]
                portfolio_turnover[n]      = pf["turnover"]
                metrics[n] = pf["metrics"]

    for name in portfolio_returns:
        portfolio_returns_gross.setdefault(name, portfolio_returns[name])
        portfolio_turnover.setdefault(name, {})

    evaluator = ModelEvaluator(
        y_true=true_returns,
        predictions=predictions,
        dates=test_dates,
        portfolio_returns=portfolio_returns,
    )

    r2_table   = evaluator.oos_r2_table()
    sr_table   = evaluator.sharpe_table()
    dm_stats, dm_pvals = dm_table_full(true_returns, predictions, test_dates)

    # Equal-weighted "market" proxy: cross-sectional mean realised return per date
    mkt_proxy = (
        pd.DataFrame({"date": test_dates, "ret": true_returns})
          .groupby("date")["ret"].mean()
    )

    comprehensive = evaluator.comprehensive_table(
        portfolio_returns_gross=portfolio_returns_gross,
        portfolio_turnover=portfolio_turnover,
        market_factor=mkt_proxy,
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"[variant={cfg['name']}] Comprehensive performance:")
    logger.info("\n" + comprehensive.to_string())

    # ── Save outputs ────────────────────────────────────────────────────────
    r2_table.to_csv(out_dir / "oos_r2.csv")
    sr_table.to_csv(out_dir / "sharpe_table.csv")
    dm_stats.to_csv(out_dir / "dm_table.csv")
    dm_pvals.to_csv(out_dir / "dm_pvalues.csv")
    comprehensive.to_csv(out_dir / "comprehensive.csv")

    from src.reporting.portfolio_io import save_portfolio_bundle

    reporting_meta = {
        "variant": cfg["name"],
        "tc_bps": cfg["tc_bps"],
        "tc_model": cfg.get("tc_model", "flat"),
        "data_start": cfg["data_start"],
        "data_end": cfg["data_end"],
        "test_start": cfg["test_start"],
        "test_end": cfg["test_end"],
        "use_macro_interactions": cfg["use_macro_interactions"],
        "use_industry_dummies": cfg["use_industry_dummies"],
        "portfolio_pickle_format": "bundle_v1",
        "primary_hl_series": "net_of_engine_transaction_costs",
        "hl_engine_tc_bps_default": cfg["tc_bps"],
        "hl_returns_are_net_of_engine_tc": cfg["tc_bps"] > 0 or cfg.get("tc_model") in ("impact", "stock_level"),
    }
    metrics_out = dict(metrics)
    metrics_out["_reporting"] = reporting_meta
    if ens_meta:
        metrics_out["_ensembles"] = ens_meta

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    save_portfolio_bundle(
        out_dir / "portfolio_returns.pkl",
        portfolio_returns,
        portfolio_returns_gross,
        portfolio_turnover,
    )

    logger.info(f"All outputs saved to {out_dir}/")
    return {
        "predictions": predictions,
        "true_returns": true_returns,
        "test_dates": test_dates,
        "test_permnos": test_permnos,
        "portfolio_returns": portfolio_returns,
        "metrics": metrics,
        "evaluator": evaluator,
        "r2_table": r2_table,
        "sr_table": sr_table,
        "dm_matrix": dm_stats,
        "dm_pvalues": dm_pvals,
        "comprehensive": comprehensive,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Full WRDS pipeline — data + train + evaluate in one shot (variant-aware)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(args) -> dict:
    from src.data.wrds_loader import WRDSLoader, merge_crsp_compustat
    from src.data.characteristics import CharacteristicsBuilder, build_feature_matrix

    cfg = _resolve_variant(args)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    Path(cfg["model_dir"]).mkdir(parents=True, exist_ok=True)

    variant = cfg["name"]
    data_source = _crsp_data_source(cfg)
    logger.info(
        f"=== Step 1: Loading WRDS data (variant={variant!r}, "
        f"source={data_source}) ==="
    )
    loader = WRDSLoader(
        wrds_username=args.wrds_username,
        cache_dir="data/cache/",
        start_date=cfg["data_start"],
        end_date=cfg["data_end"],
        data_source=data_source,
    )
    crsp   = loader.get_crsp_monthly()
    comp_a = loader.get_compustat_annual()
    _      = loader.get_compustat_quarterly()
    link   = loader.get_crsp_compustat_link()
    loader.close()

    macro = _build_macro(start=cfg["data_start"], end=cfg["data_end"])

    logger.info("=== Step 2: Merging CRSP + Compustat ===")
    panel = merge_crsp_compustat(crsp, comp_a, link, lag_months=6)

    logger.info("=== Step 3: Building characteristics ===")
    mkt_ret = (
        crsp.assign(wret=lambda x: x["ret"] * x["me"].shift(1))
            .groupby("date")
            .apply(lambda g: g["wret"].sum() / g["me"].shift(1).sum())
            .rename("mkt_ret")
    )
    builder = CharacteristicsBuilder(panel, mkt_ret)
    char_panel = builder.build()

    logger.info("=== Step 4: Building feature matrix ===")
    char_cols = builder._get_char_cols(char_panel)
    macro_cols_arg = None if cfg["use_macro_interactions"] else []
    feature_matrix = build_feature_matrix(
        char_panel, macro, char_cols, macro_cols=macro_cols_arg
    )
    if not cfg["use_industry_dummies"]:
        feature_matrix = feature_matrix.loc[
            :, [c for c in feature_matrix.columns if not c.startswith("sic2_")]
        ]
    feature_matrix.to_parquet(_feature_matrix_path(cfg), index=False)
    logger.info(f"Feature matrix: {feature_matrix.shape}")

    return _run_backtest(feature_matrix, args, cfg, save_per_model=False)


def run_test_pipeline(args) -> dict:
    """Run pipeline with synthetic data (no WRDS needed)."""
    cfg = _resolve_variant(args)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    logger.info("=== Running TEST PIPELINE with synthetic data ===")
    feature_matrix = generate_synthetic_data(
        n_stocks=200,
        start=cfg["data_start"],
        end=cfg["data_end"],
    )
    feature_matrix.to_parquet(_feature_matrix_path(cfg), index=False)
    return _run_backtest(feature_matrix, args, cfg, save_per_model=False)


def run_from_cache(args) -> dict:
    """Load cached feature matrix and run backtest."""
    cfg = _resolve_variant(args)
    cache = _feature_matrix_path(cfg)
    if not cache.exists():
        raise FileNotFoundError(
            f"No cached feature matrix at {cache}. "
            f"Run --mode full or --mode test --variant {cfg['name']} first."
        )
    feature_matrix = pd.read_parquet(cache)
    return _run_backtest(feature_matrix, args, cfg, save_per_model=False)


def _run_backtest(
    feature_matrix: pd.DataFrame,
    args,
    cfg: dict,
    save_per_model: bool = False,
) -> dict:
    from src.models.all_models import get_all_models
    from src.backtest.engine import BacktestEngine, add_forward_return_target
    from src.evaluation.metrics import ModelEvaluator, dm_table_full

    feature_matrix = add_forward_return_target(feature_matrix)

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

    if not getattr(args, "force_retrain", False):
        model_dir = Path(cfg["model_dir"])
        already_done = {p.stem for p in model_dir.glob("*.pkl")}
        skipped = [k for k in models if k in already_done]
        if skipped:
            logger.info(f"Skipping already-trained models: {skipped} "
                        f"(use --force-retrain to retrain)")
            models = {k: v for k, v in models.items() if k not in already_done}
        if not models:
            logger.info("All requested models already trained. Nothing to do.")
            return {}

    logger.info(f"Models: {list(models.keys())}")

    logger.info("=== Step 6: Running recursive backtest ===")
    ckpt_dir = (
        getattr(args, "checkpoint_dir", None)
        or f"data/cache/{cfg['checkpoint_subdir']}"
    )
    engine = BacktestEngine(
        train_start=cfg["train_start"],
        val_start=cfg["val_start"],
        val_end=cfg["val_end"],
        test_start=cfg["test_start"],
        test_end=cfg["test_end"],
        n_deciles=10,
        weighting="value",
        tc_bps=float(cfg["tc_bps"]),
        tc_model=cfg.get("tc_model", "flat"),
        refit_step_years=int(getattr(args, "refit_step_years", None) or 1),
        checkpoint_dir=ckpt_dir,
    )

    results = engine.run(feature_matrix, models)

    if save_per_model:
        model_dir = Path(cfg["model_dir"])
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
                "variant":           cfg["name"],
            }
            out_path = model_dir / f"{name}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(model_result, f)
            logger.info(f"Saved {name} -> {out_path}")

        variant = cfg["name"]
        logger.info(f"=== Train stage complete (variant={variant!r}). ===")
        existing = [p.stem for p in sorted(model_dir.glob("*.pkl"))]
        logger.info(f"Existing model results in {model_dir}: {existing}")
        return results

    # In-line evaluate (full / cache / test modes)
    logger.info("=== Step 7: Evaluating ===")
    evaluator = ModelEvaluator(
        y_true=results["true_returns"],
        predictions=results["predictions"],
        dates=results["test_dates"],
        portfolio_returns=results["portfolio_returns"],
    )

    r2_table = evaluator.oos_r2_table()
    sr_table = evaluator.sharpe_table()
    dm_stats, dm_pvals = dm_table_full(
        results["true_returns"], results["predictions"], results["test_dates"],
    )

    mkt_proxy = (
        pd.DataFrame({
            "date": results["test_dates"],
            "ret":  results["true_returns"],
        }).groupby("date")["ret"].mean()
    )
    comprehensive = evaluator.comprehensive_table(
        portfolio_returns_gross=results["portfolio_returns_gross"],
        portfolio_turnover=results["portfolio_turnover"],
        market_factor=mkt_proxy,
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    r2_table.to_csv(out_dir / "oos_r2.csv")
    sr_table.to_csv(out_dir / "sharpe_table.csv")
    dm_stats.to_csv(out_dir / "dm_table.csv")
    dm_pvals.to_csv(out_dir / "dm_pvalues.csv")
    comprehensive.to_csv(out_dir / "comprehensive.csv")

    from src.reporting.portfolio_io import save_portfolio_bundle

    metrics_out = dict(results["metrics"])
    metrics_out["_reporting"] = {
        "variant": cfg["name"],
        "tc_bps": cfg["tc_bps"],
        "tc_model": cfg.get("tc_model", "flat"),
        "data_start": cfg["data_start"],
        "data_end": cfg["data_end"],
        "test_start": cfg["test_start"],
        "test_end": cfg["test_end"],
        "use_macro_interactions": cfg["use_macro_interactions"],
        "use_industry_dummies": cfg["use_industry_dummies"],
        "portfolio_pickle_format": "bundle_v1",
        "primary_hl_series": "net_of_engine_transaction_costs",
        "hl_engine_tc_bps_default": cfg["tc_bps"],
        "hl_returns_are_net_of_engine_tc": cfg["tc_bps"] > 0 or cfg.get("tc_model") in ("impact", "stock_level"),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    save_portfolio_bundle(
        out_dir / "portfolio_returns.pkl",
        results["portfolio_returns"],
        results["portfolio_returns_gross"],
        results["portfolio_turnover"],
    )

    logger.info(f"Outputs saved to {out_dir}/")
    results["evaluator"]     = evaluator
    results["r2_table"]      = r2_table
    results["sr_table"]      = sr_table
    results["dm_matrix"]     = dm_stats
    results["dm_pvalues"]    = dm_pvals
    results["comprehensive"] = comprehensive
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Predict mode — reuse existing per-model prediction pickles from a source
#  variant ('paper' / 'improved' / ...) over the target variant's test window.
#  No training is performed.
#
#  Design note: per-model pickles in outputs/<v>/models/<m>.pkl carry the raw
#  prediction *arrays* — not fitted model objects. So "scoring fresh rows from
#  a new feature matrix" is not possible without retraining. What is possible
#  and useful is to slice the source variant's existing predictions to the
#  target variant's [test_start, test_end] window and re-emit them in the
#  same per-model format under outputs/<target>/models/, so that the existing
#  --mode evaluate / regimes / dashboard tooling Just Works.
#
#  The target variant's feature matrix is optional here — if present, it is
#  used only to confirm that the (date, permno) pairs in the sliced
#  predictions still exist in the new universe (a sanity check, not a
#  re-score). This sidesteps the CIZ-vs-legacy feature-column compatibility
#  problem entirely.
# ─────────────────────────────────────────────────────────────────────────────

def _slice_pickle_to_window(
    src_pkl: dict, test_start: pd.Timestamp, test_end: pd.Timestamp,
) -> Optional[dict]:
    """Return a new per-model dict whose arrays are restricted to dates in
    [test_start, test_end]. Returns None if no rows survive the slice."""
    import numpy as np
    dates = pd.DatetimeIndex(src_pkl["test_dates"])
    mask = (dates >= test_start) & (dates <= test_end)
    n_keep = int(mask.sum())
    if n_keep == 0:
        return None
    pred = np.asarray(src_pkl["predictions"])[mask]
    truth = np.asarray(src_pkl["true_returns"])[mask]
    sliced_dates = dates[mask]
    permnos = src_pkl.get("test_permnos")
    if permnos is not None:
        permnos = list(np.asarray(permnos)[mask])

    def _filter_pf(pf):
        if not isinstance(pf, dict):
            return pf
        out = {}
        for k, v in pf.items():
            try:
                if hasattr(v, "loc"):
                    s = v.loc[(v.index >= test_start) & (v.index <= test_end)]
                    out[k] = s
                else:
                    out[k] = v
            except Exception:
                out[k] = v
        return out

    return {
        "predictions":             pred,
        "true_returns":            truth,
        "test_dates":              sliced_dates,
        "test_permnos":            permnos,
        "portfolio_returns":       _filter_pf(src_pkl.get("portfolio_returns", {})),
        "portfolio_returns_gross": _filter_pf(src_pkl.get("portfolio_returns_gross", {})),
        "portfolio_turnover":      _filter_pf(src_pkl.get("portfolio_turnover", {})),
        "metrics":                 dict(src_pkl.get("metrics", {})),
        "variant":                 src_pkl.get("variant", "unknown"),
        "_source_pickle_variant":  src_pkl.get("variant", "unknown"),
        "_sliced_to":              {"test_start": str(test_start.date()),
                                    "test_end":   str(test_end.date())},
    }


def run_predict(args) -> dict:
    """
    Reuse pickles from ``--source-model-variant`` for the target variant's
    test window. Writes outputs/<target>/models/<m>.pkl so that downstream
    `--mode evaluate` can build comprehensive tables without retraining.

    Caveats (logged at runtime):
      * If the source pickles do not span the full [test_start, test_end]
        window, only the overlapping subset is emitted.
      * No new predictions are generated for dates outside the source
        coverage. The user is warned explicitly.
      * Feature-column compatibility is NOT checked because we are slicing
        existing predictions, not re-scoring. This is intentional.
    """
    cfg = _resolve_variant(args)
    src_name = getattr(args, "source_model_variant", None)
    if not src_name:
        raise ValueError(
            "--mode predict requires --source-model-variant "
            "(one of: paper, improved, extended_2024, extended_ciz_2026)."
        )
    if src_name == cfg["name"]:
        raise ValueError(
            f"--source-model-variant cannot equal --variant ({src_name}). "
            "Choose a different source whose pickles already exist."
        )

    src_cfg = get_variant_config(src_name)
    src_model_dir = Path(src_cfg["model_dir"])
    if not src_model_dir.exists():
        raise FileNotFoundError(
            f"Source model dir not found: {src_model_dir}. "
            f"Have the {src_name!r} variant pickles been restored from Drive?"
        )
    src_pkls = sorted(src_model_dir.glob("*.pkl"))
    if not src_pkls:
        raise FileNotFoundError(
            f"No model pickles in {src_model_dir}. Restore them from Drive "
            f"or run `--mode train --variant {src_name}` first."
        )

    requested = set(getattr(args, "models", None) or [])
    if requested:
        src_pkls = [p for p in src_pkls if p.stem in requested]
        if not src_pkls:
            raise ValueError(
                f"None of the requested models {sorted(requested)} were "
                f"found in {src_model_dir}. Available: "
                f"{[p.stem for p in sorted(src_model_dir.glob('*.pkl'))]}"
            )

    test_start = pd.Timestamp(cfg["test_start"])
    test_end   = pd.Timestamp(cfg["test_end"])
    dst_model_dir = Path(cfg["model_dir"])
    dst_model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"=== Predict mode: slicing pickles from variant={src_name!r} "
        f"into variant={cfg['name']!r} test window "
        f"[{test_start.date()} .. {test_end.date()}] ==="
    )
    logger.info(
        "Note: per-model pickles store predictions, not fitted model objects. "
        "This mode reuses already-produced predictions and does NOT re-score "
        "rows from the target feature matrix. To get predictions strictly "
        "after the source variant's test_end you must retrain."
    )

    feature_matrix_path = _feature_matrix_path(cfg)
    fm_keys = None
    if feature_matrix_path.exists():
        try:
            fm = pd.read_parquet(feature_matrix_path, columns=["date", "permno"])
            fm["date"] = pd.to_datetime(fm["date"])
            fm = fm[(fm["date"] >= test_start) & (fm["date"] <= test_end)]
            fm_keys = set(zip(fm["date"].astype("int64"), fm["permno"]))
            logger.info(
                f"Target feature matrix found ({feature_matrix_path}); "
                f"{len(fm_keys):,} (date, permno) pairs in test window will "
                f"be used as a sanity-check filter."
            )
        except Exception as e:
            logger.warning(f"Could not read target feature matrix for "
                           f"coverage check: {e}. Skipping coverage filter.")
            fm_keys = None
    else:
        logger.info(
            f"No target feature matrix at {feature_matrix_path}. "
            "Proceeding without coverage cross-check; you should still run "
            "`--mode data-only --variant post2016_ciz` before backtest/regimes "
            "if you want a populated universe in subsequent stages."
        )

    written = []
    skipped: list[tuple[str, str]] = []
    for p in src_pkls:
        with open(p, "rb") as f:
            src = pickle.load(f)
        sliced = _slice_pickle_to_window(src, test_start, test_end)
        if sliced is None:
            skipped.append((p.stem, "no rows in target window"))
            logger.warning(
                f"[{p.stem}] source pickle has no rows in "
                f"[{test_start.date()}..{test_end.date()}] — skipping."
            )
            continue

        src_dates = pd.DatetimeIndex(src["test_dates"])
        src_max = src_dates.max()
        if src_max < test_end:
            logger.warning(
                f"[{p.stem}] source predictions end at {src_max.date()} but "
                f"target test_end is {test_end.date()} — emitted slice covers "
                f"only [{sliced['test_dates'].min().date()} .. "
                f"{sliced['test_dates'].max().date()}]. The remaining tail "
                "will be missing from --mode evaluate output."
            )

        if fm_keys is not None and len(fm_keys) > 0:
            import numpy as _np
            sd = pd.DatetimeIndex(sliced["test_dates"]).astype("int64")
            sp = _np.asarray(sliced["test_permnos"])
            keep = _np.fromiter(
                (((d, pn) in fm_keys) for d, pn in zip(sd, sp)),
                dtype=bool, count=len(sd),
            )
            n_drop = int((~keep).sum())
            if n_drop > 0:
                logger.warning(
                    f"[{p.stem}] {n_drop:,}/{len(keep):,} sliced rows are not "
                    "in the target feature matrix universe; they will be "
                    "kept in the pickle but flagged in metadata."
                )
                sliced["_coverage_drop_count"] = n_drop
                sliced["_coverage_kept_count"] = int(keep.sum())

        out = dst_model_dir / p.name
        with open(out, "wb") as f:
            pickle.dump(sliced, f)
        written.append(p.stem)
        logger.info(
            f"[{p.stem}] wrote {out} "
            f"(n={len(sliced['predictions']):,}, "
            f"{sliced['test_dates'].min().date()} → "
            f"{sliced['test_dates'].max().date()})"
        )

    logger.info(
        f"=== Predict mode complete. Wrote {len(written)} pickle(s) to "
        f"{dst_model_dir}: {written}. Skipped: {skipped}. ==="
    )
    if written:
        logger.info(
            f"Next: `python main.py --mode evaluate --variant {cfg['name']}` "
            "to build OOS R², Sharpe and comprehensive tables from the "
            "sliced predictions."
        )
    return {"written": written, "skipped": skipped,
            "source_variant": src_name, "target_variant": cfg["name"]}


# ─────────────────────────────────────────────────────────────────────────────
#  Importance mode — fit each model on train+val and compute variable
#  importance on a test slice, save per-variant CSVs.
# ─────────────────────────────────────────────────────────────────────────────

def run_importance(args) -> dict:
    """
    Fit each requested model on the train+val window, compute variable
    importance on a slice of the test data, aggregate the 920 Kronecker
    features back to the 94 base characteristics, and save the results
    to outputs/<variant>/var_importance.csv.

    This is a separate stage from --mode train because it needs a single
    fitted model per name (whereas the backtest engine refits each year
    and discards the model). For computational reasons we fit on the
    initial train+val window only — this matches the variable-importance
    convention used in GKX (2019, Figure 6).
    """
    from src.models.all_models import get_all_models
    from src.backtest.engine import add_forward_return_target, feature_columns_for_training
    from src.evaluation.var_importance import (
        zero_set_importance, permutation_importance,
        aggregate_to_base_chars, fit_for_importance,
    )

    cfg = _resolve_variant(args)
    cache = _feature_matrix_path(cfg)
    if not cache.exists():
        raise FileNotFoundError(
            f"No cached feature matrix at {cache}. "
            f"Run --mode data-only --variant {cfg['name']} first."
        )

    fm = pd.read_parquet(cache)
    fm = add_forward_return_target(fm)

    # Determine fit window
    fit_end = pd.Timestamp(args.importance_fit_end or cfg["val_end"])
    train_start = pd.Timestamp(cfg["train_start"])
    val_start   = pd.Timestamp(cfg["val_start"])
    test_start  = pd.Timestamp(cfg["test_start"])

    feat_cols = feature_columns_for_training(fm, "ret_fwd")
    train_mask = (fm["date"] >= train_start) & (fm["date"] < val_start)
    val_mask   = (fm["date"] >= val_start) & (fm["date"] <= fit_end)
    test_mask  = (fm["date"] >= test_start) & (fm["date"] <= pd.Timestamp(cfg["test_end"]))

    train = fm[train_mask].dropna(subset=["ret_fwd"])
    val   = fm[val_mask].dropna(subset=["ret_fwd"])
    test  = fm[test_mask].dropna(subset=["ret_fwd"])

    if len(train) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Empty train ({len(train)}) or test ({len(test)}) slice. "
            "Check date ranges in the variant config."
        )

    # Subsample test for speed (importance with 100+ models * 920 features
    # otherwise blows up). Stratify by date to keep cross-sectional structure.
    rng = np.random.default_rng(42)
    if len(test) > 200_000:
        sample_dates = test["date"].drop_duplicates().sample(
            n=min(60, test["date"].nunique()), random_state=42
        )
        test = test[test["date"].isin(sample_dates)]
        logger.info(f"Subsampled importance test slice to {len(test):,} rows "
                    f"({len(sample_dates)} dates)")

    X_tr = train[feat_cols].fillna(0)
    y_tr = train["ret_fwd"].values
    X_v  = val[feat_cols].fillna(0) if len(val) > 0 else None
    y_v  = val["ret_fwd"].values if len(val) > 0 else None
    X_te = test[feat_cols].fillna(0)
    y_te = test["ret_fwd"].values

    # Models to evaluate
    models = get_all_models(nn_kwargs={
        "batch_size": 10000, "max_epochs": 50, "patience": 5, "n_ensemble": 3,
    })
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}

    # Trained model objects only exist after we re-fit. To save time and
    # keep this orthogonal to --mode train, we always refit here.
    logger.info(
        f"=== Variable importance: fitting {len(models)} model(s) on "
        f"{train_start.date()} -> {fit_end.date()} "
        f"(train={len(train):,}, val={len(val):,}, test_slice={len(test):,}) ==="
    )

    # Read base characteristics list
    char_cols_path = _char_cols_path(cfg)
    if char_cols_path.exists():
        with open(char_cols_path) as f:
            base_chars = json.load(f)
    else:
        # Synthetic / fallback
        base_chars = [c.replace("_const", "") for c in feat_cols if c.endswith("_const")]
        if not base_chars:
            base_chars = list(feat_cols)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    importance_method = args.importance_method
    importance_fn = (
        zero_set_importance if importance_method == "zero" else permutation_importance
    )

    all_imp_base = {}
    all_imp_full = {}
    for name, model in models.items():
        logger.info(f"[importance] fitting {name}...")
        try:
            fitted = fit_for_importance(model, X_tr, y_tr, X_v, y_v)
            logger.info(f"[importance] computing {importance_method} importance for {name}...")
            imp_full = importance_fn(fitted, X_te.copy(), y_te, feature_names=feat_cols)
            imp_base = aggregate_to_base_chars(imp_full, base_chars)
            all_imp_full[name] = imp_full
            all_imp_base[name] = imp_base
            # Also save per-model on disk for reproducibility
            imp_full.to_csv(out_dir / f"var_importance_full_{name}.csv",
                            header=["importance"])
            imp_base.to_csv(out_dir / f"var_importance_{name}.csv",
                            header=["importance"])
            logger.info(f"[importance] {name}: top-5 = "
                        f"{imp_base.head(5).to_dict()}")
        except Exception as e:
            logger.exception(f"[importance] {name} failed: {e}")
            continue

    # Combined wide table: base chars × models
    if all_imp_base:
        combined = pd.DataFrame(all_imp_base)
        # Normalise per column (so each model sums to 1) for cross-model comparison
        combined_norm = combined.div(combined.abs().sum(axis=0).replace(0, np.nan), axis=1)
        combined.to_csv(out_dir / "var_importance.csv")
        combined_norm.to_csv(out_dir / "var_importance_normalised.csv")
        logger.info(f"Saved combined importance: {out_dir / 'var_importance.csv'}")

    return {"importance_base": all_imp_base, "importance_full": all_imp_full}


# ─────────────────────────────────────────────────────────────────────────────
#  Regimes mode — regime-conditional evaluation (NBER, VIX, decades)
# ─────────────────────────────────────────────────────────────────────────────

def run_regimes(args) -> dict:
    """
    Slice each model's H-L return series by:
      * NBER recession vs expansion
      * VIX terciles (low / mid / high implied vol)
      * Decade (1990s / 2000s / 2010s / 2020s)
    and write a long-format ``regimes.csv`` to outputs/<variant>/.

    Reads per-model pickles from outputs/<variant>/models/, then loads
    the saved portfolio bundle from outputs/<variant>/portfolio_returns.pkl
    if available — that bundle includes any ENS-AVG / ENS-MSE portfolios
    that ``run_evaluate`` constructed earlier.
    """
    from src.evaluation.regimes import evaluate_regimes
    from src.reporting.portfolio_io import unpack_portfolio_bundle

    cfg = _resolve_variant(args)
    out_dir = Path(cfg["output_dir"])
    bundle_path = out_dir / "portfolio_returns.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No portfolio bundle at {bundle_path}. "
            f"Run --mode evaluate --variant {cfg['name']} first."
        )

    with open(bundle_path, "rb") as f:
        raw = pickle.load(f)
    net, gross, _turn, _meta = unpack_portfolio_bundle(raw)
    if not net:
        raise RuntimeError("Portfolio bundle is empty. Re-run --mode evaluate.")

    # Recover the test_dates from any model's H-L series
    sample = next(iter(net.values())).get("H-L")
    if sample is None or len(sample) == 0:
        raise RuntimeError("Could not find H-L series in portfolio bundle.")
    test_dates = pd.DatetimeIndex(sample.dropna().index).sort_values()

    vix_path = getattr(args, "vix_csv", None)
    df = evaluate_regimes(
        portfolio_returns=net,
        portfolio_returns_gross=gross or net,
        test_dates=test_dates,
        vix_path=vix_path,
    )

    out_path = out_dir / "regimes.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"[regimes] wrote {out_path} ({len(df):,} rows, "
                f"{df['model'].nunique()} models, "
                f"{df['regime_kind'].nunique()} regime kinds)")

    # Quick console summary: NBER recession Sharpe vs expansion Sharpe
    rec_view = (
        df[df["regime_kind"] == "nber"]
        .pivot(index="model", columns="regime", values="sharpe_net")
    )
    if not rec_view.empty:
        logger.info("\n=== H-L Sharpe (net) by NBER regime ===")
        logger.info("\n" + rec_view.round(3).to_string())

    return {"regimes": df}


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
        "--variant",
        choices=["paper", "improved", "extended_2024",
                 "extended_ciz_2026", "post2016_ciz",
                 "future2026_base", "future2026_trending",
                 "future2026_mean_reversion", "future2026_rotating_leaders",
                 "future2026_choppy", "future2026_crisis",
                 "future2026_factor_rotation"],
        default="paper",
        help=(
            "Which pipeline to run. "
            "'paper' = strict GKX (2019) reproduction (1957-2016, TC=0). "
            "'improved' = extended sample to 2024 + transaction costs modelled. "
            "'extended_2024' = real-only post-paper extension pinned to legacy "
            "crsp.msf (2024-12-31). "
            "'extended_ciz_2026' = CIZ/v2-aware extension to 2026-03-31 using "
            "the crsp_q_stock.* monthly tables (legacy schema preserved via "
            "column mapping). "
            "'post2016_ciz' = CIZ-aware *scoring* variant (2017-01..2026-03) "
            "intended for --mode predict to reuse pickles from 'paper' or "
            "'improved' over the post-2016 OOS window without retraining. "
            "Macro interactions are on in all variants. "
            "Each variant writes to its own outputs/<variant>/ directory and "
            "uses its own cached feature matrix, so the pipelines do not "
            "overwrite each other."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["full", "test", "cache", "dashboard",
                 "data-only", "train", "evaluate", "importance",
                 "regimes", "predict"],
        default="test",
        help=(
            "'full' = WRDS + train + evaluate in one shot; "
            "'data-only' = build feature matrix then stop; "
            "'train' = load cached data, train --models subset, save per-model; "
            "'evaluate' = merge per-model .pkl into final tables (incl. ENS-AVG, ENS-MSE); "
            "'importance' = compute variable importance for trained models; "
            "'regimes' = regime-conditional evaluation (NBER, VIX, decades); "
            "'predict' = reuse existing per-model prediction pickles from "
            "another variant (--source-model-variant) over the target "
            "variant's test window — no training required; "
            "'test' = synthetic data; 'cache' = reuse feature matrix; "
            "'dashboard' = launch Streamlit"
        ),
    )
    parser.add_argument("--wrds-username", default=os.environ.get("WRDS_USERNAME", ""))

    # Date / TC overrides — default None so the variant config wins unless
    # the user explicitly passes a value on the command line.
    parser.add_argument("--data-start",  default=None)
    parser.add_argument("--data-end",    default=None)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--val-start",   default=None)
    parser.add_argument("--val-end",     default=None)
    parser.add_argument("--test-start",  default=None)
    parser.add_argument("--test-end",    default=None)
    parser.add_argument("--tc-bps",      default=None, type=float,
                        help="Transaction cost in bps (one-way). Overrides variant default.")
    parser.add_argument("--refit-step-years", default=None, type=int,
                        help="Refit each model every N years (default 1 = paper-faithful annual). "
                             "Set to 2 for ~2x NN speedup with minimal R² loss during development. "
                             "Checkpoint filename includes this value, so different cadences don't "
                             "load each other's state.")

    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models to run (e.g. OLS-3 ENet+H RF NN3)")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Retrain models even if their pickles already exist in "
                             "outputs/<variant>/models/. By default, models with "
                             "existing pickles are skipped.")
    parser.add_argument("--data-step",
                        choices=["all", "fetch", "merge", "chars", "features"],
                        default="all",
                        help="Which data sub-step to run in data-only mode: "
                             "'all' = run 1->4; 'fetch' = WRDS download; "
                             "'merge' = CRSP+Compustat merge; "
                             "'chars' = build characteristics; "
                             "'features' = Kronecker feature matrix")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override directory for per-year backtest checkpoints. "
             "Defaults to data/cache/backtest_checkpoint_<variant>/.",
    )
    parser.add_argument("--no-ensembles", action="store_true",
                        help="Skip ENS-AVG and ENS-MSE forecast combinations in --mode evaluate.")
    parser.add_argument("--vix-csv", default=None,
                        help="Optional path to a CSV with monthly VIX (cols: date, vix). "
                             "If absent, the embedded offline VIX series is used.")
    parser.add_argument("--importance-method",
                        choices=["zero", "permutation"],
                        default="zero",
                        help="Variable importance method (only used in --mode importance).")
    parser.add_argument("--importance-fit-end",
                        default=None,
                        help="End date of the train+val window used to fit models for "
                             "variable importance (default: variant's val_end).")
    parser.add_argument(
        "--source-model-variant",
        choices=["paper", "improved", "extended_2024", "extended_ciz_2026"],
        default=None,
        help="Used by --mode predict. Source variant whose per-model pickles "
             "in outputs/<source>/models/ should be sliced into the target "
             "variant's test window. 'improved' is the recommended source for "
             "post2016_ciz since its pickles cover 1987-2024.",
    )
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
    elif args.mode == "importance":
        results = run_importance(args)
    elif args.mode == "regimes":
        results = run_regimes(args)
    elif args.mode == "predict":
        results = run_predict(args)
    elif args.mode == "dashboard":
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "streamlit", "run",
                        "src/dashboard/app.py"])
        return

    logger.info("Pipeline complete.")
    return results


if __name__ == "__main__":
    main()