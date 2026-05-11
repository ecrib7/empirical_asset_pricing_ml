"""
generate_synthetic_results.py
-----------------------------
Produces fully synthetic backtest artifacts for the project without
hitting WRDS and without training any real model. Used to populate the
``outputs/<variant>/`` tree (and feed the Streamlit dashboard) when the
underlying real data is unavailable or when stress-testing the post-WRDS
horizon.

Two families of variants are supported:

  * ``post2016_ciz``         — 2017-01-31 .. 2026-03-31 (CIZ-window
    scoring synthetic; same horizon as the real CIZ extension).
  * ``future2026_*``         — 2026-04-30 .. 2036-03-31 (forward
    post-WRDS scenarios inspired by anticor-trader regimes). For these
    variants the artifacts are derived from a stock-level synthetic
    panel (``src/synthetic/panels.py``) so per-model decile returns
    actually come from sorting 800 synthetic stocks on a synthetic
    prediction signal — not from a decile-only shortcut. The panels
    themselves live at ``data/cache/synthetic_panels/<variant>.parquet``
    (96,000 rows: 120 month-ends × 800 permnos).

Each variant writes the standard artifact set the rest of the project
already consumes:

  outputs/<variant>/
    metrics.json
    comprehensive.csv
    oos_r2.csv
    sharpe_table.csv
    dm_table.csv
    dm_pvalues.csv
    regimes.csv
    var_importance.csv
    portfolio_returns.pkl     # bundle_v1 (see src/reporting/portfolio_io.py)
    models/<MODEL>.pkl        # one per model with predictions + metrics

The scenario generators below are NOT seed permutations — they produce
qualitatively distinct dynamics so the dashboard tells the regimes
apart at a glance (turnover, drawdown, leadership stability,
factor-tilt sign).

IMPORTANT: nothing here is a real WRDS training run. All artifacts are
labelled ``synthetic`` in ``metrics.json`` and the per-model pickles —
they emulate model evaluation outputs for dashboard compatibility but
must never be presented as real model results.

Run:
    # Generate panels + outputs for one variant.
    python generate_synthetic_results.py --variant future2026_base

    # Generate panels + outputs for every future2026 variant.
    python generate_synthetic_results.py --variant future2026_all

    # Only (re)generate parquet panels — skip the per-model artifacts.
    python generate_synthetic_results.py --variant future2026_all --panels-only

    # Build outputs from an already-generated panel without re-generating
    # the parquet.
    python generate_synthetic_results.py --variant future2026_base --from-panel
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Local imports: avoid heavyweight project imports so this script runs
# even when optional deps (torch, wrds) are missing.
try:
    from src.config import (
        FUTURE2026_END,
        FUTURE2026_SCENARIOS,
        FUTURE2026_START,
        VARIANT_DEFAULTS,
        get_variant_config,
    )
    from src.synthetic.panels import (
        REQUIRED_COLUMNS as PANEL_REQUIRED_COLUMNS,
        decile_returns_from_panel,
        generate_panel,
        load_panel,
        panel_path,
        write_panel,
    )
    _PANELS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    # Allow execution from arbitrary CWD; fall back to literals.
    FUTURE2026_START = "2026-04-30"
    FUTURE2026_END = "2036-03-31"
    FUTURE2026_SCENARIOS = (
        "future2026_base",
        "future2026_trending",
        "future2026_mean_reversion",
        "future2026_rotating_leaders",
        "future2026_choppy",
        "future2026_crisis",
        "future2026_factor_rotation",
    )
    VARIANT_DEFAULTS = {}

    def get_variant_config(name):  # type: ignore[no-redef]
        if name not in VARIANT_DEFAULTS:
            raise ValueError(f"Unknown variant {name!r}")
        return dict(VARIANT_DEFAULTS[name])

    _PANELS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODELS: Tuple[str, ...] = (
    "OLS-3", "ENet+H", "PCR", "PLS", "GLM+H", "GBRT+H",
    "NN1", "NN2", "NN3", "NN4", "ENS-AVG", "ENS-MSE",
)

# A small but representative characteristic universe, used for variable-
# importance reporting. Matches the abbreviations in src/config.py.
CHAR_NAMES: Tuple[str, ...] = (
    "mom1m", "mom6m", "mom12m", "mom36m", "chmom", "indmom",
    "mvel1", "dolvol", "turn", "ill", "baspread",
    "beta", "betasq", "idiovol", "retvol",
    "bm", "ep", "sp", "cfp", "dy",
    "agr", "invest", "operprof", "gma", "roeq", "roaq",
    "acc", "lev", "sgr", "lgr",
    "age", "depr", "chinv",
)

DECILES: Tuple[str, ...] = tuple(str(d) for d in range(1, 11))


# Variant -> (data_start, data_end). future2026_* read from config.
POST2016_RANGE = ("2017-01-31", "2026-03-31")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario primitives
# ─────────────────────────────────────────────────────────────────────────────

def _month_index(start: str, end: str) -> pd.DatetimeIndex:
    """Month-end index from ``start`` to ``end`` inclusive."""
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="ME")


def _scenario_params(scenario: str) -> dict:
    """Per-scenario tuning of the dynamics.

    The keys define *qualitatively* different synthetic regimes so each
    variant's dashboard view differs by more than just seed.

    Fields
    ------
    drift           per-month expected H-L spread (decile mean shift)
    vol             monthly stdev of the decile return
    momentum        autoregressive coefficient on leadership; large +ve
                    -> persistent trending, ~0 -> noisy, -ve -> reversion
    leader_period   months between leadership rotations (0 = no rotation)
    crisis_month    if >0, a drawdown event at month index `crisis_month`
                    followed by partial recovery
    char_signal     |signal| baseline for variable-importance (smaller
                    = weaker characteristic effect; larger = stronger)
    factor_rotate   bool — flips the sign of the dominant factor every
                    `factor_period` months (style/value vs growth cycle)
    factor_period   rotation period for factor_rotate; ignored otherwise
    tc_bps          baseline transaction cost (one-way, bps) used to
                    derive net-vs-gross H-L Sharpes
    """
    table = {
        # Calibrated baseline. Roughly mirrors the improved-variant tail
        # stats: ~12-15% gross H-L, ~8-10% net, modest persistence.
        "base": dict(
            drift=0.0085, vol=0.035, momentum=0.20,
            leader_period=0, crisis_month=-1,
            char_signal=0.6, factor_rotate=False, factor_period=0,
            tc_bps=10.0,
        ),
        # Strong persistent leadership: long winners, short losers,
        # both deciles' returns drift in the same sign for years.
        "trending": dict(
            drift=0.012, vol=0.030, momentum=0.55,
            leader_period=0, crisis_month=-1,
            char_signal=0.9, factor_rotate=False, factor_period=0,
            tc_bps=10.0,
        ),
        # Strong mean reversion: high-turnover, contrarian winners.
        # Negative momentum coef -> last month's loser leads next month.
        "mean_reversion": dict(
            drift=0.006, vol=0.034, momentum=-0.40,
            leader_period=0, crisis_month=-1,
            char_signal=0.5, factor_rotate=False, factor_period=0,
            tc_bps=10.0,
        ),
        # Leaders rotate every 12 months: rank ordering of deciles is
        # permuted on a fixed clock.
        "rotating_leaders": dict(
            drift=0.009, vol=0.033, momentum=0.10,
            leader_period=12, crisis_month=-1,
            char_signal=0.7, factor_rotate=False, factor_period=0,
            tc_bps=10.0,
        ),
        # High noise, low signal: vol up, |drift|/vol ratio collapses.
        "choppy": dict(
            drift=0.003, vol=0.055, momentum=0.05,
            leader_period=0, crisis_month=-1,
            char_signal=0.25, factor_rotate=False, factor_period=0,
            tc_bps=10.0,
        ),
        # Correlated drawdown shock around month 30 (~ 2.5y in), then
        # gradual recovery. Cross-sectional dispersion shrinks during
        # the shock (all deciles down together).
        "crisis": dict(
            drift=0.007, vol=0.045, momentum=0.15,
            leader_period=0, crisis_month=30,
            char_signal=0.5, factor_rotate=False, factor_period=0,
            tc_bps=10.0,
        ),
        # Style/factor rotation: dominant char sign flips every 18 mo.
        "factor_rotation": dict(
            drift=0.008, vol=0.034, momentum=0.20,
            leader_period=0, crisis_month=-1,
            char_signal=0.6, factor_rotate=True, factor_period=18,
            tc_bps=10.0,
        ),
        # post2016_ciz uses the calibrated baseline as a stand-in.
        "post2016_ciz": dict(
            drift=0.0090, vol=0.034, momentum=0.22,
            leader_period=0, crisis_month=-1,
            char_signal=0.65, factor_rotate=False, factor_period=0,
            tc_bps=10.0,
        ),
    }
    return table[scenario]


def _decile_returns_for_scenario(
    scenario: str,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> Dict[str, pd.Series]:
    """Generate decile-return Series under a named scenario.

    Returns a dict mapping decile name ("1".."10", "H-L") to a Series
    indexed by `dates`. The H-L Series is the (10) - (1) spread by
    construction so downstream code is internally consistent.
    """
    p = _scenario_params(scenario)
    n = len(dates)

    # Latent leadership process. Each decile has a baseline rank-based
    # tilt; momentum applies to the deviation of the spread around the
    # cross-section mean.
    rank_tilt = np.linspace(-1.0, 1.0, len(DECILES))  # decile 1 -> -1, decile 10 -> +1

    # Optional leader rotation: permute rank_tilt every leader_period.
    rank_path = np.tile(rank_tilt, (n, 1))  # (T, 10)
    if p["leader_period"] > 0:
        for start in range(0, n, p["leader_period"]):
            block_seed = rng.integers(0, 1_000_000)
            block_rng = np.random.default_rng(block_seed)
            perm = block_rng.permutation(len(DECILES))
            rank_path[start:start + p["leader_period"]] = rank_tilt[perm]

    # Optional factor rotation: flip dominant tilt sign every factor_period.
    if p["factor_rotate"] and p["factor_period"] > 0:
        sign = np.ones(n)
        for start in range(0, n, p["factor_period"]):
            if (start // p["factor_period"]) % 2 == 1:
                sign[start:start + p["factor_period"]] = -1.0
        rank_path = rank_path * sign[:, None]

    # Autoregressive shock on the spread direction.
    eps = rng.normal(0.0, 1.0, size=n)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = p["momentum"] * z[t - 1] + np.sqrt(1.0 - p["momentum"] ** 2) * eps[t]

    # Cross-sectional draws per month.
    cs_noise = rng.normal(0.0, p["vol"], size=(n, len(DECILES)))
    common = rng.normal(0.0, p["vol"] * 0.6, size=n)  # market factor

    # Compose decile returns: rank-tilted drift + AR(1) leadership *
    # rank_tilt + market factor + idiosyncratic.
    decile_mat = (
        p["drift"] * rank_path
        + (z[:, None]) * rank_path
        + common[:, None]
        + cs_noise
    )

    # Crisis: large correlated drawdown around `crisis_month`, then
    # 6-month exponential recovery.
    if p["crisis_month"] > 0 and p["crisis_month"] < n:
        cm = p["crisis_month"]
        decile_mat[cm] -= 0.18              # ~18% broad drawdown
        decile_mat[cm + 1] -= 0.04 if cm + 1 < n else 0.0
        for k in range(1, 7):
            if cm + k < n:
                decile_mat[cm + k] += 0.025 * np.exp(-k / 3.0)

    out: Dict[str, pd.Series] = {}
    for j, name in enumerate(DECILES):
        out[name] = pd.Series(decile_mat[:, j], index=dates, name=name)
    out["H-L"] = (out["10"] - out["1"]).rename("H-L")
    return out


def _apply_tc(
    portfolio_gross: Dict[str, pd.Series],
    tc_bps: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Subtract per-month transaction cost from each decile series.

    Returns (net_returns, turnover) — turnover is a synthetic
    proxy in (0.5, 2.0) one-way.
    """
    n = len(next(iter(portfolio_gross.values())))
    turnover_arr = np.clip(rng.normal(1.55, 0.15, size=n), 0.5, 2.5)
    tc = (tc_bps / 1e4) * turnover_arr
    net: Dict[str, pd.Series] = {}
    turnover: Dict[str, pd.Series] = {}
    for k, s in portfolio_gross.items():
        net[k] = (s - tc).rename(k)
        turnover[k] = pd.Series(turnover_arr, index=s.index, name=k)
    return net, turnover


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return float(np.sqrt(12) * s.mean() / s.std())


def _max_drawdown_pct(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) == 0:
        return 0.0
    nav = (1.0 + s).cumprod()
    peak = nav.cummax()
    dd = (nav / peak - 1.0)
    return float(-dd.min() * 100.0)


def _model_metrics(
    model: str,
    portfolio_net: Dict[str, pd.Series],
    portfolio_gross: Dict[str, pd.Series],
    portfolio_turnover: Dict[str, pd.Series],
    oos_r2_pct: float,
    tc_bps: float,
) -> dict:
    hl_net = portfolio_net["H-L"]
    hl_gross = portfolio_gross["H-L"]
    return {
        "oos_r2_pct": round(float(oos_r2_pct), 6),
        "hl_sharpe": round(_sharpe(hl_net), 6),
        "hl_sharpe_gross": round(_sharpe(hl_gross), 6),
        "hl_mean_turnover_one_way": round(float(portfolio_turnover["H-L"].mean()), 6),
        "hl_engine_tc_bps": float(tc_bps),
        "hl_returns_are_net_of_tc": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-model synthetic prediction realisation
# ─────────────────────────────────────────────────────────────────────────────

def _model_oos_r2_pct(scenario: str, model: str, rng: np.random.Generator) -> float:
    """Synthetic OOS R² (%) per model.

    Scaled so ensembles and NNs do best in regimes with structure,
    weakest in choppy/crisis. Sign matches the empirical pattern from
    the project's improved-variant metrics. Values reflect a plausible
    monthly cross-sectional OOS R² (≈ 0.2-0.6%) — the bulk of return
    variance is idio noise that no realistic model captures.
    """
    base = {
        "OLS-3": 0.10, "ENet+H": 0.08, "GLM+H": 0.09, "PCR": 0.18,
        "PLS": 0.16, "GBRT+H": 0.35, "NN1": 0.25, "NN2": 0.28,
        "NN3": 0.40, "NN4": 0.38, "ENS-AVG": 0.45, "ENS-MSE": 0.50,
    }.get(model, 0.10)
    regime_mult = {
        "base": 1.00,
        "trending": 1.30,
        "mean_reversion": 0.65,
        "rotating_leaders": 0.55,
        "choppy": 0.15,
        "crisis": -0.20,
        "factor_rotation": 0.65,
        "post2016_ciz": 1.10,
    }.get(scenario, 1.0)
    noise = rng.normal(0.0, 0.03)
    return float(base * regime_mult + noise)


def _build_model_pickle(
    variant: str,
    scenario: str,
    model: str,
    dates: pd.DatetimeIndex,
    n_per_month: int,
    rng: np.random.Generator,
    portfolio_returns: Dict[str, pd.Series],
    portfolio_returns_gross: Dict[str, pd.Series],
    portfolio_turnover: Dict[str, pd.Series],
    tc_bps: float,
) -> dict:
    """Build the per-model dict consumed by src/reporting and dashboard."""
    n_total = len(dates) * n_per_month
    true_returns = rng.normal(0.008, 0.10, size=n_total).astype(np.float32)
    pred_sigma = 0.05
    # Models with higher OOS R² explain a larger share of true returns.
    r2_decimal = max(0.0, _model_oos_r2_pct(scenario, model, rng) / 100.0)
    rho = np.clip(np.sqrt(r2_decimal * 25.0), 0.0, 0.6)  # implied corr
    noise = rng.normal(0.0, 1.0, size=n_total).astype(np.float32)
    predictions = (rho * (true_returns / true_returns.std()) +
                   np.sqrt(1.0 - rho ** 2) * noise) * pred_sigma
    test_dates = np.repeat(dates.values, n_per_month)
    permnos = list(range(10001, 10001 + n_per_month))
    return {
        "predictions": predictions,
        "true_returns": true_returns,
        "test_dates": pd.DatetimeIndex(test_dates),
        "test_permnos": permnos * len(dates),
        "portfolio_returns": portfolio_returns,
        "portfolio_returns_gross": portfolio_returns_gross,
        "portfolio_turnover": portfolio_turnover,
        "metrics": _model_metrics(
            model, portfolio_returns, portfolio_returns_gross,
            portfolio_turnover, _model_oos_r2_pct(scenario, model, rng),
            tc_bps,
        ),
        "variant": variant,
    }


def _build_model_pickle_from_panel(
    variant: str,
    scenario: str,
    model: str,
    panel: pd.DataFrame,
    rng: np.random.Generator,
    portfolio_returns: Dict[str, pd.Series],
    portfolio_returns_gross: Dict[str, pd.Series],
    portfolio_turnover: Dict[str, pd.Series],
    tc_bps: float,
) -> dict:
    """Build the per-model dict from a stock-level panel.

    Predictions, true returns, dates and permnos are sliced from the
    panel rather than randomly drawn — this is the path that satisfies
    requirement #4 (model artefacts emulate model evaluation from the
    underlying synthetic stock-level series).
    """
    predictions, true_returns, test_dates, test_permnos = _panel_predictions_for_model(
        panel, model, rng
    )
    oos_r2 = _model_oos_r2_pct(scenario, model, rng)
    return {
        "predictions": predictions,
        "true_returns": true_returns,
        "test_dates": test_dates,
        "test_permnos": list(test_permnos),
        "portfolio_returns": portfolio_returns,
        "portfolio_returns_gross": portfolio_returns_gross,
        "portfolio_turnover": portfolio_turnover,
        "metrics": _model_metrics(
            model, portfolio_returns, portfolio_returns_gross,
            portfolio_turnover, oos_r2, tc_bps,
        ),
        "variant": variant,
        "synthetic": True,
        "source": "synthetic_panel",
        "panel_path": str(panel_path(variant)) if _PANELS_AVAILABLE else None,
        "n_stocks_per_month": int(panel["permno"].nunique()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate artifacts
# ─────────────────────────────────────────────────────────────────────────────

def _comprehensive_row(model: str, hl_net: pd.Series, hl_gross: pd.Series,
                       oos_r2_pct: float, mean_to: float) -> dict:
    return {
        "Model": model,
        "Sharpe (net)": round(_sharpe(hl_net), 6),
        "Sharpe (gross)": round(_sharpe(hl_gross), 6),
        "SR*": None,
        "Max DD (%)": round(_max_drawdown_pct(hl_net), 6),
        "Skew": round(float(hl_net.skew()) if len(hl_net.dropna()) > 2 else 0.0, 6),
        "Kurt": round(float(hl_net.kurt()) if len(hl_net.dropna()) > 3 else 0.0, 6),
        "OOS R² (%)": round(float(oos_r2_pct), 6),
        "Mean TO (1-way)": round(float(mean_to), 6),
        "Alpha (% / yr)": round(float(hl_net.mean() * 12 * 100), 6),
        "t(alpha)": round(float(np.sqrt(len(hl_net.dropna())) *
                                hl_net.mean() / max(hl_net.std(), 1e-9)), 6),
    }


def _dm_tables(models: Sequence[str], rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Symmetric-by-construction synthetic DM stat + p-value tables."""
    n = len(models)
    raw = rng.normal(0.0, 0.5, size=(n, n))
    raw = (raw - raw.T) / 2.0
    np.fill_diagonal(raw, np.nan)
    p_raw = rng.uniform(0.3, 0.95, size=(n, n)).astype(np.float64, copy=True)
    np.fill_diagonal(p_raw, np.nan)
    stat = pd.DataFrame(raw, index=list(models), columns=list(models))
    pvals = pd.DataFrame(p_raw, index=list(models), columns=list(models))
    return stat, pvals


def _var_importance(rng: np.random.Generator) -> pd.DataFrame:
    """Per-model variable importance: rows = characteristics, cols = models."""
    rows = []
    for char in CHAR_NAMES:
        rows.append(rng.dirichlet(np.ones(len(MODELS)) * 2.0) * 100.0)
    df = pd.DataFrame(rows, index=list(CHAR_NAMES), columns=list(MODELS))
    return df.round(4)


def _regime_rows(hl_by_model: Dict[str, Tuple[pd.Series, pd.Series]],
                 dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Long-format regime evaluation analogous to src/evaluation/regimes.py."""
    rows = []
    # Synthetic regime labels per month.
    rng = np.random.default_rng(seed=0)
    nber = rng.uniform(0, 1, len(dates)) < 0.15  # ~15% "recession" months
    vix_terc = pd.qcut(rng.uniform(0, 1, len(dates)), 3,
                       labels=["low_vix", "mid_vix", "high_vix"])
    decade = pd.Series(dates.year // 10 * 10, index=dates).astype(str) + "s"

    for model, (hl_net, hl_gross) in hl_by_model.items():
        def add(kind: str, label: str, mask) -> None:
            sub = hl_net.loc[mask]
            sub_g = hl_gross.loc[mask]
            rows.append({
                "regime_kind": kind,
                "regime": label,
                "model": model,
                "sharpe_net": round(_sharpe(sub), 6),
                "sharpe_gross": round(_sharpe(sub_g), 6),
                "mean_return": round(float(sub.mean()) if len(sub) else 0.0, 6),
                "n_months": int(len(sub)),
            })

        add("full", "all", pd.Series(True, index=hl_net.index))
        add("nber", "recession", pd.Series(nber, index=hl_net.index))
        add("nber", "expansion", pd.Series(~nber, index=hl_net.index))
        for lbl in ("low_vix", "mid_vix", "high_vix"):
            add("vix", lbl, pd.Series(vix_terc == lbl, index=hl_net.index))
        for lbl in decade.unique():
            add("decade", lbl, pd.Series(decade == lbl, index=hl_net.index).values)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_dates(variant: str) -> Tuple[pd.DatetimeIndex, str]:
    """Return (month-end index, scenario_name) for a variant.

    The scenario_name drives ``_scenario_params`` and is *not* the same
    as the variant — e.g. ``future2026_trending`` -> ``trending``.
    """
    if variant == "post2016_ciz":
        return _month_index(*POST2016_RANGE), "post2016_ciz"
    if variant in FUTURE2026_SCENARIOS:
        scn = variant.replace("future2026_", "")
        return _month_index(FUTURE2026_START, FUTURE2026_END), scn
    raise ValueError(f"Unsupported variant for synthetic generation: {variant!r}")


def _stable_seed(*parts: object) -> int:
    """Deterministic, process-independent seed derived from ``parts``.

    ``hash()`` is randomized per Python process when PYTHONHASHSEED is
    not set, which makes re-runs non-reproducible — so we hash a UTF-8
    string with ``zlib.crc32`` instead.
    """
    import zlib
    blob = "::".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.crc32(blob)) & 0x7FFFFFFF


def _variant_seed(variant: str) -> int:
    """Deterministic seed per variant so re-runs are reproducible."""
    return _stable_seed("variant", variant)


def _load_or_create_panel(
    variant: str,
    *,
    panel: pd.DataFrame | None,
    panel_root: Path | None,
    generate_if_missing: bool,
) -> pd.DataFrame:
    """Resolve the stock-level synthetic panel for a future2026 variant."""
    if panel is not None:
        return panel
    parquet = (
        (Path(panel_root) / f"{variant}.parquet")
        if panel_root is not None
        else panel_path(variant)
    )
    if parquet.exists():
        return load_panel(variant, in_path=parquet)
    if not generate_if_missing:
        raise FileNotFoundError(
            f"Panel parquet not found at {parquet}; "
            "pass --panels-only first or generate it in-memory."
        )
    df = generate_panel(variant)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    write_panel(variant, df, out_path=parquet)
    return df


def resolve_panel_path(variant: str) -> Path:
    """Repo-relative parquet path for a future2026 variant's panel."""
    if not _PANELS_AVAILABLE:
        raise RuntimeError(
            "src.synthetic.panels could not be imported; cannot resolve panel path."
        )
    return panel_path(variant)


def generate_panel_for_variant(
    variant: str,
    panel_root: Path | None = None,
    *,
    overwrite: bool = True,
) -> Path:
    """Generate the stock-level parquet panel for one future2026 variant.

    Writes ``<panel_root>/<variant>.parquet``. Skips regeneration if
    ``overwrite=False`` and the file already exists.

    Raises if called for a non-future2026 variant.
    """
    if not _PANELS_AVAILABLE:
        raise RuntimeError(
            "src.synthetic.panels could not be imported; cannot generate panels."
        )
    if variant not in FUTURE2026_SCENARIOS:
        raise ValueError(
            f"generate_panel_for_variant only supports future2026_* variants; got {variant!r}"
        )
    out_path = (
        (Path(panel_root) / f"{variant}.parquet")
        if panel_root is not None
        else panel_path(variant)
    )
    if out_path.exists() and not overwrite:
        return out_path
    df = generate_panel(variant)
    return write_panel(variant, df, out_path=out_path)


# Per-model BASE skill (correlation with the latent expected return).
# These are deliberately low: averaging 80 stocks per decile collapses
# idio noise, so a per-stock signal correlation of even 0.10 produces an
# H-L Sharpe > 2 after monthly aggregation. The calibration target is
# plausible model performance (per-scenario avg Sharpe ~0.7-1.5, best
# ~1.5-2.5), not a finance textbook factor model.
_MODEL_SKILL: Dict[str, float] = {
    "OLS-3":   0.040,
    "ENet+H":  0.035,
    "GLM+H":   0.038,
    "PCR":     0.050,
    "PLS":     0.048,
    "GBRT+H":  0.075,
    "NN1":     0.055,
    "NN2":     0.062,
    "NN3":     0.080,
    "NN4":     0.078,
    "ENS-AVG": 0.090,
    "ENS-MSE": 0.095,
}

# Per-scenario multiplier on the per-model skill. Trending regimes
# reward the latent signal more; choppy/crisis regimes degrade it.
_SCENARIO_SKILL_MULT: Dict[str, float] = {
    "base":             1.00,
    "trending":         1.35,
    "mean_reversion":   0.85,
    "rotating_leaders": 0.85,
    "choppy":           0.45,
    "crisis":           0.50,
    "factor_rotation":  0.90,
    "post2016_ciz":     1.05,
}

# Per-scenario stdev of an additive "implementation noise" overlay applied
# to each model's H-L series (monthly, decile-mean-return units). This
# represents model-variance / execution slippage that decile averaging
# does NOT erase. Higher in choppy/crisis to widen drawdowns and keep
# Sharpe modest even for the best ensembles.
_SCENARIO_HL_NOISE: Dict[str, float] = {
    "base":             0.012,
    "trending":         0.010,
    "mean_reversion":   0.013,
    "rotating_leaders": 0.013,
    "choppy":           0.022,
    "crisis":           0.018,
    "factor_rotation":  0.012,
    "post2016_ciz":     0.012,
}

# Per-scenario H-L drawdown injection: (month_index, magnitude) pairs
# applied to the long-short series after construction. Lets us mark a
# crisis regime as one where the model also loses money (not just the
# market). Empty list = no injection.
_SCENARIO_HL_SHOCKS: Dict[str, Tuple[Tuple[int, float], ...]] = {
    "base":             (),
    "trending":         (),
    "mean_reversion":   (),
    "rotating_leaders": (),
    "choppy":           (),
    "crisis":           ((30, -0.055), (31, -0.030), (32, -0.015)),
    "factor_rotation":  (),
    "post2016_ciz":     (),
}


def _model_skill(model: str, scenario: str | None = None) -> float:
    base = float(_MODEL_SKILL.get(model, 0.040))
    if scenario is None:
        return base
    mult = float(_SCENARIO_SKILL_MULT.get(scenario, 1.0))
    return float(np.clip(base * mult, 0.0, 0.30))


def _model_signal_array(
    panel: pd.DataFrame, model: str, rng: np.random.Generator,
) -> np.ndarray:
    """Per-model synthetic signal = correlated mix of latent + noise.

    The mix weight is the model's per-scenario "skill" — small numbers
    (≤0.10) because 80-stock decile averaging makes even tiny per-stock
    correlations translate into substantial H-L Sharpes.
    """
    latent = panel["latent_expected_ret"].to_numpy(dtype=np.float64)
    z = (latent - latent.mean()) / (latent.std() + 1e-9)
    scenario = (
        str(panel["scenario"].iloc[0]).replace("future2026_", "")
        if "scenario" in panel.columns and len(panel)
        else None
    )
    rho = _model_skill(model, scenario)
    noise = rng.standard_normal(size=z.shape)
    return rho * z + float(np.sqrt(max(1.0 - rho ** 2, 0.0))) * noise


def _panel_predictions_for_model(
    panel: pd.DataFrame, model: str, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-model synthetic predictions + true returns sliced from the panel.

    Returns (predictions, true_returns, test_dates, test_permnos) flattened
    in panel row order. Uses a deterministic per-(scenario, model) seed
    for the signal generation so the predictions align with the deciles.
    """
    scenario = (
        str(panel["scenario"].iloc[0])
        if "scenario" in panel.columns and len(panel)
        else "unknown"
    )
    seed = _stable_seed("model_signal", scenario, model)
    local_rng = np.random.default_rng(seed)
    sig = _model_signal_array(panel, model, local_rng)
    # Bring back to roughly return scale.
    predictions = (sig * 0.01).astype(np.float32)
    true_returns = panel["ret"].to_numpy(dtype=np.float32)
    test_dates = pd.DatetimeIndex(panel["date"].values)
    test_permnos = panel["permno"].to_numpy()
    return predictions, true_returns, test_dates, test_permnos


def _decile_returns_from_panel_for_model(
    panel: pd.DataFrame, model: str, rng: np.random.Generator | None = None,
) -> Dict[str, pd.Series]:
    """Per-model decile sort. Stronger-skill models earn wider H-L spreads.

    Uses a deterministic per-(scenario, model) seed for the signal so
    decile outputs are reproducible regardless of caller-side RNG state.

    Adds two calibration overlays so panel-derived Sharpes are plausible:
      * an additive monthly H-L noise overlay (model variance / slippage
        that survives decile averaging), spread evenly across decile 10
        (+) and decile 1 (-) so the H-L identity 10 - 1 still holds;
      * scenario-specific H-L drawdown shocks (e.g. the crisis regime
        also costs the model money, not just the market).
    """
    scenario_full = (
        str(panel["scenario"].iloc[0])
        if "scenario" in panel.columns and len(panel)
        else "unknown"
    )
    scenario = scenario_full.replace("future2026_", "")
    seed = _stable_seed("model_signal", scenario_full, model)
    local_rng = np.random.default_rng(seed)
    sig = _model_signal_array(panel, model, local_rng)
    work = panel[["date", "permno", "ret"]].copy()
    work["__signal__"] = sig
    decile_map = decile_returns_from_panel(work, signal_col="__signal__")
    out = {k: v.copy() for k, v in decile_map.items()}

    # H-L noise overlay — deterministic per (scenario, model).
    overlay_sigma = float(_SCENARIO_HL_NOISE.get(scenario, 0.012))
    overlay_rng = np.random.default_rng(_stable_seed("hl_overlay", scenario_full, model))
    n = len(out["H-L"])
    overlay = overlay_rng.normal(0.0, overlay_sigma, size=n)

    # Apply overlay symmetrically: top decile +overlay/2, bottom -overlay/2.
    # This preserves the H-L = 10 - 1 identity in the resulting series.
    if "10" in out:
        out["10"] = (out["10"] + overlay / 2.0).rename("10")
    if "1" in out:
        out["1"] = (out["1"] - overlay / 2.0).rename("1")

    # Scenario-specific H-L drawdown shocks — applied to top/bottom
    # deciles symmetrically as well, so the H-L identity stays consistent.
    for (t_idx, mag) in _SCENARIO_HL_SHOCKS.get(scenario, ()):
        if 0 <= t_idx < n:
            if "10" in out:
                out["10"].iloc[t_idx] = float(out["10"].iloc[t_idx] + mag / 2.0)
            if "1" in out:
                out["1"].iloc[t_idx] = float(out["1"].iloc[t_idx] - mag / 2.0)

    out["H-L"] = (out["10"] - out["1"]).rename("H-L")
    return out


def generate_variant(
    variant: str,
    out_root: Path | None = None,
    *,
    panel_root: Path | None = None,
    panel: pd.DataFrame | None = None,
    use_panel: bool | None = None,
    generate_panel_if_missing: bool = True,
) -> Path:
    """Write the full artifact set for ``variant`` and return its dir.

    For ``future2026_*`` variants, decile/H-L returns are derived from a
    stock-level synthetic panel:

      * If ``panel`` is provided, it's used directly.
      * Else if a parquet exists at ``panel_root/<variant>.parquet``
        (default ``data/cache/synthetic_panels/<variant>.parquet``),
        the panel is loaded from disk.
      * Else, when ``generate_panel_if_missing`` is true, the panel is
        generated in-memory and persisted to that path before deriving
        outputs from it.

    For ``post2016_ciz``, the legacy decile-only path is used (no panel).
    """
    out_root = out_root or Path("outputs")
    dates, scenario = _resolve_dates(variant)
    if variant in FUTURE2026_SCENARIOS:
        assert len(dates) == 120, (
            f"future2026 expects 120 month-ends (got {len(dates)}); "
            f"first={dates[0]}, last={dates[-1]}"
        )
        assert str(dates[0].date()) == FUTURE2026_START
        assert str(dates[-1].date()) == FUTURE2026_END
    rng = np.random.default_rng(_variant_seed(variant))
    p = _scenario_params(scenario)
    tc_bps = float(p["tc_bps"])

    if use_panel is None:
        use_panel = variant in FUTURE2026_SCENARIOS and _PANELS_AVAILABLE

    # 1) Per-model decile/H-L returns.
    #    future2026_* variants derive these from a stock-level panel (so
    #    decile portfolios actually trade on the underlying synthetic
    #    universe). post2016_ciz keeps the legacy decile-only path.
    panel_used: pd.DataFrame | None = None
    if use_panel and variant in FUTURE2026_SCENARIOS:
        panel_used = _load_or_create_panel(
            variant, panel=panel, panel_root=panel_root,
            generate_if_missing=generate_panel_if_missing,
        )
        gross_decile = _decile_returns_from_panel_for_model(panel_used, model="ENS-AVG", rng=rng)
    else:
        gross_decile = _decile_returns_for_scenario(scenario, dates, rng)

    # Per-model variants: each model uses its own signal column for the
    # cross-sectional sort (so different model "skill" levels translate
    # into different decile spreads), plus a small per-model rescale so
    # leaderboards still differ row-to-row.
    portfolio_returns_gross: Dict[str, Dict[str, pd.Series]] = {}
    portfolio_returns_net: Dict[str, Dict[str, pd.Series]] = {}
    portfolio_turnover: Dict[str, Dict[str, pd.Series]] = {}
    for m in MODELS:
        scale = float(np.clip(rng.normal(1.0, 0.10), 0.7, 1.4))
        bias = float(np.clip(rng.normal(0.0, 0.0015), -0.004, 0.004))
        if panel_used is not None:
            decile_base = _decile_returns_from_panel_for_model(panel_used, model=m, rng=rng)
        else:
            decile_base = gross_decile
        per_model_gross: Dict[str, pd.Series] = {}
        for k, s in decile_base.items():
            if k == "H-L":
                continue
            per_model_gross[k] = (s * scale + bias).rename(k)
        per_model_gross["H-L"] = (per_model_gross["10"] - per_model_gross["1"]).rename("H-L")
        net, to = _apply_tc(per_model_gross, tc_bps, rng)
        portfolio_returns_gross[m] = per_model_gross
        portfolio_returns_net[m] = net
        portfolio_turnover[m] = to

    # 2) Per-model pickles.
    variant_dir = out_root / variant
    models_dir = variant_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload: Dict[str, dict] = {}
    comprehensive_rows: List[dict] = []
    oos_r2_rows: Dict[str, float] = {}
    sharpe_rows: Dict[str, float] = {}
    hl_for_regimes: Dict[str, Tuple[pd.Series, pd.Series]] = {}

    for m in MODELS:
        oos_r2 = _model_oos_r2_pct(scenario, m, rng)
        if panel_used is not None:
            pkl = _build_model_pickle_from_panel(
                variant, scenario, m, panel_used, rng=rng,
                portfolio_returns=portfolio_returns_net[m],
                portfolio_returns_gross=portfolio_returns_gross[m],
                portfolio_turnover=portfolio_turnover[m],
                tc_bps=tc_bps,
            )
        else:
            pkl = _build_model_pickle(
                variant, scenario, m, dates,
                n_per_month=200,  # 200 synthetic permnos / month -> 24k rows
                rng=rng,
                portfolio_returns=portfolio_returns_net[m],
                portfolio_returns_gross=portfolio_returns_gross[m],
                portfolio_turnover=portfolio_turnover[m],
                tc_bps=tc_bps,
            )
        # Pin oos_r2 (the same value flows into metrics + comprehensive).
        pkl["metrics"]["oos_r2_pct"] = round(float(oos_r2), 6)
        with open(models_dir / f"{m}.pkl", "wb") as f:
            pickle.dump(pkl, f)

        metrics_payload[m] = pkl["metrics"]
        hl_net = portfolio_returns_net[m]["H-L"]
        hl_gross = portfolio_returns_gross[m]["H-L"]
        hl_for_regimes[m] = (hl_net, hl_gross)
        comprehensive_rows.append(
            _comprehensive_row(m, hl_net, hl_gross, oos_r2,
                               portfolio_turnover[m]["H-L"].mean())
        )
        oos_r2_rows[m] = round(float(oos_r2), 6)
        sharpe_rows[m] = round(_sharpe(hl_net), 6)

    # 3) Reporting metadata so the dashboard banner shows correct labels.
    cfg = {}
    try:
        cfg = get_variant_config(variant)
    except Exception:
        cfg = {}
    panel_meta: Dict[str, object] = {}
    if panel_used is not None:
        panel_meta = {
            "panel_source": "synthetic_stock_level",
            "panel_path": str(panel_path(variant)) if _PANELS_AVAILABLE else None,
            "n_stocks": int(panel_used["permno"].nunique()),
            "n_rows": int(len(panel_used)),
            "panel_start": str(pd.Timestamp(panel_used["date"].min()).date()),
            "panel_end": str(pd.Timestamp(panel_used["date"].max()).date()),
            "training_kind": "synthetic_training",
            "evaluation_kind": "synthetic_evaluation",
        }
    metrics_payload["_reporting"] = {
        "variant": variant,
        "scenario": scenario,
        "data_start": cfg.get("data_start", str(dates[0].date())),
        "data_end": cfg.get("data_end", str(dates[-1].date())),
        "test_start": cfg.get("test_start", str(dates[0].date())),
        "test_end": cfg.get("test_end", str(dates[-1].date())),
        "synthetic": True,
        "n_months": int(len(dates)),
        "engine_tc_bps": float(tc_bps),
        "source": "generate_synthetic_results.py",
        **panel_meta,
    }

    # 4) Flat artifacts.
    with open(variant_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    pd.DataFrame(comprehensive_rows).to_csv(variant_dir / "comprehensive.csv", index=False)

    pd.Series(oos_r2_rows, name="OOS R² (%)").to_csv(variant_dir / "oos_r2.csv")
    pd.Series(sharpe_rows, name="H-L Sharpe").to_csv(variant_dir / "sharpe_table.csv")

    dm_stat, dm_p = _dm_tables(MODELS, rng)
    dm_stat.to_csv(variant_dir / "dm_table.csv")
    dm_p.to_csv(variant_dir / "dm_pvalues.csv")

    _regime_rows(hl_for_regimes, dates).to_csv(variant_dir / "regimes.csv", index=False)
    _var_importance(rng).to_csv(variant_dir / "var_importance.csv")

    # 5) Portfolio bundle expected by src/reporting/portfolio_io.py.
    bundle = {
        "_format": "bundle_v1",
        "_version": 1,
        "net": portfolio_returns_net,
        "gross": portfolio_returns_gross,
        "turnover": portfolio_turnover,
    }
    with open(variant_dir / "portfolio_returns.pkl", "wb") as f:
        pickle.dump(bundle, f)

    return variant_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

ALL_GENERATABLE: Tuple[str, ...] = ("post2016_ciz",) + FUTURE2026_SCENARIOS
ARG_CHOICES: Tuple[str, ...] = ALL_GENERATABLE + ("future2026_all",)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=list(ARG_CHOICES),
        default="future2026_base",
        help=(
            "Which synthetic variant to generate. "
            "'future2026_all' expands to every future2026_* variant."
        ),
    )
    parser.add_argument(
        "--out-root",
        default="outputs",
        help="Root directory where outputs/<variant>/ will be written.",
    )
    parser.add_argument(
        "--panel-root",
        default=None,
        help=(
            "Directory for stock-level synthetic panel parquets. "
            "Defaults to 'data/cache/synthetic_panels/'."
        ),
    )
    parser.add_argument(
        "--panels-only",
        action="store_true",
        help=(
            "Generate ONLY the stock-level parquet panels for the "
            "selected future2026 variant(s); skip artifact generation."
        ),
    )
    parser.add_argument(
        "--from-panel",
        action="store_true",
        help=(
            "Skip panel regeneration; require an existing parquet at "
            "--panel-root/<variant>.parquet and derive outputs from it."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out_root)
    panel_root = Path(args.panel_root) if args.panel_root else None

    variants: List[str]
    if args.variant == "future2026_all":
        variants = list(FUTURE2026_SCENARIOS)
    else:
        variants = [args.variant]

    if args.panels_only:
        if not _PANELS_AVAILABLE:
            print("[generate_synthetic_results] panels module unavailable", flush=True)
            return 1
        for v in variants:
            if v not in FUTURE2026_SCENARIOS:
                print(
                    f"[generate_synthetic_results] --panels-only is "
                    f"future2026_* only; skipping {v}",
                    flush=True,
                )
                continue
            path = generate_panel_for_variant(v, panel_root=panel_root)
            print(f"[generate_synthetic_results] wrote panel {path}")
        return 0

    for v in variants:
        path = generate_variant(
            v,
            out_root=out_root,
            panel_root=panel_root,
            generate_panel_if_missing=not args.from_panel,
        )
        print(f"[generate_synthetic_results] wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
