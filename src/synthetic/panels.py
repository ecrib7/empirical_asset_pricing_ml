"""
src/synthetic/panels.py
-----------------------
Stock-level synthetic monthly panels for the ``future2026_*`` scenarios.

These panels are the *source of truth* for the future2026 synthetic
artifacts: one parquet per scenario at
``data/cache/synthetic_panels/<variant>.parquet`` (configurable),
covering exactly 120 month-ends from ``2026-04-30`` through
``2036-03-31`` and 800 synthetic permnos (96,000 rows).

The dynamics are inspired by — but not copied from —
`anticor-trader <https://github.com/cvxgrp/anticor-trader>`_: each
scenario draws stock-level returns from a common-factor + style-factor
+ idiosyncratic model with regime-specific parameters, so trending,
mean-reversion, leadership rotation, choppiness, correlated crisis, and
factor rotation are all visibly distinct in the resulting panel.

IMPORTANT: nothing here is a real WRDS draw or a real forecast. The
panels are deterministic synthetic stress fixtures. Downstream code
that consumes them must label artifacts as ``synthetic_training`` /
``synthetic_evaluation`` and never claim real model results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

PANEL_START = "2026-04-30"
PANEL_END = "2036-03-31"
N_PERMNOS = 800
N_MONTHS = 120  # 2026-04-30 .. 2036-03-31 inclusive
PERMNO_BASE = 900000  # synthetic permnos: 900000..900799 (no collision w/ real CRSP)

# Style factors used in the cross-sectional return model.
STYLE_FACTORS: Tuple[str, ...] = (
    "size", "value", "momentum", "quality", "volatility", "liquidity",
)

# Required columns in every panel parquet.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "date", "permno", "scenario",
    "ret", "mkt_ret", "common_factor",
    "latent_expected_ret",
    "market_beta",
    "size", "value", "momentum", "quality", "volatility", "liquidity",
    "model_signal_strong", "model_signal_medium", "model_signal_weak",
)

# Scenario names (without the ``future2026_`` prefix).
SCENARIOS: Tuple[str, ...] = (
    "base",
    "trending",
    "mean_reversion",
    "rotating_leaders",
    "choppy",
    "crisis",
    "factor_rotation",
)


# ─────────────────────────────────────────────────────────────────────
# Scenario parameters
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PanelParams:
    """Per-scenario knobs for the stock-level return model.

    Fields
    ------
    market_vol         monthly stdev of the common (market) factor
    idio_vol           monthly stdev of per-stock idiosyncratic noise
    style_premia       baseline expected premium per style factor (monthly)
    style_persistence  AR(1) coefficient on the style premia path
    char_persistence   AR(1) coefficient on each stock's characteristics
    leader_period      months between leader-group rotations (0 = none)
    factor_rotate      whether to flip the dominant style sign periodically
    factor_period      rotation period for ``factor_rotate``
    crisis_month       index of the crisis shock (-1 = no crisis)
    """

    market_vol: float
    idio_vol: float
    style_premia: Tuple[float, ...]            # one per STYLE_FACTORS
    style_persistence: float
    char_persistence: float
    leader_period: int
    factor_rotate: bool
    factor_period: int
    crisis_month: int


_PARAMS: Dict[str, PanelParams] = {
    # Calibrated baseline: modest market vol, modest style premia, mild
    # persistence in both characteristics and style returns.
    "base": PanelParams(
        market_vol=0.04, idio_vol=0.07,
        style_premia=(0.0010, 0.0015, 0.0030, 0.0015, -0.0020, 0.0010),
        style_persistence=0.30, char_persistence=0.90,
        leader_period=0, factor_rotate=False, factor_period=0,
        crisis_month=-1,
    ),
    # Strong persistent leadership: high AR(1) on style returns + char.
    "trending": PanelParams(
        market_vol=0.035, idio_vol=0.06,
        style_premia=(0.0008, 0.0010, 0.0055, 0.0020, -0.0010, 0.0008),
        style_persistence=0.75, char_persistence=0.97,
        leader_period=0, factor_rotate=False, factor_period=0,
        crisis_month=-1,
    ),
    # Mean reversion: negative AR(1) on style returns; characteristics
    # also revert (lower char_persistence + sign flip on the style path).
    "mean_reversion": PanelParams(
        market_vol=0.04, idio_vol=0.075,
        style_premia=(0.0008, 0.0015, 0.0025, 0.0010, -0.0020, 0.0010),
        style_persistence=-0.55, char_persistence=0.50,
        leader_period=0, factor_rotate=False, factor_period=0,
        crisis_month=-1,
    ),
    # Leader-group rotation: every 12 months, the stocks whose style
    # exposures earn the largest premia switch.
    "rotating_leaders": PanelParams(
        market_vol=0.04, idio_vol=0.07,
        style_premia=(0.0010, 0.0015, 0.0030, 0.0015, -0.0020, 0.0010),
        style_persistence=0.10, char_persistence=0.90,
        leader_period=12, factor_rotate=False, factor_period=0,
        crisis_month=-1,
    ),
    # Choppy: high idiosyncratic vol, near-zero persistence, premia ~0.
    "choppy": PanelParams(
        market_vol=0.06, idio_vol=0.11,
        style_premia=(0.0003, 0.0004, 0.0005, 0.0003, -0.0005, 0.0003),
        style_persistence=0.05, char_persistence=0.85,
        leader_period=0, factor_rotate=False, factor_period=0,
        crisis_month=-1,
    ),
    # Crisis: correlated drawdown around month 30 + recovery (handled in
    # the generator). Base regime is otherwise close to ``base`` with
    # slightly elevated vol.
    "crisis": PanelParams(
        market_vol=0.05, idio_vol=0.08,
        style_premia=(0.0010, 0.0015, 0.0030, 0.0015, -0.0020, 0.0010),
        style_persistence=0.25, char_persistence=0.90,
        leader_period=0, factor_rotate=False, factor_period=0,
        crisis_month=30,
    ),
    # Factor rotation: style premia sign flips every 18 months for the
    # value + momentum factors. Quality / size unaffected.
    "factor_rotation": PanelParams(
        market_vol=0.04, idio_vol=0.07,
        style_premia=(0.0010, 0.0025, 0.0040, 0.0015, -0.0020, 0.0010),
        style_persistence=0.30, char_persistence=0.92,
        leader_period=0, factor_rotate=True, factor_period=18,
        crisis_month=-1,
    ),
}


def get_panel_params(scenario: str) -> PanelParams:
    """Return :class:`PanelParams` for a scenario name.

    Accepts both bare names (``"trending"``) and ``"future2026_*"`` form.
    """
    short = scenario.replace("future2026_", "")
    if short not in _PARAMS:
        raise ValueError(
            f"Unknown panel scenario {scenario!r}; "
            f"expected one of {sorted(_PARAMS)}"
        )
    return _PARAMS[short]


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def panel_dates() -> pd.DatetimeIndex:
    """Month-end index for the canonical synthetic panel horizon."""
    return pd.date_range(PANEL_START, PANEL_END, freq="ME")


def panel_permnos() -> np.ndarray:
    """Stable synthetic permnos used across every scenario."""
    return np.arange(PERMNO_BASE, PERMNO_BASE + N_PERMNOS, dtype=np.int64)


def panel_seed(scenario: str) -> int:
    """Deterministic seed per scenario so re-generation is reproducible.

    Uses :mod:`zlib.crc32` rather than :func:`hash` so the seed is stable
    across Python processes (``hash()`` is randomized per process when
    ``PYTHONHASHSEED`` is unset).
    """
    import zlib
    blob = f"synthetic_panel::{scenario}".encode("utf-8")
    return int(zlib.crc32(blob)) & 0x7FFFFFFF


def _ar1_path(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a stationary AR(1) path of length ``n`` with given rho/sigma."""
    rho = float(np.clip(rho, -0.99, 0.99))
    eps = rng.normal(0.0, sigma, size=n)
    z = np.empty(n, dtype=np.float64)
    z[0] = eps[0]
    scale = float(np.sqrt(max(1.0 - rho ** 2, 1e-9)))
    for t in range(1, n):
        z[t] = rho * z[t - 1] + scale * eps[t]
    return z


def _ar1_panel(
    n_t: int, n_p: int, rho: float, sigma: float, rng: np.random.Generator,
) -> np.ndarray:
    """Per-stock AR(1) paths for characteristic dynamics.

    Returns an (n_t, n_p) array — column ``i`` is one stock's AR(1) path.
    """
    rho = float(np.clip(rho, -0.99, 0.99))
    eps = rng.normal(0.0, sigma, size=(n_t, n_p))
    z = np.empty((n_t, n_p), dtype=np.float64)
    z[0] = eps[0]
    scale = float(np.sqrt(max(1.0 - rho ** 2, 1e-9)))
    for t in range(1, n_t):
        z[t] = rho * z[t - 1] + scale * eps[t]
    return z


# ─────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────

def generate_panel(scenario: str, seed: int | None = None) -> pd.DataFrame:
    """Generate one stock-level synthetic panel for ``scenario``.

    Parameters
    ----------
    scenario:
        Either a bare scenario name (``"trending"``) or the variant form
        (``"future2026_trending"``).
    seed:
        Override the deterministic per-scenario seed. ``None`` uses the
        default from :func:`panel_seed`, which is stable across runs.

    Returns
    -------
    DataFrame with exactly ``120 * 800 = 96000`` rows and the columns in
    :data:`REQUIRED_COLUMNS`. The frame is sorted by ``(date, permno)``.
    """
    params = get_panel_params(scenario)
    bare = scenario.replace("future2026_", "")
    variant = scenario if scenario.startswith("future2026_") else f"future2026_{bare}"
    if seed is None:
        seed = panel_seed(bare)
    rng = np.random.default_rng(seed)

    dates = panel_dates()
    permnos = panel_permnos()
    n_t, n_p = len(dates), len(permnos)
    assert n_t == N_MONTHS and n_p == N_PERMNOS

    # ── Stock-static traits ───────────────────────────────────────────
    # Market beta around 1.0, slightly fat-tailed.
    market_beta = rng.normal(1.0, 0.35, size=n_p).astype(np.float64)

    # Each stock has a baseline loading on each style factor. We then
    # add slow AR(1) drift on top so characteristics actually move.
    base_loadings = rng.normal(0.0, 1.0, size=(n_p, len(STYLE_FACTORS)))

    # Per-stock characteristic paths: (n_t, n_p) for each style.
    char_paths: Dict[str, np.ndarray] = {}
    char_sigma = 0.30  # innovation scale on the standardized characteristic
    for k, style in enumerate(STYLE_FACTORS):
        drift = _ar1_panel(n_t, n_p, params.char_persistence, char_sigma, rng)
        # Add base loading and slow renormalisation toward zero mean per date.
        path = base_loadings[None, :, k] + drift
        # Z-score within each cross-section so it reads like a clean characteristic.
        mean = path.mean(axis=1, keepdims=True)
        std = path.std(axis=1, keepdims=True) + 1e-9
        char_paths[style] = (path - mean) / std

    # ── Style premia paths ────────────────────────────────────────────
    # Each style has an AR(1) realised premium centred on its baseline.
    style_premia = np.array(params.style_premia, dtype=np.float64)
    style_returns = np.empty((n_t, len(STYLE_FACTORS)), dtype=np.float64)
    for k in range(len(STYLE_FACTORS)):
        # AR(1) deviation around the baseline premium.
        dev_sigma = 0.015 if bare != "choppy" else 0.025
        dev = _ar1_path(n_t, params.style_persistence, dev_sigma, rng)
        style_returns[:, k] = style_premia[k] + dev

    # Factor rotation: flip sign of value + momentum (idx 1, 2) every
    # `factor_period` months.
    if params.factor_rotate and params.factor_period > 0:
        for t0 in range(0, n_t, params.factor_period):
            if (t0 // params.factor_period) % 2 == 1:
                t1 = min(t0 + params.factor_period, n_t)
                style_returns[t0:t1, 1] *= -1.0
                style_returns[t0:t1, 2] *= -1.0

    # Leader rotation: at each rotation boundary, permute style premia
    # across factor slots so a different combination of styles leads.
    if params.leader_period > 0:
        for t0 in range(0, n_t, params.leader_period):
            t1 = min(t0 + params.leader_period, n_t)
            perm = rng.permutation(len(STYLE_FACTORS))
            style_returns[t0:t1] = style_returns[t0:t1][:, perm]

    # ── Common (market) factor ────────────────────────────────────────
    market = rng.normal(0.006, params.market_vol, size=n_t)
    if params.crisis_month >= 0 and params.crisis_month < n_t:
        cm = params.crisis_month
        market[cm] -= 0.22
        if cm + 1 < n_t:
            market[cm + 1] -= 0.06
        for k in range(1, 9):
            if cm + k < n_t:
                market[cm + k] += 0.03 * float(np.exp(-k / 4.0))

    # ── Realised returns ──────────────────────────────────────────────
    # ret_{i,t} = beta_i * mkt_t + sum_k (char_{i,k,t} * style_ret_{k,t}) + idio
    # Expected (latent) is the same minus the idio noise.
    chars_stack = np.stack([char_paths[s] for s in STYLE_FACTORS], axis=-1)  # (T, N, K)
    style_contrib = np.einsum("tnk,tk->tn", chars_stack, style_returns)  # (T, N)
    mkt_contrib = np.outer(market, market_beta)  # (T, N)
    latent = mkt_contrib + style_contrib

    idio = rng.normal(0.0, params.idio_vol, size=(n_t, n_p))
    if params.crisis_month >= 0:
        # During the crisis month, idio dispersion shrinks: stocks all
        # drop with the market.
        cm = params.crisis_month
        idio[cm] *= 0.5

    ret = latent + idio

    # ── Model-signal columns ──────────────────────────────────────────
    # These emulate model predictions of varying signal strength so the
    # generator can sort stocks into deciles without needing a separate
    # prediction stage. They are NOT real model outputs.
    z_latent = (latent - latent.mean(axis=1, keepdims=True)) / (
        latent.std(axis=1, keepdims=True) + 1e-9
    )
    sig_strong = 0.75 * z_latent + np.sqrt(1.0 - 0.75 ** 2) * rng.normal(0, 1, size=(n_t, n_p))
    sig_medium = 0.45 * z_latent + np.sqrt(1.0 - 0.45 ** 2) * rng.normal(0, 1, size=(n_t, n_p))
    sig_weak = 0.15 * z_latent + np.sqrt(1.0 - 0.15 ** 2) * rng.normal(0, 1, size=(n_t, n_p))

    # ── Assemble the long frame ───────────────────────────────────────
    date_col = np.repeat(dates.values, n_p)
    permno_col = np.tile(permnos, n_t)
    beta_col = np.tile(market_beta, n_t)
    mkt_col = np.repeat(market, n_p)

    frame = pd.DataFrame({
        "date": date_col,
        "permno": permno_col,
        "scenario": variant,
        "ret": ret.reshape(-1).astype(np.float64),
        "mkt_ret": mkt_col.astype(np.float64),
        "common_factor": mkt_col.astype(np.float64),
        "latent_expected_ret": latent.reshape(-1).astype(np.float64),
        "market_beta": beta_col.astype(np.float64),
        "size": char_paths["size"].reshape(-1).astype(np.float64),
        "value": char_paths["value"].reshape(-1).astype(np.float64),
        "momentum": char_paths["momentum"].reshape(-1).astype(np.float64),
        "quality": char_paths["quality"].reshape(-1).astype(np.float64),
        "volatility": char_paths["volatility"].reshape(-1).astype(np.float64),
        "liquidity": char_paths["liquidity"].reshape(-1).astype(np.float64),
        "model_signal_strong": sig_strong.reshape(-1).astype(np.float64),
        "model_signal_medium": sig_medium.reshape(-1).astype(np.float64),
        "model_signal_weak": sig_weak.reshape(-1).astype(np.float64),
    })
    frame.sort_values(["date", "permno"], inplace=True, kind="stable")
    frame.reset_index(drop=True, inplace=True)
    return frame


# ─────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────

def panel_path(scenario: str, root: Path | str | None = None) -> Path:
    """Default parquet path for a scenario's panel.

    Repo-relative default: ``data/cache/synthetic_panels/<variant>.parquet``.
    """
    if root is None:
        root = Path("data") / "cache" / "synthetic_panels"
    root = Path(root)
    short = scenario.replace("future2026_", "")
    variant = scenario if scenario.startswith("future2026_") else f"future2026_{short}"
    return root / f"{variant}.parquet"


def write_panel(
    scenario: str,
    panel: pd.DataFrame,
    out_path: Path | str | None = None,
) -> Path:
    """Persist ``panel`` to parquet, creating directories as needed."""
    out_path = Path(out_path) if out_path is not None else panel_path(scenario)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    return out_path


def load_panel(
    scenario: str,
    in_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load a previously-generated parquet panel for a scenario."""
    in_path = Path(in_path) if in_path is not None else panel_path(scenario)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Panel parquet not found at {in_path}. "
            "Run `python generate_synthetic_results.py --panels-only "
            "--variant future2026_all` to (re)generate."
        )
    df = pd.read_parquet(in_path)
    df.sort_values(["date", "permno"], inplace=True, kind="stable")
    df.reset_index(drop=True, inplace=True)
    return df


def generate_all_panels(
    out_root: Path | str | None = None,
    scenarios: Tuple[str, ...] = SCENARIOS,
) -> Dict[str, Path]:
    """Generate and persist a panel for every scenario.

    Returns a mapping ``variant_name -> parquet_path``.
    """
    out_root = Path(out_root) if out_root is not None else Path("data") / "cache" / "synthetic_panels"
    out_root.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for bare in scenarios:
        variant = f"future2026_{bare}" if not bare.startswith("future2026_") else bare
        panel = generate_panel(bare)
        path = out_root / f"{variant}.parquet"
        write_panel(variant, panel, out_path=path)
        paths[variant] = path
    return paths


# ─────────────────────────────────────────────────────────────────────
# Decile + signal helpers
# ─────────────────────────────────────────────────────────────────────

def assign_deciles(
    panel: pd.DataFrame,
    signal_col: str,
    n_deciles: int = 10,
) -> pd.Series:
    """Per-date decile assignment (1..n_deciles, 10 = top signal)."""
    if signal_col not in panel.columns:
        raise KeyError(f"missing signal column {signal_col!r}")

    def _rank(group: pd.Series) -> pd.Series:
        ranks = group.rank(method="first", na_option="keep")
        # qcut on float rank to guarantee equal-sized bins.
        try:
            bins = pd.qcut(ranks, n_deciles, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.Series(np.full(len(group), np.nan), index=group.index)
        return bins.astype("Int64")

    deciles = panel.groupby("date", group_keys=False)[signal_col].apply(_rank)
    deciles.name = "decile"
    return deciles


def decile_returns_from_panel(
    panel: pd.DataFrame,
    signal_col: str,
    n_deciles: int = 10,
) -> Mapping[str, pd.Series]:
    """Per-month decile mean returns + H-L spread, derived from the panel.

    Returns a dict keyed by decile name (``"1"``..``"10"`` plus ``"H-L"``)
    of pandas Series indexed by month-end date.
    """
    work = panel[["date", "permno", "ret", signal_col]].copy()
    work["decile"] = assign_deciles(work, signal_col, n_deciles=n_deciles).values
    work = work.dropna(subset=["decile"])
    work["decile"] = work["decile"].astype(int)
    grp = work.groupby(["date", "decile"])["ret"].mean().unstack("decile")
    grp = grp.sort_index()
    out: Dict[str, pd.Series] = {}
    for d in range(1, n_deciles + 1):
        if d in grp.columns:
            out[str(d)] = grp[d].rename(str(d))
    out["H-L"] = (out[str(n_deciles)] - out["1"]).rename("H-L")
    return out
