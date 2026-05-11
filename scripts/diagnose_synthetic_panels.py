"""
scripts/diagnose_synthetic_panels.py
------------------------------------
Validate that ``data/cache/synthetic_panels/*.parquet`` stock-level
panels exhibit market and cross-sectional behaviour consistent with
their scenario label *before* downstream code consumes them.

The diagnostics are computed from the realised panel data only — never
from the scenario's target parameters. This is the point: a panel can
silently drift from its label (wrong seed, wrong knob, corrupted
parquet, schema change in :mod:`src.synthetic.panels`), so we compute
observable statistics and emit warnings when they violate
scenario-specific expectations.

Outputs
-------
* A CSV row per scenario panel with one column per diagnostic.
* (Optional) A markdown summary with the same content rendered as a
  human-readable report.

Refuse to claim that downstream results are valid if any panel emits a
warning that contradicts its scenario label.

This script does NOT use WRDS or any credentials.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Constants & defaults
# ─────────────────────────────────────────────────────────────────────

DEFAULT_PANEL_ROOT = Path("data") / "cache" / "synthetic_panels"
DEFAULT_OUTPUT = Path("outputs") / "synthetic_panel_diagnostics.csv"
ANNUALIZATION = 12.0  # monthly panel -> annualised vol

# Conservative scenario thresholds.  These are deliberately *loose* — we
# only flag if the panel deviates in the wrong direction by a large
# margin, since the panels are drawn with finite samples and a single
# tight threshold would generate spurious warnings.
CRISIS_MAX_WORST_MONTH = -0.15      # crisis: worst market month must be < -15%
CRISIS_MAX_DD_PCT = -0.20           # crisis: market drawdown must be < -20%
CHOPPY_VOL_RATIO_VS_BASE = 1.15     # choppy: vol >= 1.15x of base
TRENDING_MIN_RANK_AC = 0.10         # trending: signal rank AR(1) >= +0.10
MEAN_REVERSION_MAX_RANK_AC = 0.05   # mean_reversion: rank AR(1) <= +0.05
MEAN_REVERSION_MIN_REVERSAL = 0.0   # mean_reversion: reversal spread > 0
ROTATING_MIN_RANK_CHURN = 0.20      # rotating_leaders: avg per-month rank churn fraction
FACTOR_ROTATION_MIN_SIGN_CHANGES = 3  # factor_rotation: at least 3 sign changes in value or momentum H-L

# How many panel columns we need for full diagnostics.  Anything beyond
# this is treated as best-effort.
REQUIRED_PANEL_COLS = ("date", "permno", "ret", "mkt_ret")
OPTIONAL_PANEL_COLS = (
    "scenario", "common_factor", "latent_expected_ret", "market_beta",
    "size", "value", "momentum", "quality", "volatility", "liquidity",
    "model_signal_strong", "model_signal_medium", "model_signal_weak",
)

DIAG_COLUMNS: Tuple[str, ...] = (
    "variant",
    "scenario",
    "n_rows",
    "n_months",
    "n_permnos",
    "start_date",
    "end_date",
    "market_mean_monthly",
    "market_vol_monthly",
    "market_ann_vol",
    "market_max_dd_pct",
    "market_worst_month",
    "market_best_month",
    "avg_cross_sectional_dispersion",
    "avg_abs_pairwise_corr_or_factor_proxy",
    "rank_autocorr_1m",
    "momentum_spread_1m",
    "reversal_spread_1m",
    "factor_value_corr",
    "factor_momentum_corr",
    "crisis_min_market_month",
    "notes",
    "warnings",
)


# ─────────────────────────────────────────────────────────────────────
# Diagnostic record
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PanelDiagnostics:
    """Diagnostics computed from a single panel parquet."""

    variant: str
    scenario: str
    n_rows: int
    n_months: int
    n_permnos: int
    start_date: str
    end_date: str
    market_mean_monthly: float
    market_vol_monthly: float
    market_ann_vol: float
    market_max_dd_pct: float
    market_worst_month: float
    market_best_month: float
    avg_cross_sectional_dispersion: float
    avg_abs_pairwise_corr_or_factor_proxy: float
    rank_autocorr_1m: float
    momentum_spread_1m: float
    reversal_spread_1m: float
    factor_value_corr: float
    factor_momentum_corr: float
    crisis_min_market_month: float
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_row(self) -> Dict[str, object]:
        return {
            "variant": self.variant,
            "scenario": self.scenario,
            "n_rows": self.n_rows,
            "n_months": self.n_months,
            "n_permnos": self.n_permnos,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "market_mean_monthly": self.market_mean_monthly,
            "market_vol_monthly": self.market_vol_monthly,
            "market_ann_vol": self.market_ann_vol,
            "market_max_dd_pct": self.market_max_dd_pct,
            "market_worst_month": self.market_worst_month,
            "market_best_month": self.market_best_month,
            "avg_cross_sectional_dispersion": self.avg_cross_sectional_dispersion,
            "avg_abs_pairwise_corr_or_factor_proxy": self.avg_abs_pairwise_corr_or_factor_proxy,
            "rank_autocorr_1m": self.rank_autocorr_1m,
            "momentum_spread_1m": self.momentum_spread_1m,
            "reversal_spread_1m": self.reversal_spread_1m,
            "factor_value_corr": self.factor_value_corr,
            "factor_momentum_corr": self.factor_momentum_corr,
            "crisis_min_market_month": self.crisis_min_market_month,
            "notes": "; ".join(self.notes),
            "warnings": "; ".join(self.warnings),
        }


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation that returns NaN on degenerate inputs."""
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return float("nan")
    return float(a.corr(b))


def _bare_scenario(variant: str) -> str:
    """Strip the ``future2026_`` prefix if present."""
    return variant.replace("future2026_", "")


def _read_panel(parquet: Path) -> pd.DataFrame:
    """Read a panel parquet and sort by (date, permno)."""
    df = pd.read_parquet(parquet)
    missing = [c for c in REQUIRED_PANEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{parquet}: missing required columns {missing}")
    df = df.sort_values(["date", "permno"], kind="stable").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────────────────────────────
# Diagnostic computation
# ─────────────────────────────────────────────────────────────────────

def compute_market_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Market-level moments derived from the panel's ``mkt_ret`` column."""
    mkt = df.groupby("date")["mkt_ret"].first().sort_index()
    nav = (1.0 + mkt).cumprod()
    drawdown = (nav / nav.cummax()) - 1.0
    return {
        "market_mean_monthly": float(mkt.mean()),
        "market_vol_monthly": float(mkt.std(ddof=0)),
        "market_ann_vol": float(mkt.std(ddof=0) * np.sqrt(ANNUALIZATION)),
        "market_max_dd_pct": float(drawdown.min()),
        "market_worst_month": float(mkt.min()),
        "market_best_month": float(mkt.max()),
        "crisis_min_market_month": float(mkt.min()),
    }


def compute_cross_sectional_dispersion(df: pd.DataFrame) -> float:
    """Average cross-sectional standard deviation of returns."""
    return float(df.groupby("date")["ret"].std(ddof=0).mean())


def compute_factor_proxy(df: pd.DataFrame) -> float:
    """
    Average absolute pairwise correlation between stock returns, *or*,
    when the per-stock series are too noisy to estimate directly, the
    average |corr| between each stock and the panel's market factor.

    The market-factor proxy is a fast lower-bound on average pairwise
    correlation (if every stock has factor loading β and idio noise,
    pairwise corr ≈ β²·var(mkt) / (β²·var(mkt) + var(idio))).
    """
    if "mkt_ret" not in df.columns:
        return float("nan")
    # Pivot to (date × permno) returns.
    wide = df.pivot(index="date", columns="permno", values="ret")
    mkt = df.groupby("date")["mkt_ret"].first().loc[wide.index]
    if len(wide.columns) == 0 or len(mkt) < 3:
        return float("nan")
    # |corr| of each stock with the market factor.  This is the proxy.
    corrs = wide.apply(lambda col: _safe_corr(col, mkt), axis=0)
    return float(corrs.abs().mean())


def compute_rank_autocorr(df: pd.DataFrame, signal_col: str = "latent_expected_ret") -> float:
    """Average lag-1 autocorrelation of per-stock cross-sectional ranks.

    Uses the latent expected return when available so the diagnostic
    isolates *signal* persistence from idio noise.  Falls back to the
    realised return rank when the latent column is missing.
    """
    if signal_col not in df.columns:
        signal_col = "ret"
    ranks = df.groupby("date")[signal_col].rank(method="average")
    work = df[["date", "permno"]].copy()
    work["rank"] = ranks.values
    wide = work.pivot(index="date", columns="permno", values="rank").sort_index()
    if wide.shape[0] < 3:
        return float("nan")
    # Per-stock AR(1) on rank time series, averaged across stocks.
    def _ac1(col: pd.Series) -> float:
        v = col.dropna().values
        if len(v) < 3:
            return float("nan")
        c = np.corrcoef(v[:-1], v[1:])[0, 1]
        return float(c) if np.isfinite(c) else float("nan")

    ac = wide.apply(_ac1, axis=0)
    return float(ac.mean(skipna=True))


def compute_momentum_reversal_spread(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute simple 1-month momentum and reversal H-L spreads.

    Both are derived from realised panel data only:
      * momentum_spread_1m — rank stocks at month ``t`` on ``ret_t``,
        compute the mean return at ``t+1`` of the top vs bottom decile.
        Positive ⇒ winners keep winning.
      * reversal_spread_1m — rank stocks at month ``t`` on ``ret_t``,
        compute the mean return at ``t+1`` of the **bottom** decile
        minus the top decile.  Positive ⇒ losers bounce back.
    """
    work = df[["date", "permno", "ret"]].copy()
    work = work.sort_values(["permno", "date"])
    work["ret_next"] = work.groupby("permno")["ret"].shift(-1)
    work = work.dropna(subset=["ret_next"])
    # Rank within each formation month.
    work["decile"] = (
        work.groupby("date")["ret"]
        .rank(method="first", pct=True)
        .mul(10.0)
        .clip(upper=9.999)
        .astype(int)
        + 1
    )
    grp = work.groupby(["date", "decile"])["ret_next"].mean().unstack("decile")
    if grp.shape[0] == 0 or 1 not in grp.columns or 10 not in grp.columns:
        return float("nan"), float("nan")
    momentum = float((grp[10] - grp[1]).mean())
    reversal = float((grp[1] - grp[10]).mean())
    return momentum, reversal


def compute_factor_correlations(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Per-month cross-sectional correlation between a characteristic and
    realised return, averaged over months.  Captures the *direction*
    of the factor premium so that ``factor_rotation`` can be detected
    via sign changes — the diagnostic itself returns the time-average.
    """
    out: Dict[str, float] = {}
    for col in ("value", "momentum"):
        if col not in df.columns:
            out[col] = float("nan")
            continue
        corrs = (
            df.groupby("date")
            .apply(lambda g, c=col: _safe_corr(g[c], g["ret"]))
        )
        out[col] = float(corrs.mean(skipna=True))
    return out.get("value", float("nan")), out.get("momentum", float("nan"))


def compute_factor_sign_changes(df: pd.DataFrame, col: str) -> int:
    """Count sign changes in the per-month cross-sectional ``col~ret`` correlation."""
    if col not in df.columns:
        return 0
    corrs = (
        df.groupby("date")
        .apply(lambda g: _safe_corr(g[col], g["ret"]))
        .dropna()
        .values
    )
    if len(corrs) < 2:
        return 0
    signs = np.sign(corrs)
    return int(np.sum(signs[1:] != signs[:-1]))


def compute_rank_churn(df: pd.DataFrame, signal_col: str = "latent_expected_ret") -> float:
    """
    Average fraction of stocks whose decile assignment changes
    month-over-month.  Higher values ⇒ leadership rotates more.
    """
    if signal_col not in df.columns:
        signal_col = "ret"
    work = df[["date", "permno", signal_col]].copy()
    work["decile"] = (
        work.groupby("date")[signal_col]
        .rank(method="first", pct=True)
        .mul(10.0)
        .clip(upper=9.999)
        .astype(int)
        + 1
    )
    wide = work.pivot(index="date", columns="permno", values="decile").sort_index()
    if wide.shape[0] < 2:
        return float("nan")
    diffs = (wide.diff() != 0).iloc[1:].mean(axis=1)
    return float(diffs.mean())


# ─────────────────────────────────────────────────────────────────────
# Scenario validation
# ─────────────────────────────────────────────────────────────────────

def evaluate_scenario_flags(
    diag: PanelDiagnostics,
    bare_scenario: str,
    df: pd.DataFrame,
    base_market_vol: Optional[float] = None,
) -> None:
    """Apply scenario-specific checks and append notes/warnings in-place.

    Thresholds are deliberately conservative — see module docstring and
    the constants near the top of the file for the exact values used.
    """

    if bare_scenario == "crisis":
        if diag.market_worst_month > CRISIS_MAX_WORST_MONTH:
            diag.warnings.append(
                f"crisis: worst market month {diag.market_worst_month:.4f} "
                f"is not below {CRISIS_MAX_WORST_MONTH:.2f}"
            )
        if diag.market_max_dd_pct > CRISIS_MAX_DD_PCT:
            diag.warnings.append(
                f"crisis: market drawdown {diag.market_max_dd_pct:.4f} "
                f"is not below {CRISIS_MAX_DD_PCT:.2f}"
            )
        diag.notes.append(
            f"crisis check: worst_month={diag.market_worst_month:.4f}, "
            f"max_dd={diag.market_max_dd_pct:.4f}"
        )

    elif bare_scenario == "choppy":
        if base_market_vol is not None and base_market_vol > 0:
            ratio = diag.market_vol_monthly / base_market_vol
            diag.notes.append(
                f"choppy/base market vol ratio={ratio:.2f} "
                f"(threshold>={CHOPPY_VOL_RATIO_VS_BASE})"
            )
            if ratio < CHOPPY_VOL_RATIO_VS_BASE:
                diag.warnings.append(
                    f"choppy: market vol ratio vs base ({ratio:.2f}) "
                    f"below threshold {CHOPPY_VOL_RATIO_VS_BASE}"
                )
        else:
            diag.notes.append("choppy: base panel unavailable for vol comparison")
        # Also expect low rank persistence under choppy.
        if np.isfinite(diag.rank_autocorr_1m) and diag.rank_autocorr_1m > 0.85:
            diag.warnings.append(
                f"choppy: rank AR(1) {diag.rank_autocorr_1m:.3f} is "
                "unexpectedly high"
            )

    elif bare_scenario == "trending":
        if np.isfinite(diag.rank_autocorr_1m) and diag.rank_autocorr_1m < TRENDING_MIN_RANK_AC:
            diag.warnings.append(
                f"trending: rank AR(1) {diag.rank_autocorr_1m:.3f} "
                f"below threshold {TRENDING_MIN_RANK_AC}"
            )
        if np.isfinite(diag.momentum_spread_1m) and diag.momentum_spread_1m <= 0:
            diag.warnings.append(
                f"trending: momentum spread {diag.momentum_spread_1m:.4f} "
                "is not positive"
            )

    elif bare_scenario == "mean_reversion":
        if np.isfinite(diag.rank_autocorr_1m) and diag.rank_autocorr_1m > MEAN_REVERSION_MAX_RANK_AC:
            diag.warnings.append(
                f"mean_reversion: rank AR(1) {diag.rank_autocorr_1m:.3f} "
                f"above threshold {MEAN_REVERSION_MAX_RANK_AC}"
            )
        if np.isfinite(diag.reversal_spread_1m) and diag.reversal_spread_1m <= MEAN_REVERSION_MIN_REVERSAL:
            diag.warnings.append(
                f"mean_reversion: reversal spread {diag.reversal_spread_1m:.4f} "
                "is not positive"
            )

    elif bare_scenario == "rotating_leaders":
        churn = compute_rank_churn(df)
        diag.notes.append(f"rotating_leaders: rank churn={churn:.3f}")
        if np.isfinite(churn) and churn < ROTATING_MIN_RANK_CHURN:
            diag.warnings.append(
                f"rotating_leaders: rank churn {churn:.3f} below "
                f"threshold {ROTATING_MIN_RANK_CHURN}"
            )

    elif bare_scenario == "factor_rotation":
        v_sign = compute_factor_sign_changes(df, "value")
        m_sign = compute_factor_sign_changes(df, "momentum")
        diag.notes.append(
            f"factor_rotation: value_sign_changes={v_sign}, "
            f"momentum_sign_changes={m_sign}"
        )
        if max(v_sign, m_sign) < FACTOR_ROTATION_MIN_SIGN_CHANGES:
            diag.warnings.append(
                f"factor_rotation: too few factor sign changes "
                f"(value={v_sign}, momentum={m_sign}; "
                f"threshold>={FACTOR_ROTATION_MIN_SIGN_CHANGES})"
            )

    elif bare_scenario == "base":
        # Baseline panel: just a smoke check that nothing is pathological.
        if not np.isfinite(diag.market_mean_monthly):
            diag.warnings.append("base: market mean is non-finite")

    else:
        diag.notes.append(f"no scenario-specific checks for '{bare_scenario}'")


# ─────────────────────────────────────────────────────────────────────
# Panel diagnostics driver
# ─────────────────────────────────────────────────────────────────────

def diagnose_panel(
    parquet: Path,
    base_market_vol: Optional[float] = None,
) -> PanelDiagnostics:
    """Compute the full diagnostic record for one panel parquet."""
    df = _read_panel(parquet)

    variant = parquet.stem
    scenario_label = (
        str(df["scenario"].iloc[0])
        if "scenario" in df.columns and len(df) > 0
        else variant
    )

    market = compute_market_stats(df)
    cs_disp = compute_cross_sectional_dispersion(df)
    factor_proxy = compute_factor_proxy(df)
    rank_ac = compute_rank_autocorr(df)
    momentum, reversal = compute_momentum_reversal_spread(df)
    fv_corr, fm_corr = compute_factor_correlations(df)

    diag = PanelDiagnostics(
        variant=variant,
        scenario=scenario_label,
        n_rows=int(len(df)),
        n_months=int(df["date"].nunique()),
        n_permnos=int(df["permno"].nunique()),
        start_date=str(df["date"].min().date()),
        end_date=str(df["date"].max().date()),
        market_mean_monthly=market["market_mean_monthly"],
        market_vol_monthly=market["market_vol_monthly"],
        market_ann_vol=market["market_ann_vol"],
        market_max_dd_pct=market["market_max_dd_pct"],
        market_worst_month=market["market_worst_month"],
        market_best_month=market["market_best_month"],
        avg_cross_sectional_dispersion=cs_disp,
        avg_abs_pairwise_corr_or_factor_proxy=factor_proxy,
        rank_autocorr_1m=rank_ac,
        momentum_spread_1m=momentum,
        reversal_spread_1m=reversal,
        factor_value_corr=fv_corr,
        factor_momentum_corr=fm_corr,
        crisis_min_market_month=market["crisis_min_market_month"],
    )

    bare = _bare_scenario(variant)
    evaluate_scenario_flags(diag, bare, df, base_market_vol=base_market_vol)
    return diag


def find_panel_paths(
    panel_root: Path,
    variant: Optional[str] = None,
) -> List[Path]:
    """Resolve which parquet files to diagnose."""
    panel_root = Path(panel_root)
    if not panel_root.exists():
        raise FileNotFoundError(
            f"panel root {panel_root} does not exist. "
            "Run `python generate_synthetic_results.py --variant future2026_all "
            "--panels-only` first."
        )

    if variant and variant.lower() != "all":
        # Allow both bare and prefixed names.
        bare = _bare_scenario(variant)
        candidates = [
            panel_root / f"future2026_{bare}.parquet",
            panel_root / f"{variant}.parquet",
        ]
        for p in candidates:
            if p.exists():
                return [p]
        raise FileNotFoundError(
            f"no panel parquet for variant {variant!r} under {panel_root}; "
            f"looked for: {[str(c) for c in candidates]}"
        )

    paths = sorted(panel_root.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no .parquet files under {panel_root}. "
            "Generate panels first via `python generate_synthetic_results.py "
            "--variant future2026_all --panels-only`."
        )
    return paths


def diagnose_panels(
    panel_root: Path,
    variant: Optional[str] = None,
) -> pd.DataFrame:
    """Diagnose all matching panels and return a DataFrame keyed by variant."""
    paths = find_panel_paths(panel_root, variant=variant)

    # Compute base market vol first (if available) so choppy can be
    # compared against it.
    base_market_vol: Optional[float] = None
    base_path = Path(panel_root) / "future2026_base.parquet"
    if base_path.exists():
        try:
            base_df = _read_panel(base_path)
            base_market_vol = float(
                base_df.groupby("date")["mkt_ret"].first().std(ddof=0)
            )
        except Exception:
            base_market_vol = None

    rows: List[Dict[str, object]] = []
    for path in paths:
        diag = diagnose_panel(path, base_market_vol=base_market_vol)
        rows.append(diag.to_row())

    df = pd.DataFrame(rows, columns=list(DIAG_COLUMNS))
    return df


# ─────────────────────────────────────────────────────────────────────
# Markdown summary
# ─────────────────────────────────────────────────────────────────────

def render_markdown_summary(df: pd.DataFrame) -> str:
    """Render the diagnostic DataFrame as a readable markdown report."""
    lines: List[str] = []
    lines.append("# Synthetic panel diagnostics")
    lines.append("")
    lines.append(
        "Diagnostics derived from `data/cache/synthetic_panels/*.parquet`. "
        "Values are computed from realised panel data only — not from "
        "scenario target parameters."
    )
    lines.append("")
    for _, row in df.iterrows():
        lines.append(f"## {row['variant']}")
        lines.append("")
        lines.append(
            f"- rows: {int(row['n_rows'])}, months: {int(row['n_months'])}, "
            f"permnos: {int(row['n_permnos'])}"
        )
        lines.append(f"- date range: {row['start_date']} → {row['end_date']}")
        lines.append(
            f"- market: mean(monthly)={row['market_mean_monthly']:.4f}, "
            f"vol(monthly)={row['market_vol_monthly']:.4f}, "
            f"ann.vol={row['market_ann_vol']:.4f}"
        )
        lines.append(
            f"- market drawdown: max_dd={row['market_max_dd_pct']:.4f}, "
            f"worst_month={row['market_worst_month']:.4f}, "
            f"best_month={row['market_best_month']:.4f}"
        )
        lines.append(
            f"- cross-sectional dispersion (avg std of ret per date): "
            f"{row['avg_cross_sectional_dispersion']:.4f}"
        )
        lines.append(
            f"- factor proxy |corr(stock, mkt)| avg: "
            f"{row['avg_abs_pairwise_corr_or_factor_proxy']:.4f}"
        )
        lines.append(
            f"- rank AR(1) (latent signal): {row['rank_autocorr_1m']:.4f}"
        )
        lines.append(
            f"- momentum spread (1m fwd): {row['momentum_spread_1m']:.4f}, "
            f"reversal spread (1m fwd): {row['reversal_spread_1m']:.4f}"
        )
        lines.append(
            f"- factor corrs: value~ret={row['factor_value_corr']:.4f}, "
            f"momentum~ret={row['factor_momentum_corr']:.4f}"
        )
        if row.get("notes"):
            lines.append(f"- notes: {row['notes']}")
        warn = row.get("warnings", "") or ""
        if warn:
            lines.append(f"- **warnings**: {warn}")
        else:
            lines.append("- warnings: none")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute observable diagnostics on synthetic panel parquets "
            "and emit warnings when a panel's behaviour contradicts its "
            "scenario label."
        ),
    )
    p.add_argument(
        "--panel-root",
        type=Path,
        default=DEFAULT_PANEL_ROOT,
        help="Directory containing future2026_*.parquet panel files "
             f"(default: {DEFAULT_PANEL_ROOT}).",
    )
    p.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Restrict diagnostics to one variant (bare name like "
             "'crisis' or full 'future2026_crisis'). Default: all panels.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=None,
        help="Optional markdown summary path (e.g. "
             "outputs/synthetic_panel_diagnostics.md).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any panel emits a warning.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    df = diagnose_panels(args.panel_root, variant=args.variant)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")

    if args.summary_md is not None:
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text(render_markdown_summary(df))
        print(f"wrote markdown summary to {args.summary_md}")

    # Print one-line per-variant summary.
    for _, row in df.iterrows():
        warn = row.get("warnings", "") or ""
        status = "OK " if not warn else "WARN"
        print(
            f"[{status}] {row['variant']}: "
            f"mkt_vol={row['market_vol_monthly']:.4f} "
            f"max_dd={row['market_max_dd_pct']:.4f} "
            f"rank_ac={row['rank_autocorr_1m']:.3f} "
            f"mom={row['momentum_spread_1m']:.4f} "
            f"rev={row['reversal_spread_1m']:.4f}"
            + (f" :: {warn}" if warn else "")
        )

    if args.strict and df["warnings"].astype(str).str.len().sum() > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
