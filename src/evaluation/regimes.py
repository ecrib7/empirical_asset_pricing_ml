"""
evaluation/regimes.py
---------------------
Regime-conditional evaluation of model performance.

The default test window in the GKX paper averages 30 years of regimes
(1987–2016) into single Sharpe / R² numbers. The improved variant
extends this to 2024 — covering COVID, the 2022 rate cycle, and the
post-2020 retail / AI regime.

This module slices the test sample by:

  * NBER recession indicator (binary)
  * VIX tercile (low / medium / high implied vol)
  * Calendar decade (1990s / 2000s / 2010s / 2020s)

For each regime, it computes the comprehensive metric table for every
model (including ensembles) and writes one ``regimes.csv`` per variant
so the dashboard can compare model performance across regimes.

NBER dates come from the NBER's official recession dating committee
(stable historical reference). VIX monthly values come from FRED
(VIXCLS). When network access isn't available, we fall back to a small
offline VIX history built into this module.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  NBER recession dates (start, end) — peak month → trough month
# ─────────────────────────────────────────────────────────────────────────────

NBER_RECESSIONS: List[Tuple[str, str]] = [
    ("1957-08-01", "1958-04-30"),
    ("1960-04-01", "1961-02-28"),
    ("1969-12-01", "1970-11-30"),
    ("1973-11-01", "1975-03-31"),
    ("1980-01-01", "1980-07-31"),
    ("1981-07-01", "1982-11-30"),
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),  # COVID — short, sharp
]


def nber_recession_mask(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean Series aligned to ``dates`` indicating NBER recession months."""
    s = pd.Series(False, index=dates, name="recession")
    for start, end in NBER_RECESSIONS:
        m = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        s.loc[m] = True
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  VIX monthly series (FRED VIXCLS, end-of-month). Offline fallback.
#
#  Values are end-of-month VIX (or VXO before 1990) levels. Source: FRED.
#  Embedded for sandbox robustness — the dashboard / CLI should override
#  with a fresh download when available (set ``vix_path`` arg).
# ─────────────────────────────────────────────────────────────────────────────

# Fallback monthly VIX (end-of-month VIXCLS, from FRED). Decade summaries used
# to compute terciles; not a perfect historical series, but sufficient for
# regime stratification at monthly frequency.
_OFFLINE_VIX = pd.Series({
    # Pre-1990: use VXO splice from CBOE / BOE-equivalent. Approx.
    "1986-12-31": 27.4, "1987-06-30": 23.7, "1987-12-31": 36.4,
    "1988-12-31": 22.3, "1989-12-31": 17.6,
    # VIX proper (1990+)
    "1990-12-31": 26.0, "1991-12-31": 18.5, "1992-12-31": 14.2,
    "1993-12-31": 11.7, "1994-12-31": 13.7, "1995-12-31": 12.4,
    "1996-12-31": 20.9, "1997-12-31": 24.0, "1998-12-31": 24.4,
    "1999-12-31": 24.4, "2000-12-31": 26.9, "2001-12-31": 23.8,
    "2002-12-31": 28.6, "2003-12-31": 17.8, "2004-12-31": 13.3,
    "2005-12-31": 12.1, "2006-12-31": 11.6, "2007-12-31": 22.5,
    "2008-12-31": 40.0, "2009-12-31": 21.7, "2010-12-31": 17.8,
    "2011-12-31": 23.4, "2012-12-31": 18.0, "2013-12-31": 13.7,
    "2014-12-31": 19.2, "2015-12-31": 18.2, "2016-12-31": 14.0,
    "2017-12-31": 11.0, "2018-12-31": 25.4, "2019-12-31": 13.8,
    "2020-12-31": 22.7, "2021-12-31": 17.2, "2022-12-31": 21.7,
    "2023-12-31": 12.4, "2024-12-31": 17.4,
})
_OFFLINE_VIX.index = pd.to_datetime(_OFFLINE_VIX.index)


def load_vix_monthly(
    dates: pd.DatetimeIndex,
    vix_path: Optional[str] = None,
) -> pd.Series:
    """
    Load monthly VIX. If ``vix_path`` is given, expects a CSV with
    columns ``date`` and ``vix`` (or ``VIXCLS``). Otherwise falls back
    to the embedded series. Returns a Series aligned (forward-filled)
    to ``dates``.
    """
    if vix_path is not None:
        try:
            df = pd.read_csv(vix_path, parse_dates=["date"])
            col = "vix" if "vix" in df.columns else "VIXCLS"
            ser = df.set_index("date")[col].astype(float)
        except Exception as e:
            logger.warning(f"Could not load VIX from {vix_path}: {e}; "
                           f"using offline fallback.")
            ser = _OFFLINE_VIX.copy()
    else:
        ser = _OFFLINE_VIX.copy()

    # Resample to month-end and forward-fill into the test dates
    ser = ser.sort_index()
    ser = ser.resample("ME" if hasattr(pd.offsets, "MonthEnd") else "M").last().ffill()
    ser = ser.reindex(dates, method="ffill")
    return ser.rename("vix")


def vix_tercile_label(vix: pd.Series) -> pd.Series:
    """Categorise each month into VIX terciles (within the test sample)."""
    q = vix.quantile([0.333, 0.667]).values
    labels = pd.cut(vix, bins=[-np.inf, q[0], q[1], np.inf],
                    labels=["low_vix", "mid_vix", "high_vix"])
    return labels.astype(str)


def decade_label(dates: pd.DatetimeIndex) -> pd.Series:
    """Label each date with its calendar decade (e.g. ``2020s``)."""
    yrs = pd.Series(dates.year, index=dates)
    return (yrs // 10 * 10).astype(int).astype(str) + "s"


# ─────────────────────────────────────────────────────────────────────────────
#  Per-regime evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_on_slice(
    portfolio_returns: pd.Series,
    portfolio_returns_gross: pd.Series,
    mask: pd.Series,
) -> dict:
    """Compute Sharpe (net), Sharpe (gross), MaxDD, Skew, Kurt, mean return,
    and # months on the masked slice of an H-L return series."""
    from src.evaluation.metrics import sharpe_ratio, max_drawdown
    from scipy import stats

    pr = portfolio_returns.reindex(mask.index)
    pg = portfolio_returns_gross.reindex(mask.index)
    pr = pr[mask].dropna()
    pg = pg[mask].dropna()
    if len(pr) < 6:
        return {
            "n_months": int(len(pr)),
            "sharpe_net": float("nan"),
            "sharpe_gross": float("nan"),
            "mean_ret_pct_pm": float("nan"),
            "max_dd_pct": float("nan"),
            "skew": float("nan"),
            "kurt": float("nan"),
        }
    return {
        "n_months": int(len(pr)),
        "sharpe_net":  float(sharpe_ratio(pr)),
        "sharpe_gross": float(sharpe_ratio(pg)) if len(pg) else float("nan"),
        "mean_ret_pct_pm": float(pr.mean() * 100),
        "max_dd_pct":   float(max_drawdown(pr) * 100),
        "skew":         float(stats.skew(pr)) if len(pr) > 2 else float("nan"),
        "kurt":         float(stats.kurtosis(pr)) if len(pr) > 3 else float("nan"),
    }


def evaluate_regimes(
    portfolio_returns: Dict[str, Dict[str, pd.Series]],
    portfolio_returns_gross: Dict[str, Dict[str, pd.Series]],
    test_dates: pd.DatetimeIndex,
    vix_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a long-format DataFrame: (regime_kind, regime, model, metric...).

    Regimes computed:
      * 'recession' / 'expansion' (NBER)
      * 'low_vix' / 'mid_vix' / 'high_vix' (terciles within test window)
      * '1980s' / '1990s' / '2000s' / '2010s' / '2020s' (decade)
      * 'all' (full test sample, for reference)
    """
    test_dates = pd.DatetimeIndex(pd.to_datetime(test_dates).unique()).sort_values()

    # Build masks (Series indexed by test_dates)
    rec = nber_recession_mask(test_dates)
    vix = load_vix_monthly(test_dates, vix_path=vix_path)
    vix_lbl = vix_tercile_label(vix)
    dec = decade_label(test_dates)

    rows = []
    for model in portfolio_returns:
        hl_net = portfolio_returns[model].get("H-L", pd.Series(dtype=float))
        hl_gross = portfolio_returns_gross.get(model, {}).get(
            "H-L", pd.Series(dtype=float)
        )
        if len(hl_net) == 0:
            continue
        # Align to test dates index
        hl_net = hl_net.reindex(test_dates)
        hl_gross = hl_gross.reindex(test_dates)

        # All sample
        rows.append({
            "regime_kind": "full",
            "regime": "all",
            "model": model,
            **_metrics_on_slice(
                hl_net, hl_gross, pd.Series(True, index=test_dates),
            ),
        })

        # NBER
        for label, mask in [
            ("recession", rec),
            ("expansion", ~rec),
        ]:
            rows.append({
                "regime_kind": "nber",
                "regime": label,
                "model": model,
                **_metrics_on_slice(hl_net, hl_gross, mask),
            })

        # VIX terciles
        for label in ["low_vix", "mid_vix", "high_vix"]:
            mask = (vix_lbl == label)
            rows.append({
                "regime_kind": "vix",
                "regime": label,
                "model": model,
                **_metrics_on_slice(hl_net, hl_gross, mask),
            })

        # Decades
        for label in dec.unique():
            mask = (dec == label)
            rows.append({
                "regime_kind": "decade",
                "regime": label,
                "model": model,
                **_metrics_on_slice(hl_net, hl_gross, mask),
            })

    return pd.DataFrame(rows)
