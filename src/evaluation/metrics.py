"""
evaluation/metrics.py
---------------------
Statistical and economic evaluation tools from GKX (2019):

  • oos_r2()             – panel out-of-sample R² (eq. 19)
  • diebold_mariano()    – modified DM test for panel forecasts (Section 2.8)
  • sharpe_ratio()       – annualised Sharpe ratio
  • sr_improvement()     – Campbell-Thompson (2008) SR* formula
  • variable_importance() – R²-based variable importance
  • portfolio_performance() – full performance table
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Return-prediction metrics
# ─────────────────────────────────────────────────────────────────────────────

def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    GKX (2019) Equation (19): panel OOS R²  benchmarked against zero forecast.
    R²_oos = 1 − Σ(r − r̂)² / Σ r²
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def oos_r2_monthly(
    y_true: pd.Series,
    y_pred: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    """Monthly time series of OOS R²."""
    df = pd.DataFrame({"y": y_true.values, "yhat": y_pred.values, "date": dates.values})
    def _r2(g):
        ss_res = ((g["y"] - g["yhat"]) ** 2).sum()
        ss_tot = (g["y"] ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return df.groupby("date").apply(_r2)


# ─────────────────────────────────────────────────────────────────────────────
#  Diebold-Mariano test (panel version, GKX Section 2.8)
# ─────────────────────────────────────────────────────────────────────────────

def diebold_mariano(
    y_true: np.ndarray,
    pred_1: np.ndarray,
    pred_2: np.ndarray,
    dates: np.ndarray,
    max_lags: int = 12,
) -> Tuple[float, float]:
    """
    Modified Diebold-Mariano test for panel forecasts.

    Tests H0: equal predictive accuracy between model 1 and model 2.
    Positive DM statistic → model 2 outperforms model 1.

    Parameters
    ----------
    y_true   : realised returns  (N×T,)
    pred_1   : forecasts model 1 (N×T,)
    pred_2   : forecasts model 2 (N×T,)
    dates    : date labels       (N×T,)
    max_lags : Newey-West lags

    Returns
    -------
    dm_stat, p_value
    """
    # Cross-sectional average of squared error differences at each date
    df = pd.DataFrame({
        "date": dates,
        "e1": (y_true - pred_1) ** 2,
        "e2": (y_true - pred_2) ** 2,
    })
    d_t = df.groupby("date").apply(lambda g: (g["e1"] - g["e2"]).mean()).sort_index()

    d_bar = d_t.mean()
    T     = len(d_t)

    # Newey-West standard error
    nw_se = _newey_west_se(d_t.values, max_lags=max_lags)

    dm_stat = d_bar / nw_se if nw_se > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat))) if not np.isnan(dm_stat) else np.nan
    return float(dm_stat), float(p_value)


def _newey_west_se(x: np.ndarray, max_lags: int = 12) -> float:
    """Newey-West HAC standard error."""
    T   = len(x)
    xc  = x - x.mean()
    var = np.sum(xc ** 2) / T
    for lag in range(1, min(max_lags + 1, T)):
        w    = 1 - lag / (max_lags + 1)
        cov  = np.sum(xc[lag:] * xc[:-lag]) / T
        var += 2 * w * cov
    return np.sqrt(max(var, 0) / T)


def dm_table(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    dates: np.ndarray,
) -> pd.DataFrame:
    """
    Build the full DM test table (like GKX Table 3).
    Row model vs column model.
    """
    models = list(predictions.keys())
    n = len(models)
    mat = np.full((n, n), np.nan)
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i != j:
                dm, _ = diebold_mariano(y_true, predictions[m1], predictions[m2], dates)
                mat[i, j] = dm
    return pd.DataFrame(mat, index=models, columns=models)


# ─────────────────────────────────────────────────────────────────────────────
#  Sharpe ratio metrics
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(ret: np.ndarray | pd.Series, annualise: int = 12) -> float:
    """Annualised Sharpe ratio (assumes ret is monthly excess return)."""
    r = np.asarray(ret)
    r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(annualise))


def sr_star(sr: float, r2: float) -> float:
    """
    Campbell & Thompson (2008) implied Sharpe ratio:
    SR* = sqrt(SR² + R²_oos / (1 - R²_oos))
    """
    if np.isnan(r2) or r2 >= 1 or r2 < 0:
        return np.nan
    return float(np.sqrt(sr ** 2 + r2 / (1 - r2)))


def sr_improvement(sr: float, r2: float) -> float:
    """SR* - SR."""
    s = sr_star(sr, r2)
    return float(s - sr) if not np.isnan(s) else np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  Variable importance
# ─────────────────────────────────────────────────────────────────────────────

def variable_importance_r2(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
) -> pd.Series:
    """
    GKX (2019) variable importance: reduction in OOS R² when predictor j
    is set to zero (holding other estimates fixed).

    IMPORTANT: This method sets each feature to 0.0 as the perturbation.
    This is only meaningful if features are cross-sectionally rank-normalised
    to the interval [-1, 1] (as in GKX), so that 0.0 corresponds to the
    cross-sectional median. If features are not normalised, importances
    will be distorted. Ensure normalisation is applied before calling this.
    """
    present = [c for c in feature_names if c in X.columns]
    if present:
        feature_means = X[present].mean()
        if (feature_means.abs() > 0.5).any():
            warnings.warn(
                "variable_importance_r2: some features have |mean| > 0.5. "
                "Features may not be cross-sectionally normalised. "
                "Importances computed by zeroing features may be unreliable.",
                stacklevel=2,
            )

    baseline = oos_r2(y, model.predict(X))
    importances = {}
    for col in feature_names:
        if col not in X.columns:
            continue
        X_pert = X.copy()
        X_pert[col] = 0.0
        r2_j = oos_r2(y, model.predict(X_pert))
        importances[col] = baseline - r2_j   # positive → important
    return pd.Series(importances).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Portfolio performance table
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_performance_table(
    portfolios: Dict[str, pd.Series],
    rf: float = 0.0,
    annualise: int = 12,
) -> pd.DataFrame:
    """
    Build summary performance table for a dictionary of monthly return series.
    Columns: Mean Return, Std, Sharpe, Max DD, Skew, Kurtosis.
    """
    rows = []
    for name, ret in portfolios.items():
        r = ret.dropna()
        rows.append({
            "Strategy":     name,
            "Mean Ret (%)": r.mean() * 100,
            "Std (%)":      r.std() * 100,
            "Ann. Sharpe":  sharpe_ratio(r, annualise),
            "Max DD (%)":   max_drawdown(r) * 100,
            "Skew":         float(stats.skew(r)),
            "Kurt":         float(stats.kurtosis(r)),
        })
    return pd.DataFrame(rows).set_index("Strategy")


def max_drawdown(ret: pd.Series | np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of cumulative log returns."""
    cum = np.log1p(np.asarray(ret)).cumsum()
    max_dd = 0.0
    peak   = cum[0]
    for v in cum:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def alpha_tstat(returns: pd.Series, factors: pd.DataFrame) -> Tuple[float, float]:
    """
    OLS alpha and t-stat of returns regressed on factor portfolio returns.
    Returns: (alpha, t_stat)
    """
    common = returns.index.intersection(factors.index)
    y = returns.loc[common].values
    X = np.column_stack([np.ones(len(common)), factors.loc[common].values])
    try:
        b, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
        e  = y - X @ b
        s2 = np.sum(e**2) / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        return float(b[0] * 12 * 100), float(b[0] / se[0])  # annualised alpha %
    except Exception:
        return np.nan, np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  Comprehensive evaluation wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Aggregates all evaluation metrics for a set of models over the test period.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        dates: np.ndarray,
        portfolio_returns: Optional[Dict[str, Dict[str, pd.Series]]] = None,
    ):
        """
        Parameters
        ----------
        y_true           : realised individual stock returns (panel)
        predictions      : {model_name: predicted returns} for the test set
        dates            : date array aligned with y_true
        portfolio_returns : {model_name: {decile: return_series}}
        """
        self.y_true    = y_true
        self.preds     = predictions
        self.dates     = dates
        self.port_rets = portfolio_returns or {}

    def oos_r2_table(self) -> pd.Series:
        """Panel OOS R² for each model (%)."""
        return pd.Series(
            {name: oos_r2(self.y_true, p) * 100
             for name, p in self.preds.items()},
            name="OOS R² (%)"
        )

    def dm_table(self) -> pd.DataFrame:
        return dm_table(self.y_true, self.preds, self.dates)

    def sharpe_table(self) -> pd.DataFrame:
        """Sharpe ratios for long-short decile spread portfolios."""
        rows = []
        for model, deciles in self.port_rets.items():
            if "H-L" in deciles:
                sr = sharpe_ratio(deciles["H-L"])
                rows.append({"Model": model, "H-L Sharpe": sr})
        return pd.DataFrame(rows).set_index("Model") if rows else pd.DataFrame()

    def summary_table(self) -> pd.DataFrame:
        r2  = self.oos_r2_table()
        sr  = self.sharpe_table()
        df  = r2.to_frame()
        if not sr.empty:
            df = df.join(sr, how="left")
        return df
