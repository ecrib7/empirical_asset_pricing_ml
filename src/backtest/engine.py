"""
backtest/engine.py
------------------
Portfolio construction and backtesting engine.

Implements:
  • DecilePortfolioBuilder  – long-short decile spread portfolios
  • MarketTimer             – Campbell-Thompson (2008) market timing
  • TransactionCostModel    – round-trip cost model
  • BacktestEngine          – full pipeline: predictions → performance

Following GKX (2019) Section 3.4 exactly.

Per-year checkpointing
----------------------
``BacktestEngine.run`` writes a checkpoint after each test year completes.
On Google Colab with Drive mounted the default directory is
``/content/drive/MyDrive/Algo Trading Project/backtest_checkpoint``;
otherwise it falls back to ``data/cache/backtest_checkpoint``.
If a run is interrupted, calling ``run`` again with the same ``models``
dict resumes from the next un-finished year. To force a clean run delete
the checkpoint file.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import FREQ_MONTH_END, FREQ_YEAR_START

logger = logging.getLogger(__name__)

_DRIVE_CKPT = Path("/content/drive/MyDrive/Algo Trading Project/backtest_checkpoint")


def _default_checkpoint_dir() -> str:
    """On Colab with Drive mounted, default to Drive; otherwise local."""
    if _DRIVE_CKPT.parent.exists():
        return str(_DRIVE_CKPT)
    return "data/cache/backtest_checkpoint"


def add_forward_return_target(
    df: pd.DataFrame,
    permno_col: str = "permno",
    date_col: str = "date",
    ret_col: str = "ret",
    out_col: str = "ret_fwd",
) -> pd.DataFrame:
    """
    ``out_col`` at (permno, date=t) equals ``ret_col`` at (permno, date=t+1),
    i.e. next calendar month's return on the same stock.  Last month per permno
    is NaN.  Rows are aligned by (permno, date) merge so original order is kept.
    """
    out = df.copy()
    if out_col in out.columns:
        out = out.drop(columns=[out_col])
    tmp = out[[permno_col, date_col, ret_col]].sort_values([permno_col, date_col])
    tmp[out_col] = tmp.groupby(permno_col)[ret_col].shift(-1)
    out = out.merge(
        tmp[[permno_col, date_col, out_col]],
        on=[permno_col, date_col],
        how="left",
    )
    return out


def feature_columns_for_training(
    fm: pd.DataFrame,
    target_col: str,
    id_col: str = "permno",
    date_col: str = "date",
    me_col: str = "me",
) -> List[str]:
    """Columns used as X; raw contemporaneous ``ret`` is excluded when target is ``ret_fwd``.
    ``adv_dollar`` is engine-side TC metadata, not a predictor — exclude it too."""
    exclude = {id_col, date_col, target_col, me_col, "adv_dollar"}
    if target_col == "ret_fwd":
        exclude.add("ret")
    return [c for c in fm.columns if c not in exclude]


def one_way_portfolio_turnover(w_new: pd.Series, w_old: pd.Series) -> float:
    """Σ|Δw|/2 for two weight vectors on the same security index (union of indices)."""
    idx = w_new.index.union(w_old.index)
    a = w_new.reindex(idx).fillna(0.0)
    b = w_old.reindex(idx).fillna(0.0)
    return float((a - b).abs().sum() / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Transaction Cost Model
# ─────────────────────────────────────────────────────────────────────────────

class TransactionCostModel:
    """Simple proportional one-way transaction cost model (cost in bps)."""

    def __init__(self, cost_bps: float = 10.0):
        self.cost = cost_bps / 10_000.0

    def period_one_way_turnover(self, w_new: pd.Series, w_old: pd.Series) -> float:
        return one_way_portfolio_turnover(w_new, w_old)

    def period_turnover_cost(self, w_new: pd.Series, w_old: pd.Series, **kwargs) -> float:
        return float(self.cost * self.period_one_way_turnover(w_new, w_old))

    def net_return(
        self,
        gross_ret: pd.Series,
        weights: pd.DataFrame,
        weights_prev: pd.DataFrame,
    ) -> pd.Series:
        idx = gross_ret.index
        w   = weights.reindex(idx).fillna(0.0)
        w_l = weights_prev.reindex(idx).fillna(0.0)
        turnover = (w - w_l).abs().sum(axis=1) / 2.0
        return gross_ret - self.cost * turnover

    def cost_series(
        self,
        weights: pd.DataFrame,
        weights_prev: pd.DataFrame,
    ) -> pd.Series:
        w   = weights.fillna(0.0)
        w_l = weights_prev.fillna(0.0)
        return self.cost * (w - w_l).abs().sum(axis=1) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
#  Impact-aware transaction cost model
#  ----------------------------------------------------------------------------
#  Inspired by Frazzini, Israel & Moskowitz (2018) "Trading Costs", which
#  decomposes execution cost into a *half-spread* component (cap-size-
#  dependent) and a *price-impact* component (square-root of trade size /
#  ADV). For each stock the per-trade cost rate (in bps of notional) is::
#
#     cost_bps_i = half_spread_bps(log_mcap_i)
#                + impact_coef_bps * sqrt( (|Δw_i| * NAV) / ADV_i )
#
#  where ADV_i is monthly dollar volume divided by 21 trading days and
#  NAV is normalised to $1 per leg. The portfolio-level cost at time t is::
#
#     tcost_t = Σ_i cost_bps_i * |Δw_i,t| / 10_000
#
#  Defaults: half-spread is 5 bps for the 95th-percentile log market cap
#  and 25 bps for the 5th-percentile, log-linearly interpolated in between
#  using each month's cross-sectional distribution. Impact coefficient is
#  10 bps per √($-traded / ADV). These match the post-decimalisation
#  calibration in FIM (2018) Tables 2-3.
# ─────────────────────────────────────────────────────────────────────────────

class ImpactAwareTransactionCostModel:
    """
    Per-stock proportional transaction cost with size-dependent half-spread
    and FIM-style square-root market impact. Falls back to the simple flat
    bps cost when per-stock metadata (market cap or dollar volume) is missing.

    Parameters
    ----------
    half_spread_small_bps : float
        Half-spread in bps applied to the smallest market-cap decile.
    half_spread_large_bps : float
        Half-spread in bps applied to the largest market-cap decile. The
        full-sample half-spread is interpolated log-linearly between
        these two anchors as a function of log(market equity).
    impact_coef_bps : float
        Coefficient λ in cost_bps = half_spread + λ * sqrt(trade$/ADV).
    fallback_bps : float
        Flat one-way cost (bps) used for stocks with missing metadata.
    """

    def __init__(
        self,
        half_spread_small_bps: float = 25.0,
        half_spread_large_bps: float = 5.0,
        impact_coef_bps: float = 10.0,
        fallback_bps: float = 10.0,
    ):
        self.hs_small = half_spread_small_bps
        self.hs_large = half_spread_large_bps
        self.impact_coef = impact_coef_bps
        self.fallback = fallback_bps / 10_000.0
        self._mcap_log_min: Optional[float] = None
        self._mcap_log_max: Optional[float] = None
        self._mcap_t: Optional[pd.Series] = None
        self._adv_t: Optional[pd.Series] = None

    def set_metadata(
        self,
        mcap_t: Optional[pd.Series],
        adv_t: Optional[pd.Series],
    ) -> None:
        """Update per-stock market cap and average daily $ volume for the
        current test month. ADV is in raw dollars per trading day (we use
        monthly $-volume / 21)."""
        self._mcap_t = mcap_t
        self._adv_t = adv_t
        if mcap_t is not None and len(mcap_t.dropna()) > 0:
            valid = mcap_t.dropna()
            valid = valid[valid > 0]
            if len(valid) > 1:
                lm = np.log(valid)
                self._mcap_log_min = float(lm.quantile(0.05))
                self._mcap_log_max = float(lm.quantile(0.95))

    def _half_spread_bps(self, stocks: pd.Index) -> pd.Series:
        """Log-linear interpolation between small-cap and large-cap anchors."""
        if (self._mcap_t is None
                or self._mcap_log_min is None
                or self._mcap_log_max is None):
            return pd.Series(self.fallback * 10_000, index=stocks)
        mc = self._mcap_t.reindex(stocks)
        log_mc = np.log(mc.where(mc > 0))
        denom = max(self._mcap_log_max - self._mcap_log_min, 1e-9)
        pos = ((log_mc - self._mcap_log_min) / denom).clip(0.0, 1.0)
        # Larger pos → larger mcap → smaller half-spread
        hs = self.hs_small + (self.hs_large - self.hs_small) * pos
        return hs.fillna(self.fallback * 10_000)

    def _impact_bps(self, stocks: pd.Index, dollar_traded: pd.Series) -> pd.Series:
        """λ * sqrt(dollar_traded / ADV). Returns 0 when ADV is missing or zero."""
        if self._adv_t is None:
            return pd.Series(0.0, index=stocks)
        adv = self._adv_t.reindex(stocks)
        ratio = dollar_traded / adv.where(adv > 0)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        return self.impact_coef * np.sqrt(ratio)

    def period_one_way_turnover(self, w_new: pd.Series, w_old: pd.Series) -> float:
        return one_way_portfolio_turnover(w_new, w_old)

    def period_turnover_cost(
        self,
        w_new: pd.Series,
        w_old: pd.Series,
        nav: float = 1.0,
    ) -> float:
        """
        Per-stock cost: |Δw_i| * (half_spread_i + λ * sqrt(|Δw_i|*NAV/ADV_i)) / 10000.

        Divided by 2 to match the one-way turnover convention used by the
        flat ``TransactionCostModel`` (i.e. half_spread_bps is quoted as a
        one-way spread; round-trip rebalancing pays it twice).
        """
        idx = w_new.index.union(w_old.index)
        a = w_new.reindex(idx).fillna(0.0)
        b = w_old.reindex(idx).fillna(0.0)
        dw = (a - b).abs()
        traded = dw[dw > 0]
        if len(traded) == 0:
            return 0.0
        hs = self._half_spread_bps(traded.index)
        dollar_traded = traded * float(nav)
        impact = self._impact_bps(traded.index, dollar_traded)
        cost_bps = (hs + impact).reindex(traded.index).fillna(self.fallback * 10_000)
        # /2 → align with one-way turnover convention (cf. TransactionCostModel)
        return float((traded * cost_bps).sum() / 2.0 / 10_000)

    @property
    def cost(self) -> float:
        """Compatibility shim — `tc_model.cost > 0` truthiness check should
        still succeed for impact-aware models (so `hl_returns_are_net_of_tc`
        is reported correctly)."""
        return max(self.fallback, (self.hs_small + self.hs_large) / 2 / 10_000)



# ─────────────────────────────────────────────────────────────────────────────
#  Stock-level impact cost model
#  ----------------------------------------------------------------------------
#  Extends the FIM-style ``ImpactAwareTransactionCostModel`` with per-stock
#  return volatility. Empirically, both quoted spreads and impact coefficients
#  scale with idiosyncratic volatility:
#
#    * Stoll (2000) — half-spread is a near-linear function of σ_ret cross-
#      sectionally (high-vol stocks have wider quotes because dealers face
#      more inventory risk).
#    * Hasbrouck (2009) — fitted impact coefficients are ~3× larger for
#      top-quartile-vol stocks vs bottom-quartile, controlling for size.
#
#  The cost-decomposition in this model:
#
#     half_spread_bps_i = hs_size_i + γ_spread · max(0, σ_i − σ̄)        [bps]
#     impact_bps_i      = λ_0 · (1 + γ_impact · σ̃_i) · √(trade$_i / ADV_i)
#
#  where σ̃_i = σ_i / σ̄ − 1 is the cross-sectional vol z-score (median = 0)
#  and σ̄ is the median of σ_i in the current month. γ_spread, γ_impact
#  default to values that give the top-quartile-vol stock ~50% higher half-
#  spread and ~40% higher impact than the median stock at equal size/ADV.
# ─────────────────────────────────────────────────────────────────────────────

class StockLevelImpactCostModel(ImpactAwareTransactionCostModel):
    """
    Stock-conditional FIM-style impact model. In addition to size and ADV,
    incorporates per-stock return volatility (``retvol``) into both the
    half-spread and the market-impact components.

    Parameters
    ----------
    half_spread_small_bps, half_spread_large_bps, impact_coef_bps, fallback_bps
        Forwarded to ``ImpactAwareTransactionCostModel``.
    vol_spread_bps : float
        Additional bps of half-spread per (σ_i − σ_median) cross-sectionally.
        Default 8 bps → top-quartile-vol stock pays ~+8 bps extra spread.
    vol_impact_scale : float
        Multiplier on the impact coefficient: ``λ → λ · (1 + scale · σ̃)``
        where σ̃ is the standardised cross-sectional vol z-score.
        Default 0.4 → top-quartile-vol stock has ~+40% impact at given trade$.
    nav_billions : float
        Strategy NAV in billions, used to convert weights to trade$. Default
        1.0 ($1B AUM). Larger NAV → larger trade$ → larger √-impact.
    """

    def __init__(
        self,
        half_spread_small_bps: float = 25.0,
        half_spread_large_bps: float = 5.0,
        impact_coef_bps: float = 10.0,
        fallback_bps: float = 10.0,
        vol_spread_bps: float = 8.0,
        vol_impact_scale: float = 0.4,
        nav_billions: float = 1.0,
    ):
        super().__init__(
            half_spread_small_bps=half_spread_small_bps,
            half_spread_large_bps=half_spread_large_bps,
            impact_coef_bps=impact_coef_bps,
            fallback_bps=fallback_bps,
        )
        self.vol_spread_bps    = float(vol_spread_bps)
        self.vol_impact_scale  = float(vol_impact_scale)
        self.nav               = float(nav_billions) * 1e9
        self._vol_t: Optional[pd.Series] = None
        self._vol_median_t: Optional[float] = None

    def set_metadata(
        self,
        mcap_t: Optional[pd.Series],
        adv_t:  Optional[pd.Series],
        retvol_t: Optional[pd.Series] = None,
    ) -> None:
        """Update per-stock context for the current test month."""
        super().set_metadata(mcap_t, adv_t)
        self._vol_t = retvol_t
        if retvol_t is not None and len(retvol_t.dropna()) > 0:
            v = retvol_t.dropna()
            v = v[v > 0]
            self._vol_median_t = float(v.median()) if len(v) > 0 else None
        else:
            self._vol_median_t = None

    # ── Overrides ────────────────────────────────────────────────────────

    def _half_spread_bps(self, stocks: pd.Index) -> pd.Series:
        """Size component (parent) + vol-premium component (this class)."""
        hs = super()._half_spread_bps(stocks)
        if self._vol_t is None or self._vol_median_t is None:
            return hs
        v = self._vol_t.reindex(stocks)
        # Add bps of vol-premium for stocks above the median vol; clip at 0.
        excess = (v - self._vol_median_t).clip(lower=0.0).fillna(0.0)
        return hs + self.vol_spread_bps * excess / max(self._vol_median_t, 1e-9)

    def _impact_bps(self, stocks: pd.Index, dollar_traded: pd.Series) -> pd.Series:
        """Vol-scaled FIM impact: λ · (1 + γ · σ̃) · √(trade$ / ADV)."""
        base_imp = super()._impact_bps(stocks, dollar_traded)
        if self._vol_t is None or self._vol_median_t is None:
            return base_imp
        v = self._vol_t.reindex(stocks)
        # σ̃ = σ_i / σ_median − 1, centred so median stock has multiplier 1.
        sigma_tilde = (v / max(self._vol_median_t, 1e-9) - 1.0).fillna(0.0)
        multiplier = (1.0 + self.vol_impact_scale * sigma_tilde).clip(lower=0.5)
        return base_imp * multiplier

    def period_turnover_cost(
        self,
        w_new: pd.Series,
        w_old: pd.Series,
        nav: Optional[float] = None,
    ) -> float:
        """Use constructor NAV unless caller overrides explicitly."""
        nav_used = float(nav) if nav is not None else self.nav
        return super().period_turnover_cost(w_new, w_old, nav=nav_used)


# ─────────────────────────────────────────────────────────────────────────────
#  Decile portfolio builder
# ─────────────────────────────────────────────────────────────────────────────

class DecilePortfolioBuilder:
    """Sorts stocks into deciles each month based on model predictions."""

    def __init__(
        self,
        n_deciles: int = 10,
        weighting: str = "value",
        tc_model: Optional[TransactionCostModel] = None,
    ):
        self.n_deciles = n_deciles
        self.weighting = weighting
        self.tc_model  = tc_model

    def build(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame,
        market_caps: Optional[pd.DataFrame] = None,
        adv: Optional[pd.DataFrame] = None,
        retvol: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series]]:
        dates = predictions.index.intersection(returns.index)
        port_net = {str(d): [] for d in range(1, self.n_deciles + 1)}
        port_net["H-L"] = []
        port_gross = {str(d): [] for d in range(1, self.n_deciles + 1)}
        port_gross["H-L"] = []
        port_turn = {str(d): [] for d in range(1, self.n_deciles + 1)}
        port_turn["H-L"] = []
        date_idx: List[pd.Timestamp] = []

        universe = sorted(predictions.columns.union(returns.columns))
        z = pd.Series(0.0, index=universe, dtype=float)
        prev_w = {str(d): z.copy() for d in range(1, self.n_deciles + 1)}
        prev_w["H-L"] = z.copy()

        impact_aware = isinstance(self.tc_model, ImpactAwareTransactionCostModel)

        for t in dates:
            pred_t = predictions.loc[t].dropna()
            ret_t = returns.loc[t].reindex(pred_t.index).dropna()
            common = pred_t.index.intersection(ret_t.index)
            if len(common) < self.n_deciles:
                continue

            pred_t = pred_t.loc[common]
            ret_t = ret_t.loc[common]

            if impact_aware:
                mc_t = (
                    market_caps.loc[t]
                    if market_caps is not None and t in market_caps.index
                    else None
                )
                adv_t = adv.loc[t] if adv is not None and t in adv.index else None
                # Stock-level models additionally consume per-stock retvol.
                # The parent ImpactAwareTransactionCostModel.set_metadata()
                # ignores this kwarg; the child class uses it. Falls back
                # cleanly when retvol is unavailable.
                rv_t = retvol.loc[t] if retvol is not None and t in retvol.index else None
                try:
                    self.tc_model.set_metadata(mc_t, adv_t, retvol_t=rv_t)
                except TypeError:
                    self.tc_model.set_metadata(mc_t, adv_t)

            try:
                labels = pd.qcut(pred_t, q=self.n_deciles,
                                 labels=range(1, self.n_deciles + 1))
            except ValueError:
                labels = pd.qcut(pred_t.rank(method="first"), q=self.n_deciles,
                                 labels=range(1, self.n_deciles + 1))

            for d in range(1, self.n_deciles + 1):
                mask = labels == d
                if mask.sum() == 0:
                    port_net[str(d)].append(np.nan)
                    port_gross[str(d)].append(np.nan)
                    port_turn[str(d)].append(np.nan)
                    continue
                stocks = common[mask]
                w = self._weights(stocks, t, market_caps)
                gross = float((w * ret_t.loc[stocks]).sum())
                w_full = z.copy()
                w_full.loc[w.index] = w.values.astype(float)
                turn = one_way_portfolio_turnover(w_full, prev_w[str(d)])
                tcost = (
                    0.0
                    if self.tc_model is None
                    else self.tc_model.period_turnover_cost(w_full, prev_w[str(d)])
                )
                net = gross - tcost
                port_gross[str(d)].append(gross)
                port_turn[str(d)].append(turn)
                port_net[str(d)].append(net)
                prev_w[str(d)] = w_full

            top_mask = labels == self.n_deciles
            bot_mask = labels == 1
            top_stocks = common[top_mask]
            bot_stocks = common[bot_mask]
            w_top = self._weights(top_stocks, t, market_caps)
            w_bot = self._weights(bot_stocks, t, market_caps)
            gross_hl = float(
                (w_top * ret_t.loc[top_stocks]).sum()
                - (w_bot * ret_t.loc[bot_stocks]).sum()
            )
            w_hl = (
                z.add(w_top.reindex(universe).fillna(0.0), fill_value=0.0)
                .sub(w_bot.reindex(universe).fillna(0.0), fill_value=0.0)
            )
            turn_hl = one_way_portfolio_turnover(w_hl, prev_w["H-L"])
            tcost_hl = (
                0.0
                if self.tc_model is None
                else self.tc_model.period_turnover_cost(w_hl, prev_w["H-L"])
            )
            net_hl = gross_hl - tcost_hl
            port_gross["H-L"].append(gross_hl)
            port_turn["H-L"].append(turn_hl)
            port_net["H-L"].append(net_hl)
            prev_w["H-L"] = w_hl

            date_idx.append(t)

        def _pack(src: dict) -> Dict[str, pd.Series]:
            return {k: pd.Series(v, index=date_idx, name=k) for k, v in src.items()}

        return _pack(port_net), _pack(port_gross), _pack(port_turn)

    def _weights(
        self,
        stocks: pd.Index,
        t: pd.Timestamp,
        market_caps: Optional[pd.DataFrame],
    ) -> pd.Series:
        if self.weighting == "value" and market_caps is not None:
            mc = market_caps.loc[t].reindex(stocks).fillna(0)
            total = mc.sum()
            if total > 0:
                return mc / total
        return pd.Series(1.0 / len(stocks), index=stocks)

    def performance_table(
        self,
        port_returns: Dict[str, pd.Series],
        predictions_avg: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        from src.evaluation.metrics import sharpe_ratio
        rows = []
        for d in list(range(1, self.n_deciles + 1)) + ["H-L"]:
            key = str(d)
            if key not in port_returns:
                continue
            r = port_returns[key].dropna()
            rows.append({
                "Decile":     "Low(L)" if d == 1 else "High(H)" if d == self.n_deciles else
                              "H-L" if key == "H-L" else str(d),
                "Pred":       predictions_avg.get(key, np.nan) if predictions_avg else np.nan,
                "Avg Ret (%)": r.mean() * 100,
                "Std (%)":    r.std() * 100,
                "Ann. Sharpe": sharpe_ratio(r),
            })
        return pd.DataFrame(rows).set_index("Decile")


# ─────────────────────────────────────────────────────────────────────────────
#  Market timer (Campbell-Thompson 2008)
# ─────────────────────────────────────────────────────────────────────────────

class MarketTimer:
    def __init__(self, max_leverage: float = 1.5):
        self.max_leverage = max_leverage

    def returns(
        self,
        predicted: pd.Series,
        realised: pd.Series,
        rf: float = 0.0,
    ) -> Tuple[pd.Series, pd.Series]:
        common = predicted.index.intersection(realised.index)
        pred   = predicted.loc[common]
        real   = realised.loc[common]

        sig = float(pred.std())
        w = pred / max(sig, 1e-8)
        w = w.clip(lower=0.0, upper=self.max_leverage)

        timed = w * real + (1 - w) * rf
        return timed, real

    def sharpe_improvement(
        self,
        predicted: pd.Series,
        realised: pd.Series,
        annualise: int = 12,
    ) -> float:
        from src.evaluation.metrics import sharpe_ratio
        timed, bah = self.returns(predicted, realised)
        return sharpe_ratio(timed, annualise) - sharpe_ratio(bah, annualise)


# ─────────────────────────────────────────────────────────────────────────────
#  Full backtest engine — with per-year checkpointing
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Recursive out-of-sample backtest with per-year checkpointing.

    After each test year, the running prediction arrays are pickled to
    ``checkpoint_dir/<model_set_hash>.pkl``. If the script is killed and
    re-run with the same models, completed years are loaded from the
    checkpoint and only the remaining years are trained.
    """

    def __init__(
        self,
        train_start: str = "1957-03-01",
        val_start:   str = "1975-01-01",
        val_end:     str = "1986-12-31",
        test_start:  str = "1987-01-01",
        test_end:    str = "2016-12-31",
        n_deciles:   int = 10,
        weighting:   str = "value",
        tc_bps:      float = 10.0,
        tc_model:    str = "flat",
        impact_kwargs: Optional[dict] = None,
        checkpoint_dir: str | None = None,
        refit_step_years: int = 1,
    ):
        """
        Parameters
        ----------
        tc_model : {"flat", "impact"}
            "flat"   = legacy proportional bps cost (uses ``tc_bps``).
            "impact" = Frazzini-Israel-Moskowitz-style: half-spread varies
                       by market cap, plus √(trade$/ADV) impact term.
                       Reads ``me`` and ``adv`` (raw monthly $-volume / 21)
                       from the wide DataFrames passed to the builder.
        impact_kwargs : dict, optional
            Forwarded to ImpactAwareTransactionCostModel — keys include
            ``half_spread_small_bps``, ``half_spread_large_bps``,
            ``impact_coef_bps``, ``fallback_bps``.
        refit_step_years : int, default 1
            How many years between successive model refits. 1 = paper-
            faithful annual refit. Setting to 2 ~halves NN training time
            with a small (~5%) relative R² hit; useful for development.
            The checkpoint key includes this value so runs with different
            cadences don't collide.
        """
        self.train_start = pd.Timestamp(train_start)
        self.val_start   = pd.Timestamp(val_start)
        self.val_end     = pd.Timestamp(val_end)
        self.test_start  = pd.Timestamp(test_start)
        self.test_end    = pd.Timestamp(test_end)
        self.n_deciles   = n_deciles
        self.weighting   = weighting
        self._tc_bps     = float(tc_bps)
        self._tc_model_kind = tc_model
        if tc_model == "impact":
            self.tc_model = ImpactAwareTransactionCostModel(
                **(impact_kwargs or {})
            )
        else:
            self.tc_model = TransactionCostModel(tc_bps)
        self.refit_step_years = max(1, int(refit_step_years))
        self.checkpoint_dir = Path(checkpoint_dir or _default_checkpoint_dir())
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, models: dict) -> Path:
        """Path keyed by sorted model names + refit cadence so different
        cadences (annual vs every-2-years) don't load each other's state."""
        key = ",".join(sorted(models.keys())) + f"|refit{self.refit_step_years}"
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        # Include readable model names in filename for human inspection
        safe_key = key.replace("/", "_").replace("+", "p").replace("|", "_")[:80]
        return self.checkpoint_dir / f"ckpt_{safe_key}_{h}.pkl"

    def _load_checkpoint(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                ck = pickle.load(f)
            logger.info(f"[checkpoint] loaded {path.name} — "
                        f"completed years: {ck.get('completed_years', [])}")
            return ck
        except Exception as e:
            logger.warning(f"[checkpoint] could not load {path}: {e}; starting fresh")
            return None

    def _save_checkpoint(self, path: Path, state: dict) -> None:
        tmp = path.with_suffix(".pkl.tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)
            logger.info(f"[checkpoint] written to {path}")
        except Exception as e:
            logger.warning(f"[checkpoint] save failed for {path}: {e}")
            if tmp.exists():
                tmp.unlink()

    def run(
        self,
        feature_matrix: pd.DataFrame,
        models: dict,
        target_col: str = "ret_fwd",
        id_col: str = "permno",
        date_col: str = "date",
        me_col: str = "me",
    ) -> Dict:
        # Don't .copy() — feature_matrix is multi-GB. We need it sorted by
        # (date, permno) for the year-loop slicing, but we can do that
        # in-place on the caller's frame (the sorted result is what we
        # consume; pandas will create the sorted copy if needed but we
        # avoid an explicit second .copy()).
        fm = feature_matrix.sort_values([date_col, id_col])
        if target_col not in fm.columns:
            raise KeyError(
                f"Missing target column {target_col!r}. "
                "Call add_forward_return_target() before BacktestEngine.run()."
            )
        feat_cols = feature_columns_for_training(
            fm, target_col, id_col=id_col, date_col=date_col, me_col=me_col
        )

        # Pre-extract the per-(date, permno) metadata used after the year
        # loop for portfolio construction. This lets us free `fm` before
        # PCA / GLM / etc. allocate their working memory.
        adv_col = "adv_dollar"
        me_lookup = (
            fm[[date_col, id_col, me_col]].copy()
            if me_col in fm.columns else None
        )
        adv_lookup = (
            fm[[date_col, id_col, adv_col]].copy()
            if adv_col in fm.columns else None
        )
        # retvol used by StockLevelImpactCostModel for vol-conditional spread/impact
        retvol_lookup = (
            fm[[date_col, id_col, "retvol"]].copy()
            if "retvol" in fm.columns else None
        )

        first_train_end = self.test_start - pd.DateOffset(years=1)
        assert first_train_end.year == self.val_end.year, (
            f"val_end year {self.val_end.year} does not match "
            f"first walk-forward train_end year {first_train_end.year}"
        )

        all_test_dates = pd.date_range(
            self.test_start, self.test_end,
            freq=f"{self.refit_step_years}{FREQ_YEAR_START}",
        )

        # ── Checkpoint setup ─────────────────────────────────────────────
        ckpt_path = self._checkpoint_path(models)
        ckpt = self._load_checkpoint(ckpt_path)
        if ckpt is not None:
            predictions = {n: list(v) for n, v in ckpt["predictions"].items()}
            true_rets = list(ckpt["true_rets"])
            test_dates_all = list(ckpt["test_dates_all"])
            test_permnos = list(ckpt["test_permnos"])
            completed_years = set(ckpt["completed_years"])
            # Backfill any new model names that weren't in the checkpoint
            for n in models:
                predictions.setdefault(n, [np.nan] * len(true_rets))
        else:
            predictions = {name: [] for name in models}
            true_rets = []
            test_dates_all = []
            test_permnos = []
            completed_years: set = set()

        # ── Year loop ────────────────────────────────────────────────────
        for yr_start in all_test_dates:
            year = int(yr_start.year)
            if year in completed_years:
                logger.info(f"[checkpoint] year {year} already done — skipping")
                continue

            yr_end = yr_start + pd.DateOffset(years=self.refit_step_years) - pd.DateOffset(days=1)
            yr_end = min(yr_end, self.test_end)
            train_end_yr = yr_start - pd.DateOffset(years=1)

            mask_train = (fm[date_col] >= self.train_start) & (fm[date_col] <= train_end_yr)
            mask_val   = (fm[date_col] > train_end_yr) & (fm[date_col] <= yr_start - pd.DateOffset(days=1))
            mask_test  = (fm[date_col] >= yr_start) & (fm[date_col] <= yr_end)

            train = fm[mask_train].dropna(subset=[target_col])
            val   = fm[mask_val].dropna(subset=[target_col])
            test  = fm[mask_test].dropna(subset=[target_col])

            if len(train) < 100 or len(test) == 0:
                completed_years.add(year)
                continue

            # Build X views *without* copying when possible. The .fillna(0)
            # below was the killer at year 2001+: it forces a full copy of
            # the (multi-GB) feature block. We do it as a single in-place
            # operation on the float32 array instead, which halves peak RAM.
            def _materialise(df_slice):
                if df_slice is None or len(df_slice) == 0:
                    return None
                arr = df_slice[feat_cols].to_numpy(dtype=np.float32, copy=False)
                # In-place NaN -> 0 (no extra allocation)
                np.nan_to_num(arr, copy=False, nan=0.0)
                return arr

            X_tr = _materialise(train)
            X_v  = _materialise(val) if len(val) > 0 else None
            X_te = _materialise(test)
            y_train = train[target_col].to_numpy(dtype=np.float32, copy=False)
            y_val   = val[target_col].to_numpy(dtype=np.float32, copy=False) if len(val) > 0 else None
            y_test  = test[target_col].to_numpy(dtype=np.float32, copy=False)
            # Capture the test dates/permnos BEFORE freeing the DataFrame views
            test_dates_for_year   = test[date_col].to_numpy()
            test_permnos_for_year = test[id_col].to_numpy()
            n_test = len(test)
            # Free the DataFrame views — we only need the np arrays from here
            del train, val, test
            import gc; gc.collect()

            logger.info(f"Test year {year}: "
                        f"train={len(X_tr):,}  val={(len(X_v) if X_v is not None else 0):,}  test={n_test:,}")

            # Track length before this year so we can roll back on partial failure
            n_before = len(true_rets)
            year_preds: Dict[str, list] = {}
            for name, model in models.items():
                try:
                    if hasattr(model, "fit"):
                        model.fit(X_tr, y_train, X_v, y_val)
                    pred = model.predict(X_te)
                    year_preds[name] = np.asarray(pred, dtype=np.float32).tolist()
                except Exception as e:
                    logger.error(f"{name} failed at {year}: {e}")
                    year_preds[name] = [np.nan] * n_test

            # All models attempted — commit this year's predictions
            for name in models:
                predictions[name].extend(year_preds[name])
            true_rets.extend(y_test.astype(np.float32).tolist())
            test_dates_all.extend(test_dates_for_year.tolist())
            test_permnos.extend(test_permnos_for_year.tolist())
            completed_years.add(year)

            # ── Persist checkpoint ──────────────────────────────────────
            self._save_checkpoint(ckpt_path, {
                "predictions": predictions,
                "true_rets": true_rets,
                "test_dates_all": test_dates_all,
                "test_permnos": test_permnos,
                "completed_years": sorted(completed_years),
                "models": sorted(models.keys()),
                "tc_bps": self._tc_bps,
            })
            logger.info(f"[checkpoint] saved through year {year} "
                        f"({len(completed_years)} years done) -> {ckpt_path.name}")

            # Free the year's float32 arrays before the next iteration
            # allocates fresh ones — the largest are 2-3 GB each on the
            # improved variant by the late 2010s.
            del X_tr, X_v, X_te, y_train, y_val, y_test
            del test_dates_for_year, test_permnos_for_year
            gc.collect()

        # ── Free the big feature matrix before portfolio construction ───
        # The portfolio builder only needs me + adv per (date, permno),
        # which we captured into me_lookup / adv_lookup at the top of run().
        del fm
        gc.collect()

        # ── Assemble panel ────────────────────────────────────────────────
        test_idx   = pd.to_datetime(test_dates_all)
        true_arr   = np.array(true_rets, dtype=np.float32)
        pred_arrays = {n: np.array(v, dtype=np.float32) for n, v in predictions.items()}

        # ── Build wide prediction DataFrames for portfolio construction ──
        portfolio_returns: Dict[str, Dict[str, pd.Series]] = {}
        portfolio_returns_gross: Dict[str, Dict[str, pd.Series]] = {}
        portfolio_turnover: Dict[str, Dict[str, pd.Series]] = {}

        # Per-(date, permno) metadata from the lightweight lookups
        keys_df = pd.DataFrame({date_col: test_idx, id_col: test_permnos})
        if me_lookup is not None:
            me_vals = (keys_df.merge(me_lookup, on=[date_col, id_col], how="left")
                                 [me_col].values)
        else:
            me_vals = np.ones(len(test_idx))
        if adv_lookup is not None:
            adv_vals = (keys_df.merge(adv_lookup, on=[date_col, id_col], how="left")
                                  [adv_col].values)
        else:
            adv_vals = None
        if retvol_lookup is not None:
            retvol_vals = (keys_df.merge(retvol_lookup, on=[date_col, id_col], how="left")
                                     ["retvol"].values)
        else:
            retvol_vals = None
        del me_lookup, adv_lookup, retvol_lookup, keys_df
        gc.collect()

        for name, pred_arr in pred_arrays.items():
            pred_df = pd.DataFrame({
                "date":    test_idx,
                "permno":  test_permnos,
                "pred":    pred_arr,
                "ret":     true_arr,
                "me":      me_vals,
            })
            if adv_vals is not None:
                pred_df["adv"] = adv_vals
            pred_wide = pred_df.pivot(index="date", columns="permno", values="pred")
            ret_wide  = pred_df.pivot(index="date", columns="permno", values="ret")
            me_wide   = pred_df.pivot(index="date", columns="permno", values="me")
            adv_wide  = (
                pred_df.pivot(index="date", columns="permno", values="adv")
                if adv_vals is not None else None
            )

            # Pivot retvol the same way as me/adv (date × permno wide frame)
            retvol_wide = (
                pd.DataFrame({
                    "date": pd.to_datetime(test_idx),
                    "permno": test_permnos,
                    "retvol": retvol_vals,
                }).pivot(index="date", columns="permno", values="retvol")
                if retvol_vals is not None else None
            )
            builder = DecilePortfolioBuilder(
                n_deciles=self.n_deciles,
                weighting=self.weighting,
                tc_model=self.tc_model,
            )
            net_r, gross_r, turn_r = builder.build(
                pred_wide, ret_wide, me_wide, adv=adv_wide, retvol=retvol_wide,
            )
            portfolio_returns[name] = net_r
            portfolio_returns_gross[name] = gross_r
            portfolio_turnover[name] = turn_r

        # ── Compute OOS metrics ───────────────────────────────────────────
        from src.evaluation.metrics import oos_r2, sharpe_ratio, diebold_mariano
        metrics = {}
        for name, pred_arr in pred_arrays.items():
            valid = ~np.isnan(pred_arr) & ~np.isnan(true_arr)
            r2    = oos_r2(true_arr[valid], pred_arr[valid]) * 100
            hl    = portfolio_returns[name].get("H-L", pd.Series(dtype=float))
            sr    = sharpe_ratio(hl.dropna()) if len(hl.dropna()) > 0 else np.nan
            hl_g = portfolio_returns_gross[name].get("H-L", pd.Series(dtype=float))
            sr_g = sharpe_ratio(hl_g.dropna()) if len(hl_g.dropna()) > 0 else np.nan
            hl_to = portfolio_turnover[name].get("H-L", pd.Series(dtype=float))
            to_m = float(hl_to.dropna().mean()) if len(hl_to.dropna()) > 0 else np.nan
            metrics[name] = {
                "oos_r2_pct": round(r2, 3),
                "hl_sharpe":  round(sr, 3) if not np.isnan(sr) else np.nan,
                "hl_sharpe_gross": round(sr_g, 3) if not np.isnan(sr_g) else np.nan,
                "hl_mean_turnover_one_way": round(to_m, 6) if not np.isnan(to_m) else np.nan,
                "hl_engine_tc_bps": self._tc_bps,
                "hl_returns_are_net_of_tc": bool(self.tc_model.cost > 0),
            }

        return {
            "predictions":              pred_arrays,
            "true_returns":             true_arr,
            "test_dates":               test_idx,
            "test_permnos":             test_permnos,
            "portfolio_returns":        portfolio_returns,
            "portfolio_returns_gross":   portfolio_returns_gross,
            "portfolio_turnover":       portfolio_turnover,
            "metrics":                  metrics,
        }