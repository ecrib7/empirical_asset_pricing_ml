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

Target timing (CRSP month-end rows):
    ``ret`` is the realised return for the calendar month of ``date`` (same row
    as month-end observables).  Supervised targets use ``ret_fwd`` = next month's
    return within the same ``permno``, so models trained at ``date`` t only
    predict returns realised over month t+1.  Portfolio sorts at ``date`` t use
    those predictions against ``ret_fwd`` (same-row realised one-month-ahead
    return).  We exclude raw ``ret`` from features when ``ret_fwd`` is the
    target to avoid labelling the contemporaneous return as a regressor.
    Excess vs risk-free is unchanged from the input panel: if ``ret`` is already
    excess, so is ``ret_fwd``; the pipeline does not merge RF here.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.config import FREQ_MONTH_END, FREQ_YEAR_START

logger = logging.getLogger(__name__)


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
    """Columns used as X; raw contemporaneous ``ret`` is excluded when target is ``ret_fwd``."""
    exclude = {id_col, date_col, target_col, me_col}
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
    """
    Simple proportional transaction cost model.

    Cost per trade = cost_bps / 10,000 (one-way).
    Applied to one-way turnover: half the L1 norm of weight changes,
    ``Σ_i |w_{i,t}-w_{i,t-1}| / 2``, matching a buy/sell decomposition on
    long-only weights that sum to one; for dollar-neutral long–short vectors
    the same formula scales proportional trading in each leg.

    Execution (close-to-close monthly approximation):
        Weights in month ``t`` are those held to earn the realised return for the
        period labelled ``t`` (already aligned with ``ret_fwd`` upstream).  Cost
        for that return is charged against turnover from weights at ``t-1`` to
        weights at ``t`` (month-over-month rebalancing).  This is a stylised
        calendar-month, month-end mark — not intra-month high/low execution.
    """

    def __init__(self, cost_bps: float = 10.0):
        """
        Parameters
        ----------
        cost_bps : one-way transaction cost in basis points
        """
        self.cost = cost_bps / 10_000.0

    def period_one_way_turnover(self, w_new: pd.Series, w_old: pd.Series) -> float:
        """One-way turnover ``Σ|Δw|/2`` (same units as ``cost_series`` row sum)."""
        return one_way_portfolio_turnover(w_new, w_old)

    def period_turnover_cost(self, w_new: pd.Series, w_old: pd.Series) -> float:
        """
        One-way turnover cost for a single rebalancing: same formula as one row
        of ``cost_series``, for weight vectors indexed by security (NaNs treated
        as zero weight on union of indices).
        """
        return float(self.cost * self.period_one_way_turnover(w_new, w_old))

    def net_return(
        self,
        gross_ret: pd.Series,
        weights: pd.DataFrame,
        weights_prev: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute net-of-transaction-cost returns.

        net_ret[t] = gross_ret[t] - cost × Σ|w[t] - w[t-1]| / 2
        """
        # Align weights
        idx = gross_ret.index
        w   = weights.reindex(idx).fillna(0.0)
        w_l = weights_prev.reindex(idx).fillna(0.0)
        turnover = (w - w_l).abs().sum(axis=1) / 2.0   # one-way
        return gross_ret - self.cost * turnover

    def cost_series(
        self,
        weights: pd.DataFrame,
        weights_prev: pd.DataFrame,
    ) -> pd.Series:
        """Return the per-period transaction cost."""
        w   = weights.fillna(0.0)
        w_l = weights_prev.fillna(0.0)
        return self.cost * (w - w_l).abs().sum(axis=1) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
#  Decile portfolio builder
# ─────────────────────────────────────────────────────────────────────────────

class DecilePortfolioBuilder:
    """
    Sorts stocks into deciles each month based on model predictions.
    Constructs value-weighted (or equal-weighted) portfolios.
    Reports GKX Table 7 statistics.

    Timing (consistent with ``ret_fwd`` upstream):
        * Predictions at calendar month-end ``t`` use information through ``t``.
        * Decile membership and portfolio weights are fixed at ``t`` for the
          realised return attributed to row ``t`` (next-month / ``ret_fwd``
          convention from the feature matrix).
        * Transaction costs (when ``tc_model`` is set) deduct one-way turnover
          from weights at ``t-1`` to weights at ``t`` from that same gross return.
        Close-to-close: one gross return per calendar month; no intra-month
        path for execution prices beyond this month-end approximation.
    """

    def __init__(
        self,
        n_deciles: int = 10,
        weighting: str = "value",         # "value" or "equal"
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
    ) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        Parameters
        ----------
        predictions : DataFrame (index=date, columns=permno) of predicted returns
        returns     : DataFrame (index=date, columns=permno) of realised returns
        market_caps : DataFrame (index=date, columns=permno) of market caps (for VW)

        Returns
        -------
        net, gross, turnover_one_way
            Three dicts keyed by decile label and ``"H-L"``.  **net** subtracts
            engine TC when ``tc_model.cost`` > 0; **gross** is always pre-cost;
            **turnover_one_way** is ``Σ|Δw|/2`` each month (0 cost when cost is 0).
        """
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

        for t in dates:
            pred_t = predictions.loc[t].dropna()
            ret_t = returns.loc[t].reindex(pred_t.index).dropna()
            common = pred_t.index.intersection(ret_t.index)
            if len(common) < self.n_deciles:
                continue

            pred_t = pred_t.loc[common]
            ret_t = ret_t.loc[common]

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
        """
        Reproduce GKX Table 7 columns:
        Pred | Avg Realized | Std | Sharpe
        """
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
    """
    Campbell & Thompson (2008) market timing strategy.

    Each month t, scale position proportional to predicted excess return,
    with constraints:
    - No short sales for long-only portfolios
    - Max leverage ≤ max_leverage
    """

    def __init__(self, max_leverage: float = 1.5):
        self.max_leverage = max_leverage

    def returns(
        self,
        predicted: pd.Series,
        realised: pd.Series,
        rf: float = 0.0,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Parameters
        ----------
        predicted : monthly predicted excess returns
        realised  : monthly realised excess returns
        rf        : risk-free rate (monthly)

        Returns
        -------
        (timed_returns, buy_and_hold_returns)
        """
        common = predicted.index.intersection(realised.index)
        pred   = predicted.loc[common]
        real   = realised.loc[common]

        # Scale signal to [0, max_leverage] (long-only)
        # Following C-T: weight proportional to prediction
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
        """Annualised SR improvement from market timing."""
        from src.evaluation.metrics import sharpe_ratio
        timed, bah = self.returns(predicted, realised)
        return sharpe_ratio(timed, annualise) - sharpe_ratio(bah, annualise)


# ─────────────────────────────────────────────────────────────────────────────
#  Full backtest engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Orchestrates the GKX (2019) recursive out-of-sample backtest.

    Timeline:
    ─────────────────────────────────────────────────────────
    1957          1974    1986             2016
    |←─ TRAIN ───→|←─VAL─→|←──── TEST ────→|
                   rolling window each year

    At each test year:
      1. Expand training window by 1 year
      2. Roll validation window forward 12 months
      3. Re-fit all models on train+val
      4. Predict returns for the next 12 months (test window)
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
    ):
        self.train_start = pd.Timestamp(train_start)
        self.val_start   = pd.Timestamp(val_start)
        self.val_end     = pd.Timestamp(val_end)
        self.test_start  = pd.Timestamp(test_start)
        self.test_end    = pd.Timestamp(test_end)
        self.n_deciles   = n_deciles
        self.weighting   = weighting
        self._tc_bps     = float(tc_bps)
        self.tc_model    = TransactionCostModel(tc_bps)

    def run(
        self,
        feature_matrix: pd.DataFrame,
        models: dict,
        target_col: str = "ret_fwd",
        id_col: str = "permno",
        date_col: str = "date",
        me_col: str = "me",
    ) -> Dict:
        """
        Execute the recursive annual backtest.

        Parameters
        ----------
        feature_matrix : wide panel (permno, date, ret, ret_fwd, me, feature_1, …)
        models         : {model_name: model_instance}  (from get_all_models())
        target_col     : supervised target (default ``ret_fwd``, next-month return)
        id_col         : cross-sectional identifier
        date_col       : time identifier
        me_col         : market equity (for value weights)

        Returns
        -------
        {
          "predictions": {model: (date, permno, pred_ret) dataframe},
          "test_y":      panel Series of actual returns,
          "test_dates":  date array,
          "portfolio_returns": {model: {decile: pd.Series}},
          "metrics": {model: {"oos_r2": float, ...}}
        }
        """
        fm = feature_matrix.sort_values([date_col, id_col]).copy()
        if target_col not in fm.columns:
            raise KeyError(
                f"Missing target column {target_col!r}. "
                "Call add_forward_return_target() before BacktestEngine.run()."
            )
        feat_cols = feature_columns_for_training(
            fm, target_col, id_col=id_col, date_col=date_col, me_col=me_col
        )

        first_train_end = self.test_start - pd.DateOffset(years=1)
        assert first_train_end.year == self.val_end.year, (
            f"val_end year {self.val_end.year} does not match "
            f"first walk-forward train_end year {first_train_end.year}"
        )

        all_test_dates = pd.date_range(self.test_start, self.test_end, freq=FREQ_YEAR_START)
        predictions    = {name: [] for name in models}
        true_rets      = []
        test_dates_all = []
        test_permnos   = []

        for yr_start in all_test_dates:
            yr_end = yr_start + pd.DateOffset(years=1) - pd.DateOffset(days=1)

            # ── sample windows ──────────────────────────────────────────────
            train_end_yr = yr_start - pd.DateOffset(years=1)

            mask_train = (fm[date_col] >= self.train_start) & (fm[date_col] <= train_end_yr)
            mask_val   = (fm[date_col] > train_end_yr) & (fm[date_col] <= yr_start - pd.DateOffset(days=1))
            mask_test  = (fm[date_col] >= yr_start) & (fm[date_col] <= yr_end)

            train = fm[mask_train].dropna(subset=[target_col])
            val   = fm[mask_val].dropna(subset=[target_col])
            test  = fm[mask_test].dropna(subset=[target_col])

            if len(train) < 100 or len(test) == 0:
                continue

            X_tr = train[feat_cols].fillna(0)
            X_v = val[feat_cols].fillna(0) if len(val) > 0 else None
            X_te = test[feat_cols].fillna(0)
            y_train = train[target_col].values
            y_val = val[target_col].values if len(val) > 0 else None
            y_test = test[target_col].values

            logger.info(f"Test year {yr_start.year}: "
                        f"train={len(train):,}  val={len(val):,}  test={len(test):,}")

            # ── fit & predict ───────────────────────────────────────────────
            for name, model in models.items():
                try:
                    if hasattr(model, "fit"):
                        model.fit(X_tr, y_train, X_v, y_val)
                    pred = model.predict(X_te)
                    predictions[name].extend(pred.tolist())
                except Exception as e:
                    logger.error(f"{name} failed at {yr_start.year}: {e}")
                    predictions[name].extend([np.nan] * len(test))

            true_rets.extend(y_test.tolist())
            test_dates_all.extend(test[date_col].values.tolist())
            test_permnos.extend(test[id_col].values.tolist())

        # ── Assemble panel ────────────────────────────────────────────────
        test_idx   = pd.to_datetime(test_dates_all)
        true_arr   = np.array(true_rets)
        pred_arrays = {n: np.array(v) for n, v in predictions.items()}

        # ── Build wide prediction DataFrames for portfolio construction ──
        me_test = fm[fm[date_col].isin(test_idx)].set_index([date_col, id_col])[me_col]

        portfolio_returns: Dict[str, Dict[str, pd.Series]] = {}
        portfolio_returns_gross: Dict[str, Dict[str, pd.Series]] = {}
        portfolio_turnover: Dict[str, Dict[str, pd.Series]] = {}
        for name, pred_arr in pred_arrays.items():
            # ``ret`` here is the realised return over the forecast horizon (``ret_fwd``),
            # aligned with ``pred`` at the same (date, permno) signal row.
            pred_df = pd.DataFrame({
                "date":    test_idx,
                "permno":  test_permnos,
                "pred":    pred_arr,
                "ret":     true_arr,
                "me":      fm.set_index([date_col, id_col]).reindex(
                    zip(test_idx, test_permnos)
                )[me_col].values if me_col in fm.columns else np.ones(len(test_idx)),
            })
            pred_wide = pred_df.pivot(index="date", columns="permno", values="pred")
            ret_wide  = pred_df.pivot(index="date", columns="permno", values="ret")
            me_wide   = pred_df.pivot(index="date", columns="permno", values="me")

            builder = DecilePortfolioBuilder(
                n_deciles=self.n_deciles,
                weighting=self.weighting,
                tc_model=self.tc_model,
            )
            net_r, gross_r, turn_r = builder.build(pred_wide, ret_wide, me_wide)
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
