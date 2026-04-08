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
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.config import FREQ_MONTH_END, FREQ_YEAR_START

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Transaction Cost Model
# ─────────────────────────────────────────────────────────────────────────────

class TransactionCostModel:
    """
    Simple proportional transaction cost model.

    Cost per trade = cost_bps / 10,000 (one-way).
    Applied to absolute changes in portfolio weights.
    """

    def __init__(self, cost_bps: float = 10.0):
        """
        Parameters
        ----------
        cost_bps : one-way transaction cost in basis points
        """
        self.cost = cost_bps / 10_000.0

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
    ) -> Dict[str, pd.Series]:
        """
        Parameters
        ----------
        predictions : DataFrame (index=date, columns=permno) of predicted returns
        returns     : DataFrame (index=date, columns=permno) of realised returns
        market_caps : DataFrame (index=date, columns=permno) of market caps (for VW)

        Returns
        -------
        {decile_label: monthly_return_series}
        Including 'H-L' (top minus bottom decile spread).
        """
        dates    = predictions.index.intersection(returns.index)
        port_ret = {str(d): [] for d in range(1, self.n_deciles + 1)}
        port_ret["H-L"] = []
        date_idx = []

        prev_weights = {str(d): pd.Series(dtype=float)
                        for d in range(1, self.n_deciles + 1)}
        prev_weights["H-L"] = pd.Series(dtype=float)

        for t in dates:
            pred_t = predictions.loc[t].dropna()
            ret_t  = returns.loc[t].reindex(pred_t.index).dropna()
            common = pred_t.index.intersection(ret_t.index)
            if len(common) < self.n_deciles:
                continue

            pred_t = pred_t.loc[common]
            ret_t  = ret_t.loc[common]

            # Assign decile labels (1 = lowest, 10 = highest predicted return)
            try:
                labels = pd.qcut(pred_t, q=self.n_deciles,
                                 labels=range(1, self.n_deciles + 1))
            except ValueError:
                # Duplicate bin edges → rank-based fallback
                labels = pd.qcut(pred_t.rank(method="first"), q=self.n_deciles,
                                 labels=range(1, self.n_deciles + 1))

            for d in range(1, self.n_deciles + 1):
                mask = labels == d
                if mask.sum() == 0:
                    port_ret[str(d)].append(np.nan)
                    continue
                stocks = common[mask]
                w = self._weights(stocks, t, market_caps)
                r = (w * ret_t.loc[stocks]).sum()
                port_ret[str(d)].append(r)

            # H-L portfolio
            top_mask    = labels == self.n_deciles
            bot_mask    = labels == 1
            top_stocks  = common[top_mask]
            bot_stocks  = common[bot_mask]
            w_top = self._weights(top_stocks, t, market_caps)
            w_bot = self._weights(bot_stocks, t, market_caps)
            r_hl  = (w_top * ret_t.loc[top_stocks]).sum() \
                  - (w_bot * ret_t.loc[bot_stocks]).sum()
            port_ret["H-L"].append(r_hl)
            date_idx.append(t)

        result = {}
        for key, rets in port_ret.items():
            result[key] = pd.Series(rets, index=date_idx, name=key)
        return result

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
        w = pred / pred.std()
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
        self.tc_model    = TransactionCostModel(tc_bps)

    def run(
        self,
        feature_matrix: pd.DataFrame,
        models: dict,
        target_col: str = "ret",
        id_col: str = "permno",
        date_col: str = "date",
        me_col: str = "me",
    ) -> Dict:
        """
        Execute the recursive annual backtest.

        Parameters
        ----------
        feature_matrix : wide panel (permno, date, ret, me, feature_1, …)
        models         : {model_name: model_instance}  (from get_all_models())
        target_col     : name of the return column
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
        feat_cols = [c for c in fm.columns
                     if c not in [id_col, date_col, target_col, me_col]]

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

            train = fm[mask_train]
            val   = fm[mask_val]
            test  = fm[mask_test]

            if len(train) < 100 or len(test) == 0:
                continue

            X_train = train[feat_cols].fillna(0).values
            y_train = train[target_col].values
            X_val   = val[feat_cols].fillna(0).values if len(val) > 0 else None
            y_val   = val[target_col].values if len(val) > 0 else None
            X_test  = test[feat_cols].fillna(0).values
            y_test  = test[target_col].values

            logger.info(f"Test year {yr_start.year}: "
                        f"train={len(train):,}  val={len(val):,}  test={len(test):,}")

            # ── fit & predict ───────────────────────────────────────────────
            for name, model in models.items():
                try:
                    if hasattr(model, "fit"):
                        # Handle DataFrame-aware models (OLS-3, etc.)
                        if name == "OLS-3":
                            model.fit(train[feat_cols].fillna(0), y_train)
                        else:
                            model.fit(X_train, y_train, X_val, y_val)
                    pred = model.predict(X_test if name != "OLS-3"
                                        else test[feat_cols].fillna(0))
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

        portfolio_returns = {}
        for name, pred_arr in pred_arrays.items():
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
            port_rets = builder.build(pred_wide, ret_wide, me_wide)
            portfolio_returns[name] = port_rets

        # ── Compute OOS metrics ───────────────────────────────────────────
        from src.evaluation.metrics import oos_r2, sharpe_ratio, diebold_mariano
        metrics = {}
        for name, pred_arr in pred_arrays.items():
            valid = ~np.isnan(pred_arr) & ~np.isnan(true_arr)
            r2    = oos_r2(true_arr[valid], pred_arr[valid]) * 100
            hl    = portfolio_returns[name].get("H-L", pd.Series(dtype=float))
            sr    = sharpe_ratio(hl.dropna()) if len(hl.dropna()) > 0 else np.nan
            metrics[name] = {
                "oos_r2_pct": round(r2, 3),
                "hl_sharpe":  round(sr, 3) if not np.isnan(sr) else np.nan,
            }

        return {
            "predictions":       pred_arrays,
            "true_returns":      true_arr,
            "test_dates":        test_idx,
            "test_permnos":      test_permnos,
            "portfolio_returns": portfolio_returns,
            "metrics":           metrics,
        }
