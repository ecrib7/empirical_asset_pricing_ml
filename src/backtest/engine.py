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
    """Simple proportional one-way transaction cost model (cost in bps)."""

    def __init__(self, cost_bps: float = 10.0):
        self.cost = cost_bps / 10_000.0

    def period_one_way_turnover(self, w_new: pd.Series, w_old: pd.Series) -> float:
        return one_way_portfolio_turnover(w_new, w_old)

    def period_turnover_cost(self, w_new: pd.Series, w_old: pd.Series) -> float:
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
        checkpoint_dir: str | None = None,
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
        self.checkpoint_dir = Path(checkpoint_dir or _default_checkpoint_dir())
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, models: dict) -> Path:
        """Path keyed by sorted model names so same set of models = same checkpoint."""
        key = ",".join(sorted(models.keys()))
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        # Include readable model names in filename for human inspection
        safe_key = key.replace("/", "_").replace("+", "p")[:80]
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

            yr_end = yr_start + pd.DateOffset(years=1) - pd.DateOffset(days=1)
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

            X_tr = train[feat_cols].fillna(0)
            X_v = val[feat_cols].fillna(0) if len(val) > 0 else None
            X_te = test[feat_cols].fillna(0)
            y_train = train[target_col].values
            y_val = val[target_col].values if len(val) > 0 else None
            y_test = test[target_col].values

            logger.info(f"Test year {year}: "
                        f"train={len(train):,}  val={len(val):,}  test={len(test):,}")

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
                    year_preds[name] = [np.nan] * len(test)

            # All models attempted — commit this year's predictions
            for name in models:
                predictions[name].extend(year_preds[name])
            true_rets.extend(y_test.astype(np.float32).tolist())
            test_dates_all.extend(test[date_col].values.tolist())
            test_permnos.extend(test[id_col].values.tolist())
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

        # ── Assemble panel ────────────────────────────────────────────────
        test_idx   = pd.to_datetime(test_dates_all)
        true_arr   = np.array(true_rets, dtype=np.float32)
        pred_arrays = {n: np.array(v, dtype=np.float32) for n, v in predictions.items()}

        # ── Build wide prediction DataFrames for portfolio construction ──
        portfolio_returns: Dict[str, Dict[str, pd.Series]] = {}
        portfolio_returns_gross: Dict[str, Dict[str, pd.Series]] = {}
        portfolio_turnover: Dict[str, Dict[str, pd.Series]] = {}
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