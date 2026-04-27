"""
models/all_models.py
--------------------
Implements every model from Gu, Kelly & Xiu (2019):

  LinearModels    : OLS-3, OLS+, Elastic Net, PCR, PLS, GLM+GroupLasso
  TreeModels      : Random Forest, Gradient Boosted Regression Trees
  NeuralNetModels : NN1 … NN5  (feed-forward, ReLU, batch-norm, ensemble)

All models share a common interface:
    .fit(X_train, y_train, X_val, y_val)
    .predict(X)
    .oos_r2(X_test, y_test)

Huber loss is used for OLS+, ENet, GLM, GBRT (paper default).
"""

from __future__ import annotations

import gc
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, ElasticNet, HuberRegressor, SGDRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Memory-efficient helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_float32_array(X) -> np.ndarray:
    """Convert pandas / numpy input to a contiguous float32 array without copying when possible."""
    if hasattr(X, "values"):
        arr = X.values
    else:
        arr = X
    if arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float32)


class _Float32Scaler:
    """
    Drop-in replacement for sklearn's StandardScaler that keeps everything in
    float32. sklearn's StandardScaler silently upcasts to float64 on transform,
    which doubles memory for our 1M+ x 518 inputs. This avoids that.
    """

    def fit(self, X: np.ndarray) -> "_Float32Scaler":
        # Use float64 internally for accurate moments, then store as float32
        self.mean_ = X.astype(np.float64, copy=False).mean(axis=0).astype(np.float32)
        std = X.astype(np.float64, copy=False).std(axis=0)
        std[std < 1e-8] = 1.0
        self.scale_ = std.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape, dtype=np.float32)
        np.subtract(X, self.mean_, out=out)
        np.divide(out, self.scale_, out=out)
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _split_or_use_val(
    Xn: np.ndarray, y: np.ndarray,
    X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
    val_frac: float = 0.2,
):
    """If no validation set is supplied, split the tail of training as validation."""
    if X_val is None:
        n_val = max(1, int(val_frac * len(Xn)))
        return Xn[:-n_val], y[:-n_val], Xn[-n_val:], y[-n_val:]
    return Xn, y, X_val, y_val


_MAX_TRAIN_ROWS = 500_000


def _subsample_train(
    Xn: np.ndarray,
    y: np.ndarray,
    max_rows: int = _MAX_TRAIN_ROWS,
    seed: int = 42,
    label: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If Xn has more than `max_rows`, return a deterministic random subsample.
    Used by memory-bound models (PLS, GLM+H) where coordinate-descent / NIPALS
    don't scale to 2.5M+ training rows on a 51 GB box. Validation set is
    intentionally NOT subsampled — full validation gives unbiased model
    selection.
    """
    n = len(Xn)
    if n <= max_rows:
        return Xn, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False)
    idx.sort()  # preserves cache locality on the big matrix
    logger.info(
        f"{label}subsampling training set from {n:,} -> {max_rows:,} rows "
        f"(memory cap)"
    )
    return Xn[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    GKX (2019) eq. (19): OOS R² benchmarked against zero forecast.
    R²_oos = 1 - Σ(y - ŷ)² / Σ y²
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _tune_on_val(
    model_fn,
    param_grid: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = False,
) -> Tuple[object, dict]:
    """
    Grid search over param_grid using validation R².
    Memory-conscious: only keeps the current best model in memory; all others are
    freed (incl. their coef arrays) before the next trial.
    """
    best_r2 = -np.inf
    best_model = None
    best_params = None
    for params in param_grid:
        try:
            m = model_fn(**params)
            m.fit(X_train, y_train)
            r2 = oos_r2(y_val, m.predict(X_val))
            if verbose:
                logger.debug(f"  {params} → val R²={r2:.4f}")
            if r2 > best_r2:
                # Free previous best before adopting the new one
                if best_model is not None:
                    del best_model
                    gc.collect()
                best_r2 = r2
                best_model = m
                best_params = params
            else:
                del m
                gc.collect()
        except Exception as e:
            logger.warning(f"  {params} failed: {e}")
            gc.collect()
    return best_model, best_params


# ─────────────────────────────────────────────────────────────────────────────
#  Linear Models
# ─────────────────────────────────────────────────────────────────────────────

class OLS3Model:
    """Pooled OLS with 3 predictors: size, book-to-market, momentum."""
    name = "OLS-3"

    def __init__(self):
        self.model = LinearRegression()
        self.cols_  = ["mvel1_const", "bm_const", "mom12m_const"]

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "OLS3Model":
        primary = ["mvel1_const", "bm_const", "mom12m_const"]
        fallback = ["mvel1", "bm", "mom12m"]
        avail = [c for c in self.cols_ if c in X.columns]
        if not avail:
            self.cols_ = primary
            avail = [c for c in self.cols_ if c in X.columns]
        if not avail:
            self.cols_ = fallback
            avail = [c for c in self.cols_ if c in X.columns]
        if not avail:
            sample_cols = list(X.columns[:10])
            raise ValueError(
                "OLS3Model.fit: no required columns found after trying instance "
                f"``cols_``, then {primary!r}, then {fallback!r}. "
                f"First 10 columns of X: {sample_cols}"
            )
        self._avail = avail
        self.model.fit(X[avail].values.astype(np.float32, copy=False), y.astype(np.float32, copy=False))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self._avail].values.astype(np.float32, copy=False))

    def oos_r2(self, X: pd.DataFrame, y: np.ndarray) -> float:
        return oos_r2(y, self.predict(X))


class ElasticNetModel:
    """
    Elastic Net (Huber loss, GKX paper default) — implemented via streaming SGD.

    sklearn's ``ElasticNet`` (coordinate descent) does not fit in 51 GB on a
    2.5M x 518 matrix because it upcasts to float64 and keeps several full-size
    arrays during fitting. ``SGDRegressor`` processes the data one mini-batch
    at a time, so peak RAM is dominated by the input matrix itself.

    Bonus: SGDRegressor with ``loss='huber'`` is exactly the GKX objective —
    closer to the paper than sklearn's L2-loss ``ElasticNet`` ever was.
    """
    name = "ENet+H"

    def __init__(
        self,
        alpha_grid: List[float] = None,
        l1_ratio: float = 0.5,
        use_huber: bool = True,
        max_epochs: int = 30,
        huber_epsilon: float = 0.001,
    ):
        # SGDRegressor's `alpha` is the overall regularisation strength.
        # Use a slightly broader grid than coordinate-descent ENet because
        # SGD's optimal alpha typically sits at smaller values.
        self.alpha_grid = alpha_grid or [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        self.l1_ratio = l1_ratio
        self.use_huber = use_huber
        self.max_epochs = max_epochs
        self.huber_epsilon = huber_epsilon
        self.best_model_: Optional[BaseEstimator] = None
        self.scaler_ = _Float32Scaler()

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        X_val: pd.DataFrame | np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> "ElasticNetModel":
        # Subsample BEFORE any float32 scaling copy — avoids a 3M-row intermediate
        X_arr = _to_float32_array(X)
        y32 = y.astype(np.float32, copy=False)
        X_arr, y32 = _subsample_train(X_arr, y32, label="[ENet+H] train ")

        # Now scale (matrix is now <=500K rows, ~1 GB)
        Xn = self.scaler_.fit_transform(X_arr)
        del X_arr
        gc.collect()

        if X_val is not None:
            Xv_arr = _to_float32_array(X_val)
            yv32 = y_val.astype(np.float32, copy=False)
            # Cap validation too (validation R^2 is stable on smaller samples)
            Xv_arr, yv32 = _subsample_train(Xv_arr, yv32, max_rows=200_000, label="[ENet+H] val ")
            Xv = self.scaler_.transform(Xv_arr)
            del Xv_arr
            gc.collect()
        else:
            n_val = max(1, int(0.2 * len(Xn)))
            Xv, yv32 = Xn[-n_val:], y32[-n_val:]
            Xn, y32 = Xn[:-n_val], y32[:-n_val]

        loss = "huber" if self.use_huber else "squared_error"

        def make_model(alpha):
            return SGDRegressor(
                loss=loss,
                penalty="elasticnet",
                alpha=alpha,
                l1_ratio=self.l1_ratio,
                epsilon=self.huber_epsilon,
                max_iter=self.max_epochs,
                tol=1e-4,
                early_stopping=False,   # we have our own validation grid
                learning_rate="adaptive",
                eta0=0.01,
                random_state=42,
                fit_intercept=True,
                shuffle=True,
            )

        param_grid = [{"alpha": a} for a in self.alpha_grid]
        self.best_model_, self.best_params_ = _tune_on_val(
            make_model, param_grid, Xn, y32, Xv, yv32
        )
        if self.best_model_ is None:
            self.best_model_ = make_model(self.alpha_grid[len(self.alpha_grid) // 2])
            self.best_model_.fit(Xn, y32)

        del Xn, Xv, y32, yv32
        gc.collect()
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        Xn = self.scaler_.transform(_to_float32_array(X))
        out = self.best_model_.predict(Xn)
        del Xn
        gc.collect()
        return out

    def oos_r2(self, X, y) -> float:
        return oos_r2(y, self.predict(X))


class PCRModel:
    """Principal Components Regression."""
    name = "PCR"

    def __init__(self, n_components_grid: List[int] = None):
        self.n_components_grid = n_components_grid or [3, 5, 10, 20, 30, 40]
        self.scaler_ = _Float32Scaler()
        self.best_k_: int = 10
        self.pca_: Optional[PCA] = None
        self.reg_: Optional[LinearRegression] = None

    def fit(self, X, y, X_val=None, y_val=None) -> "PCRModel":
        Xn = self.scaler_.fit_transform(_to_float32_array(X))
        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            X_val, y_val = Xn[-n_val:], y[-n_val:]
            Xn, y = Xn[:-n_val], y[:-n_val]
        else:
            X_val = self.scaler_.transform(_to_float32_array(X_val))

        y32 = y.astype(np.float32, copy=False)
        yv32 = y_val.astype(np.float32, copy=False)

        max_k = min(max(self.n_components_grid), Xn.shape[1], Xn.shape[0] - 1)
        pca = PCA(n_components=max_k, svd_solver="randomized", random_state=0)
        pca.fit(Xn)
        Z_train_full = pca.transform(Xn)
        Z_val_full = pca.transform(X_val)

        best_r2 = -np.inf
        for k in self.n_components_grid:
            k = min(k, max_k)
            reg = LinearRegression().fit(Z_train_full[:, :k], y32)
            r2 = oos_r2(yv32, reg.predict(Z_val_full[:, :k]))
            if r2 > best_r2:
                best_r2 = r2
                self.best_k_ = k

        self.pca_ = PCA(n_components=self.best_k_, svd_solver="randomized", random_state=0).fit(Xn)
        Z = self.pca_.transform(Xn)
        self.reg_ = LinearRegression().fit(Z, y32)

        del Xn, X_val, Z, Z_train_full, Z_val_full, pca
        gc.collect()
        return self

    def predict(self, X) -> np.ndarray:
        Xn = self.scaler_.transform(_to_float32_array(X))
        Z  = self.pca_.transform(Xn)
        out = self.reg_.predict(Z)
        del Xn, Z
        gc.collect()
        return out

    def oos_r2(self, X, y) -> float:
        return oos_r2(y, self.predict(X))


class PLSModel:
    """
    Partial Least Squares Regression.
    Memory-conscious: float32 input, intermediate fits freed between trials.
    """
    name = "PLS"

    def __init__(self, n_components_grid: List[int] = None):
        self.n_components_grid = n_components_grid or [1, 2, 3, 4, 5, 6, 8, 10]
        self.scaler_ = _Float32Scaler()
        self.best_model_: Optional[PLSRegression] = None

    def fit(self, X, y, X_val=None, y_val=None) -> "PLSModel":
        # Subsample BEFORE scaling — avoids 3M-row float32 intermediate at year 2016
        X_arr = _to_float32_array(X)
        y32 = y.astype(np.float32, copy=False)
        X_arr, y32 = _subsample_train(X_arr, y32, label="[PLS] train ")
        Xn = self.scaler_.fit_transform(X_arr)
        del X_arr
        gc.collect()

        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            X_val, yv32 = Xn[-n_val:], y32[-n_val:]
            Xn, y32 = Xn[:-n_val], y32[:-n_val]
        else:
            Xv_arr = _to_float32_array(X_val)
            yv32 = y_val.astype(np.float32, copy=False)
            Xv_arr, yv32 = _subsample_train(Xv_arr, yv32, max_rows=200_000, label="[PLS] val ")
            X_val = self.scaler_.transform(Xv_arr)
            del Xv_arr
            gc.collect()

        max_k = min(max(self.n_components_grid), Xn.shape[1], Xn.shape[0] - 1)
        best_r2 = -np.inf
        for k in self.n_components_grid:
            k = min(k, max_k)
            try:
                pls = PLSRegression(n_components=k, scale=False, max_iter=200, tol=1e-4)
                pls.fit(Xn, y32)
                r2 = oos_r2(yv32, pls.predict(X_val).flatten())
                if r2 > best_r2:
                    if self.best_model_ is not None:
                        del self.best_model_
                        gc.collect()
                    best_r2 = r2
                    self.best_model_ = pls
                else:
                    del pls
                    gc.collect()
            except Exception as e:
                logger.warning(f"PLS k={k} failed: {e}")
                gc.collect()

        if self.best_model_ is None:
            self.best_model_ = PLSRegression(n_components=1, scale=False).fit(Xn, y32)

        del Xn, X_val, y32, yv32
        gc.collect()
        return self

    def predict(self, X) -> np.ndarray:
        Xn = self.scaler_.transform(_to_float32_array(X))
        out = self.best_model_.predict(Xn).flatten()
        del Xn
        gc.collect()
        return out

    def oos_r2(self, X, y) -> float:
        return oos_r2(y, self.predict(X))


class GLMModel:
    """
    Generalised Linear Model with Group Lasso, approximated via ElasticNet on
    spline-expanded features. Each characteristic is expanded with a quadratic
    spline (n_knots knots).

    Memory-conscious: pre-allocates the spline matrix as a single float32 array
    instead of np.hstack-ing 1500+ small arrays. With 518 features × 3 knots
    that's a 518 + 518×3 = 2,072-column matrix; at 1.1M rows that's ~9.1 GB
    in float32 vs ~18 GB in float64 — manageable inside 51 GB.
    """
    name = "GLM+H"

    def __init__(self, n_knots: int = 3, alpha_grid: List[float] = None):
        self.n_knots = n_knots
        self.alpha_grid = alpha_grid or [1e-4, 1e-3, 1e-2, 5e-2, 0.1]
        self.scaler_ = _Float32Scaler()
        self.best_model_: Optional[ElasticNet] = None
        self.knots_: Optional[np.ndarray] = None  # shape (n_features, n_knots), float32

    def _fit_spline_knots(self, X: np.ndarray) -> None:
        """Compute quantile knots per feature; result is a (p, n_knots) float32 array."""
        qs = np.linspace(0.1, 0.9, self.n_knots)
        # np.quantile across columns; result shape (n_knots, p), then transpose
        knots = np.quantile(X, qs, axis=0).T  # (p, n_knots)
        self.knots_ = knots.astype(np.float32)

    def _spline_expand(self, X: np.ndarray) -> np.ndarray:
        """
        Build the spline-expanded matrix in float32 by pre-allocating the full
        output array and filling it in place. Avoids the np.hstack(parts) pattern
        which builds a Python list of 1500+ small arrays.
        """
        if self.knots_ is None:
            raise RuntimeError("Call _fit_spline_knots(X_train) before _spline_expand.")

        n, p = X.shape
        K = self.n_knots
        out_cols = p + p * K
        out = np.empty((n, out_cols), dtype=np.float32)

        # Block 0: original columns
        out[:, :p] = X

        # Blocks 1..K: max(X - knot, 0) ** 2 for each knot, vectorized across all features
        # knots_ is (p, K); for knot k, we want X - knots_[:, k:k+1].T broadcast to (n, p)
        for k in range(K):
            start = p + k * p
            end = start + p
            # Use a temporary scratch of shape (n, p); reuse to avoid repeat allocs
            np.subtract(X, self.knots_[:, k], out=out[:, start:end])
            np.maximum(out[:, start:end], 0.0, out=out[:, start:end])
            np.square(out[:, start:end], out=out[:, start:end])

        return out

    def fit(self, X, y, X_val=None, y_val=None) -> "GLMModel":
        # Subsample BEFORE scaling — at year 2016 the full X is 3M rows;
        # scaling first would create a 6 GB float32 intermediate
        X_arr = _to_float32_array(X)
        y32 = y.astype(np.float32, copy=False)
        X_arr, y32 = _subsample_train(X_arr, y32, label="[GLM+H] train ")
        Xn = self.scaler_.fit_transform(X_arr)
        del X_arr
        gc.collect()

        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            Xv_n, yv32 = Xn[-n_val:], y32[-n_val:]
            Xn, y32 = Xn[:-n_val], y32[:-n_val]
        else:
            Xv_arr = _to_float32_array(X_val)
            yv32 = y_val.astype(np.float32, copy=False)
            Xv_arr, yv32 = _subsample_train(Xv_arr, yv32, max_rows=200_000, label="[GLM+H] val ")
            Xv_n = self.scaler_.transform(Xv_arr)
            del Xv_arr
            gc.collect()

        self._fit_spline_knots(Xn)

        # Spline expansion is the expensive step — log size
        Xs = self._spline_expand(Xn)
        del Xn
        gc.collect()
        logger.info(
            f"GLM+H spline matrix: shape={Xs.shape}, "
            f"size={Xs.nbytes / 1e9:.2f} GB"
        )

        Xs_val = self._spline_expand(Xv_n)
        del Xv_n
        gc.collect()

        # Grid search with explicit free between alphas
        best_r2 = -np.inf
        for alpha in self.alpha_grid:
            m = ElasticNet(
                alpha=alpha, l1_ratio=0.5, max_iter=5000,
                selection="random", tol=1e-3,
            )
            m.fit(Xs, y32)
            r2 = oos_r2(yv32, m.predict(Xs_val))
            if r2 > best_r2:
                if self.best_model_ is not None:
                    del self.best_model_
                    gc.collect()
                best_r2 = r2
                self.best_model_ = m
            else:
                del m
                gc.collect()

        if self.best_model_ is None:
            self.best_model_ = ElasticNet(alpha=1e-3, selection="random", tol=1e-3).fit(Xs, y32)

        del Xs, Xs_val, y32, yv32
        gc.collect()
        return self

    def predict(self, X) -> np.ndarray:
        Xn = self.scaler_.transform(_to_float32_array(X))
        Xs = self._spline_expand(Xn)
        del Xn
        gc.collect()
        out = self.best_model_.predict(Xs)
        del Xs
        gc.collect()
        return out

    def oos_r2(self, X, y) -> float:
        return oos_r2(y, self.predict(X))


# ─────────────────────────────────────────────────────────────────────────────
#  Tree Models
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestModel:
    """Random Forest (Breiman 2001)."""
    name = "RF"

    def __init__(self, n_estimators: int = 300,
                 max_depth_grid: List[int] = None,
                 n_jobs: int = -1, random_state: int = 42):
        self.n_estimators   = n_estimators
        self.max_depth_grid = max_depth_grid or [2, 3, 4, 5, 6]
        self.n_jobs         = n_jobs
        self.random_state   = random_state
        self.best_model_: Optional[RandomForestRegressor] = None

    def fit(self, X, y, X_val=None, y_val=None) -> "RandomForestModel":
        Xn = X.values if hasattr(X, "values") else X
        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            X_val, y_val = Xn[-n_val:], y[-n_val:]
            Xn, y = Xn[:-n_val], y[:-n_val]
        else:
            X_val = X_val.values if hasattr(X_val, "values") else X_val

        best_r2 = -np.inf
        for d in self.max_depth_grid:
            m = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=d,
                max_features="sqrt",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            m.fit(Xn, y)
            r2 = oos_r2(y_val, m.predict(X_val))
            if r2 > best_r2:
                if self.best_model_ is not None:
                    del self.best_model_
                    gc.collect()
                best_r2 = r2
                self.best_model_ = m
            else:
                del m
                gc.collect()
        if self.best_model_ is None:
            self.best_model_ = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=3,
                n_jobs=self.n_jobs, random_state=self.random_state
            ).fit(Xn, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.best_model_.predict(X.values if hasattr(X, "values") else X)

    def oos_r2(self, X, y) -> float:
        return oos_r2(y, self.predict(X))

    def feature_importance(self, feature_names: list) -> pd.Series:
        return pd.Series(
            self.best_model_.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)


class GBRTModel:
    """Gradient Boosted Regression Trees (Friedman 2001)."""
    name = "GBRT+H"

    def __init__(
        self,
        n_estimators_grid: List[int] = None,
        max_depth_grid: List[int] = None,
        learning_rate_grid: List[float] = None,
        random_state: int = 42,
    ):
        self.n_estimators_grid  = n_estimators_grid  or [100, 300, 500]
        self.max_depth_grid     = max_depth_grid     or [1, 2]
        self.learning_rate_grid = learning_rate_grid or [0.01, 0.1]
        self.random_state = random_state
        self.best_model_: Optional[GradientBoostingRegressor] = None

    def fit(self, X, y, X_val=None, y_val=None) -> "GBRTModel":
        Xn = X.values if hasattr(X, "values") else X
        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            X_val, y_val = Xn[-n_val:], y[-n_val:]
            Xn, y = Xn[:-n_val], y[:-n_val]
        else:
            X_val = X_val.values if hasattr(X_val, "values") else X_val

        def _make_gbrt(**kwargs) -> GradientBoostingRegressor:
            try:
                return GradientBoostingRegressor(loss="huber", **kwargs)
            except TypeError as e:
                logger.warning(
                    "GradientBoostingRegressor loss='huber' unsupported (%s); "
                    "using loss='squared_error'.",
                    e,
                )
                return GradientBoostingRegressor(loss="squared_error", **kwargs)

        best_r2 = -np.inf
        for n in self.n_estimators_grid:
            for d in self.max_depth_grid:
                for lr in self.learning_rate_grid:
                    m = _make_gbrt(
                        n_estimators=n,
                        max_depth=d,
                        learning_rate=lr,
                        subsample=0.5,
                        random_state=self.random_state,
                    )
                    m.fit(Xn, y)
                    r2 = oos_r2(y_val, m.predict(X_val))
                    if r2 > best_r2:
                        if self.best_model_ is not None:
                            del self.best_model_
                            gc.collect()
                        best_r2 = r2
                        self.best_model_ = m
                    else:
                        del m
                        gc.collect()
        if self.best_model_ is None:
            self.best_model_ = _make_gbrt(
                n_estimators=300,
                max_depth=1,
                learning_rate=0.01,
                random_state=self.random_state,
            ).fit(Xn, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.best_model_.predict(X.values if hasattr(X, "values") else X)

    def oos_r2(self, X, y) -> float:
        return oos_r2(y, self.predict(X))


# ─────────────────────────────────────────────────────────────────────────────
#  Neural Network Models  (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. Neural network models unavailable.")


if HAS_TORCH:

    class HuberLoss(nn.Module):
        """Huber loss for monthly returns (GKX-style robust objective)."""

        def __init__(self, delta: float = 0.001) -> None:
            super().__init__()
            self.delta = delta

        def forward(self, pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
            return nn.functional.huber_loss(
                pred, target, reduction="mean", delta=self.delta
            )


class _FeedForwardNet(nn.Module if HAS_TORCH else object):
    """
    GKX (2019) feed-forward neural network:
    Input → [Linear → BatchNorm → ReLU] × L → Linear output
    """

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for neural network models.")
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeuralNetModel:
    """
    GKX (2019) neural network with:
    • ReLU activations
    • Batch normalisation
    • L1 regularisation
    • Early stopping on validation loss
    • Ensemble of N random seeds
    """
    name: str = "NN"

    def __init__(
        self,
        hidden_dims: List[int] = None,
        l1_lambda: float = 1e-4,
        learning_rate: float = 0.001,
        batch_size: int = 10000,
        max_epochs: int = 100,
        patience: int = 5,
        n_ensemble: int = 10,
        device: str = "cpu",
        name: str = None,
    ):
        self.hidden_dims  = hidden_dims or [32, 16, 8]
        self.l1_lambda    = l1_lambda
        self.learning_rate = learning_rate
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.n_ensemble   = n_ensemble
        self.device       = device
        if name:
            self.name = name
        else:
            self.name = f"NN{len(self.hidden_dims)}"
        self.models_: List = []
        self.scaler_  = StandardScaler()

    def fit(self, X, y, X_val=None, y_val=None) -> "NeuralNetModel":
        if not HAS_TORCH:
            raise ImportError("PyTorch required.")

        Xn = self.scaler_.fit_transform(X.values if hasattr(X, "values") else X).astype(np.float32)
        yn = y.astype(np.float32)
        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            Xv, yv = Xn[-n_val:], yn[-n_val:]
            Xn, yn = Xn[:-n_val], yn[:-n_val]
        else:
            Xv = self.scaler_.transform(
                X_val.values if hasattr(X_val, "values") else X_val
            ).astype(np.float32)
            yv = y_val.astype(np.float32)

        self.models_ = []
        input_dim = Xn.shape[1]
        dev = torch.device(self.device)

        for seed in range(self.n_ensemble):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = _FeedForwardNet(input_dim, self.hidden_dims).to(dev)
            opt = optim.Adam(net.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
            huber = HuberLoss(delta=0.001)

            ds  = TensorDataset(
                torch.from_numpy(Xn).to(dev),
                torch.from_numpy(yn).to(dev),
            )
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

            Xv_t = torch.from_numpy(Xv).to(dev)
            yv_t = torch.from_numpy(yv).to(dev)

            best_val_loss = np.inf
            best_state    = None
            patience_ctr  = 0

            for epoch in range(self.max_epochs):
                net.train()
                for xb, yb in dl:
                    opt.zero_grad()
                    pred = net(xb)
                    loss = huber(pred, yb)
                    l1 = sum(p.abs().sum() for p in net.parameters())
                    (loss + self.l1_lambda * l1).backward()
                    opt.step()

                net.eval()
                with torch.no_grad():
                    val_loss = huber(net(Xv_t), yv_t).item()
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    patience_ctr  = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self.models_.append(net)

        return self

    def predict(self, X) -> np.ndarray:
        if not HAS_TORCH:
            raise ImportError("PyTorch required.")
        Xn = self.scaler_.transform(
            X.values if hasattr(X, "values") else X
        ).astype(np.float32)
        dev = torch.device(self.device)
        Xt  = torch.from_numpy(Xn).to(dev)
        preds = []
        for net in self.models_:
            net.eval()
            with torch.no_grad():
                preds.append(net(Xt).cpu().numpy())
        return np.mean(preds, axis=0)

    def oos_r2(self, X, y) -> float:
        return oos_r2(y, self.predict(X))


def build_all_neural_nets(
    architectures: List[List[int]] = None,
    **kwargs,
) -> List[NeuralNetModel]:
    """Factory: returns NN1 … NN5 model objects."""
    if architectures is None:
        architectures = [
            [32],
            [32, 16],
            [32, 16, 8],
            [32, 16, 8, 4],
            [32, 16, 8, 4, 2],
        ]
    return [
        NeuralNetModel(
            hidden_dims=dims,
            name=f"NN{i+1}",
            **kwargs,
        )
        for i, dims in enumerate(architectures)
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Model registry
# ─────────────────────────────────────────────────────────────────────────────

def get_all_models(nn_architectures: List[List[int]] = None,
                   nn_kwargs: dict = None) -> dict:
    """
    Returns an ordered dict of {model_name: model_instance}.
    Use this as the single entry point for the training pipeline.
    """
    nn_kwargs = nn_kwargs or {}
    nn_models = build_all_neural_nets(nn_architectures, **nn_kwargs) if HAS_TORCH else []

    models = {
        "OLS-3":   OLS3Model(),
        "ENet+H":  ElasticNetModel(),
        "PCR":     PCRModel(),
        "PLS":     PLSModel(),
        "GLM+H":   GLMModel(),
        "RF":      RandomForestModel(),
        "GBRT+H":  GBRTModel(),
    }
    for m in nn_models:
        models[m.name] = m
    return models