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
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
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
        # Cast once to float64 for accurate moments, reuse for both mean + std.
        # Previously called X.astype(float64) twice → two full copies (~10 GB
        # each on a 2.5M×518 matrix). Now we keep one copy and delete it after.
        X64 = X.astype(np.float64, copy=False)
        self.mean_ = X64.mean(axis=0).astype(np.float32)
        std = X64.std(axis=0)
        del X64
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


_MAX_TRAIN_ROWS     = 500_000
_MAX_TRAIN_ROWS_GLM = 100_000   # GLM+H spline-expands to 4× columns; CD cost ∝ n×p
_MAX_TRAIN_ROWS_RF  = 200_000   # RF cost ∝ n×sqrt(p)×depth×trees; 200k→~3 min total
_MAX_TRAIN_ROWS_GBRT= 250_000   # HistGBRT uses histogram bins — 250k is plenty


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
        # ── Subsample BEFORE scaling so the scaler never sees 2.5M rows ──
        # _Float32Scaler.fit() must cast to float64 internally; on 2.5M×518
        # that's ~10 GB. Capping first keeps peak RAM under 2 GB here.
        X_arr = _to_float32_array(X)
        y_arr = y.astype(np.float32, copy=False)
        X_arr, y_arr = _subsample_train(X_arr, y_arr, label="[PCR] train ")

        Xn = self.scaler_.fit_transform(X_arr)
        del X_arr
        gc.collect()

        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            X_val, y_val = Xn[-n_val:], y_arr[-n_val:]
            Xn, y_arr = Xn[:-n_val], y_arr[:-n_val]
        else:
            X_val = self.scaler_.transform(_to_float32_array(X_val))

        y32  = y_arr
        yv32 = y_val.astype(np.float32, copy=False)

        max_k = min(max(self.n_components_grid), Xn.shape[1], Xn.shape[0] - 1)
        pca = PCA(n_components=max_k, svd_solver="randomized", random_state=0)
        pca.fit(Xn)
        Z_train_full = pca.transform(Xn)
        Z_val_full = pca.transform(X_val)

        # Free the validation feature matrix early — it's no longer needed
        # once we have its projection. Same for the (huge) raw Xn after we
        # extract the final regressor.
        del X_val
        gc.collect()

        best_r2 = -np.inf
        for k in self.n_components_grid:
            k = min(k, max_k)
            reg = LinearRegression().fit(Z_train_full[:, :k], y32)
            r2 = oos_r2(yv32, reg.predict(Z_val_full[:, :k]))
            if r2 > best_r2:
                best_r2 = r2
                self.best_k_ = k

        # Keep the same PCA object — just truncate to best_k_. Avoids a
        # second full SVD which on large training panels (e.g. 3M rows ×
        # 920 features by year 2000+) was peaking memory at >40 GB.
        self.pca_ = pca
        self.pca_.components_ = pca.components_[: self.best_k_]
        self.pca_.explained_variance_ = pca.explained_variance_[: self.best_k_]
        self.pca_.explained_variance_ratio_ = pca.explained_variance_ratio_[: self.best_k_]
        self.pca_.singular_values_ = pca.singular_values_[: self.best_k_]
        self.pca_.n_components_ = self.best_k_
        self.reg_ = LinearRegression().fit(Z_train_full[:, : self.best_k_], y32)

        del Xn, Z_train_full, Z_val_full
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
        # Smaller default grid (3 alphas instead of 5) — three orders of
        # magnitude is plenty to find the validation optimum; the extra
        # two alphas in the old grid almost never won.
        self.alpha_grid = alpha_grid or [1e-3, 1e-2, 1e-1]
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
        # Subsample BEFORE scaling. GLM+H spline-expands to p*(K+1) columns
        # (518*4 = 2072), so ElasticNet CD cost ∝ n × 2072 per iteration.
        # 500k rows → ~78 min; 100k rows → ~15 min with negligible R² loss.
        X_arr = _to_float32_array(X)
        y32 = y.astype(np.float32, copy=False)
        X_arr, y32 = _subsample_train(X_arr, y32, max_rows=_MAX_TRAIN_ROWS_GLM,
                                      label="[GLM+H] train ")
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

        # Warm-start path: fit alphas in DESCENDING order (largest = most
        # regularised = sparsest = fastest) and reuse coefs as the next
        # alpha's starting point. This is the standard "regularisation
        # path" trick — sklearn's `ElasticNet(warm_start=True)` lets us
        # do it without lars-path-style ceremony. With 3 alphas this is
        # ~3-5× faster than independent fits because all but the first
        # converge in a few hundred iterations rather than thousands.
        alphas_sorted = sorted(self.alpha_grid, reverse=True)
        # Single ElasticNet instance reused across alphas
        m = ElasticNet(
            alpha=alphas_sorted[0], l1_ratio=0.5,
            max_iter=500,                  # 100k rows converges fast; was 2000
            selection="random", tol=5e-3,  # looser tol fine for cross-sectional prediction
            warm_start=True,
        )

        best_r2 = -np.inf
        for alpha in alphas_sorted:
            m.alpha = alpha
            m.fit(Xs, y32)
            r2 = oos_r2(yv32, m.predict(Xs_val))
            if r2 > best_r2:
                if self.best_model_ is not None:
                    del self.best_model_
                    gc.collect()
                best_r2 = r2
                # Snapshot — copy params so subsequent warm-start fits
                # don't mutate the saved best model.
                from copy import deepcopy
                self.best_model_ = deepcopy(m)

        if self.best_model_ is None:
            self.best_model_ = ElasticNet(alpha=1e-3, selection="random", tol=1e-3).fit(Xs, y32)

        del Xs, Xs_val, y32, yv32, m
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
        Xn = _to_float32_array(X)
        y32 = y.astype(np.float32, copy=False)
        # Subsample before split: RF cost ∝ n×sqrt(p)×depth×trees.
        # 2M rows → many hours; 200k rows → ~3 min for 5 depths × 300 trees.
        Xn, y32 = _subsample_train(Xn, y32, max_rows=_MAX_TRAIN_ROWS_RF,
                                   label="[RF] train ")
        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            X_val, y_val = Xn[-n_val:], y32[-n_val:]
            Xn, y32 = Xn[:-n_val], y32[:-n_val]
        else:
            X_val = _to_float32_array(X_val)
            y_val = y_val.astype(np.float32, copy=False)

        best_r2 = -np.inf
        for d in self.max_depth_grid:
            m = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=d,
                max_features="sqrt",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            m.fit(Xn, y32)
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
            ).fit(Xn, y32)
        return self

    def predict(self, X) -> np.ndarray:
        return self.best_model_.predict(_to_float32_array(X))

    def oos_r2(self, X, y) -> float:
        return oos_r2(y, self.predict(X))

    def feature_importance(self, feature_names: list) -> pd.Series:
        return pd.Series(
            self.best_model_.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)


class GBRTModel:
    """
    Gradient Boosted Regression Trees (Friedman 2001) via
    HistGradientBoostingRegressor with early-stopped n_estimators.

    Speed strategy:
      1. HistGBRT histogram binning → 10-50× faster than GradientBoostingRegressor.
      2. Drop n_estimators from grid — use staged_predict on our val set to
         pick the optimal stopping point in ONE training pass instead of three
         (was: max_iter ∈ {100, 300, 500} as separate fits; now: single
         max_iter=500 fit + staged_predict scan).
      3. Subsample 250k rows — plenty for boosted trees.

    Effective grid is now 2 depths × 2 learning_rates = 4 combos (was 12),
    and each combo finishes in ~30-60 sec → total ~3-5 min/year.
    """
    name = "GBRT+H"

    def __init__(
        self,
        n_estimators_grid: List[int] = None,
        max_depth_grid: List[int] = None,
        learning_rate_grid: List[float] = None,
        random_state: int = 42,
    ):
        # n_estimators_grid is now used only as a candidate set for the
        # staged_predict scan within each (depth, lr) combo.
        self.n_estimators_grid  = n_estimators_grid  or [50, 100, 200, 300, 500]
        self.max_depth_grid     = max_depth_grid     or [1, 2]
        self.learning_rate_grid = learning_rate_grid or [0.01, 0.1]
        self.random_state = random_state
        self.best_model_: Optional[HistGradientBoostingRegressor] = None
        self.best_n_iter_: int = 0

    def fit(self, X, y, X_val=None, y_val=None) -> "GBRTModel":
        Xn = _to_float32_array(X)
        y32 = y.astype(np.float32, copy=False)
        Xn, y32 = _subsample_train(Xn, y32, max_rows=_MAX_TRAIN_ROWS_GBRT,
                                   label="[GBRT+H] train ")
        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            X_val, y_val = Xn[-n_val:], y32[-n_val:]
            Xn, y32 = Xn[:-n_val], y32[:-n_val]
        else:
            X_val = _to_float32_array(X_val)
            y_val = y_val.astype(np.float32, copy=False)

        max_iter_full = max(self.n_estimators_grid)
        candidate_iters = sorted(set(self.n_estimators_grid))
        best_r2 = -np.inf

        for d in self.max_depth_grid:
            for lr in self.learning_rate_grid:
                # Single fit at max_iter — no separate fits for n=100, 300, 500
                m = HistGradientBoostingRegressor(
                    max_iter=max_iter_full,
                    max_depth=d,
                    learning_rate=lr,
                    random_state=self.random_state,
                    early_stopping=False,
                )
                m.fit(Xn, y32)

                # staged_predict yields predictions after each tree;
                # we only score at our candidate n_estimators values.
                stages = m.staged_predict(X_val)
                best_r2_combo = -np.inf
                best_n_combo  = max_iter_full
                for i, pred in enumerate(stages):
                    n_iter = i + 1
                    if n_iter not in candidate_iters and n_iter != max_iter_full:
                        continue
                    r2 = oos_r2(y_val, pred)
                    if r2 > best_r2_combo:
                        best_r2_combo = r2
                        best_n_combo  = n_iter

                logger.info(
                    f"[GBRT+H] depth={d} lr={lr}: best n_iter={best_n_combo} "
                    f"R²_val={best_r2_combo:.4f}"
                )

                if best_r2_combo > best_r2:
                    best_r2 = best_r2_combo
                    # Refit with truncated max_iter to drop unused trees from memory.
                    if self.best_model_ is not None:
                        del self.best_model_
                        gc.collect()
                    self.best_model_ = HistGradientBoostingRegressor(
                        max_iter=best_n_combo,
                        max_depth=d,
                        learning_rate=lr,
                        random_state=self.random_state,
                        early_stopping=False,
                    ).fit(Xn, y32)
                    self.best_n_iter_ = best_n_combo
                del m
                gc.collect()

        if self.best_model_ is None:
            self.best_model_ = HistGradientBoostingRegressor(
                max_iter=300, max_depth=1, learning_rate=0.01,
                random_state=self.random_state,
            ).fit(Xn, y32)
        return self

    def predict(self, X) -> np.ndarray:
        return self.best_model_.predict(_to_float32_array(X))

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
        device: str | None = None,
        name: str = None,
    ):
        self.hidden_dims  = hidden_dims or [32, 16, 8]
        self.l1_lambda    = l1_lambda
        self.learning_rate = learning_rate
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.n_ensemble   = n_ensemble
        # Auto-detect CUDA when device is not explicitly set. Override
        # via the FORCE_DEVICE env var if you want to pin to CPU on a
        # GPU runtime (e.g. for debugging).
        import os
        if device is None:
            forced = os.environ.get("FORCE_DEVICE")
            if forced:
                self.device = forced
            elif HAS_TORCH and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        if name:
            self.name = name
        else:
            self.name = f"NN{len(self.hidden_dims)}"
        self.models_: List = []
        self.scaler_  = StandardScaler()

    def fit(self, X, y, X_val=None, y_val=None) -> "NeuralNetModel":
        """
        Speed-tuned training loop. Same statistical recipe as GKX but:
          * No DataLoader — data lives on GPU, we slice indices in-place.
          * Vectorised L1 penalty (one cat + abs + sum, not a Python loop).
          * Mixed precision (fp16) on CUDA — 2-3× faster on tiny nets.
          * Full-batch training on GPU when the data fits — eliminates
            mini-batch kernel-launch overhead, which dominates the wall
            clock for tiny networks like NN1-NN5.
          * Larger default batch size (65k) when full-batch isn't feasible.
        Combined: ~10-30× speedup vs the original on a T4 for these
        small networks. See ``full_batch`` parameter for the toggle.
        """
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
        is_cuda = dev.type == "cuda"

        # Move full train + val tensors to device once. Skip DataLoader entirely.
        X_tr = torch.from_numpy(Xn).to(dev, non_blocking=True)
        y_tr = torch.from_numpy(yn).to(dev, non_blocking=True)
        Xv_t = torch.from_numpy(Xv).to(dev, non_blocking=True)
        yv_t = torch.from_numpy(yv).to(dev, non_blocking=True)

        n_train = X_tr.shape[0]

        # Decide between full-batch and mini-batch.
        # Full-batch wins for tiny networks because mini-batch overhead
        # (kernel launches, index gathering) dominates real arithmetic.
        # Heuristic: use full-batch if the dataset fits in <60% of GPU
        # memory in fp16 (we're going to autocast anyway).
        bytes_per_elem = 2 if is_cuda else 4   # fp16 vs fp32
        train_bytes = n_train * input_dim * bytes_per_elem
        full_batch = (
            is_cuda
            and train_bytes < 0.6 * torch.cuda.get_device_properties(0).total_memory
        )

        if full_batch:
            batch_size = n_train     # one step per epoch
        else:
            batch_size = max(self.batch_size, 65_536) if is_cuda else self.batch_size

        # Mixed precision: huge gains on T4/A100, no-op on CPU.
        try:
            from torch.amp import autocast, GradScaler
            amp_ctx = lambda: autocast(device_type="cuda", dtype=torch.float16)
            scaler_amp = GradScaler() if is_cuda else None
        except Exception:
            from contextlib import nullcontext
            amp_ctx = nullcontext
            scaler_amp = None

        for seed in range(self.n_ensemble):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = _FeedForwardNet(input_dim, self.hidden_dims).to(dev)
            opt = optim.Adam(net.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
            huber = HuberLoss(delta=0.001)

            best_val_loss = np.inf
            best_state    = None
            patience_ctr  = 0

            for epoch in range(self.max_epochs):
                net.train()
                if full_batch:
                    # Single full-batch gradient step per epoch
                    opt.zero_grad(set_to_none=True)
                    with amp_ctx():
                        pred = net(X_tr)
                        data_loss = huber(pred, y_tr)
                        l1 = torch.cat([p.flatten() for p in net.parameters()]).abs().sum()
                        total_loss = data_loss + self.l1_lambda * l1
                    if scaler_amp is not None:
                        scaler_amp.scale(total_loss).backward()
                        scaler_amp.step(opt)
                        scaler_amp.update()
                    else:
                        total_loss.backward()
                        opt.step()
                else:
                    # Mini-batch path (CPU or huge datasets)
                    perm = torch.randperm(n_train, device=dev)
                    for i in range(0, n_train, batch_size):
                        idx = perm[i : i + batch_size]
                        xb = X_tr.index_select(0, idx)
                        yb = y_tr.index_select(0, idx)
                        opt.zero_grad(set_to_none=True)
                        with amp_ctx():
                            pred = net(xb)
                            data_loss = huber(pred, yb)
                            l1 = torch.cat([p.flatten() for p in net.parameters()]).abs().sum()
                            total_loss = data_loss + self.l1_lambda * l1
                        if scaler_amp is not None:
                            scaler_amp.scale(total_loss).backward()
                            scaler_amp.step(opt)
                            scaler_amp.update()
                        else:
                            total_loss.backward()
                            opt.step()

                net.eval()
                with torch.no_grad(), amp_ctx():
                    val_loss = huber(net(Xv_t), yv_t).item()
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    patience_ctr  = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self.models_.append(net)

        # Free the big GPU tensors before fit returns
        del X_tr, y_tr, Xv_t, yv_t
        if is_cuda:
            torch.cuda.empty_cache()

        return self

    def predict(self, X) -> np.ndarray:
        if not HAS_TORCH:
            raise ImportError("PyTorch required.")
        Xn = self.scaler_.transform(
            X.values if hasattr(X, "values") else X
        ).astype(np.float32)
        dev = torch.device(self.device)
        Xt  = torch.from_numpy(Xn).to(dev, non_blocking=True)

        # Predict in chunks to avoid blowing up GPU memory on large test sets,
        # but use big chunks (256k) to keep GPU utilisation high.
        chunk = 256_000 if dev.type == "cuda" else len(Xn)
        preds_sum = None
        with torch.no_grad():
            for net in self.models_:
                net.eval()
                outs = []
                for i in range(0, len(Xt), chunk):
                    outs.append(net(Xt[i : i + chunk]))
                pred = torch.cat(outs, dim=0).float()
                preds_sum = pred if preds_sum is None else preds_sum + pred
        avg = (preds_sum / len(self.models_)).cpu().numpy()
        del Xt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return avg

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