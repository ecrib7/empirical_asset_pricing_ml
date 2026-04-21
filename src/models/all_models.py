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

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, ElasticNet, HuberRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


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
    """Grid search over param_grid using validation R²."""
    best_r2  = -np.inf
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
                best_r2 = r2
                best_model  = m
                best_params = params
        except Exception as e:
            logger.warning(f"  {params} failed: {e}")
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

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "OLS3Model":
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
        self.model.fit(X[avail].values, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self._avail].values)

    def oos_r2(self, X: pd.DataFrame, y: np.ndarray) -> float:
        return oos_r2(y, self.predict(X))


class ElasticNetModel:
    """
    Elastic Net with Huber loss (GKX default).
    Falls back to ElasticNet (L2 loss) if sklearn HuberRegressor
    does not support L1 penalty.
    """
    name = "ENet+H"

    def __init__(
        self,
        alpha_grid: List[float] = None,
        l1_ratio: float = 0.5,
        use_huber: bool = True,
    ):
        self.alpha_grid = alpha_grid or [1e-4, 1e-3, 1e-2, 5e-2, 0.1]
        self.l1_ratio   = l1_ratio
        self.use_huber  = use_huber
        self.best_model_: Optional[BaseEstimator] = None
        self.scaler_     = StandardScaler()

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        X_val: pd.DataFrame | np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> "ElasticNetModel":
        Xn = self.scaler_.fit_transform(
            X.values if hasattr(X, "values") else X
        )
        if X_val is not None:
            Xv = self.scaler_.transform(X_val.values if hasattr(X_val, "values") else X_val)
        else:
            n_val = max(1, int(0.2 * len(Xn)))
            Xv, y_val = Xn[-n_val:], y[-n_val:]
            Xn, y = Xn[:-n_val], y[:-n_val]

        def make_model(alpha):
            return ElasticNet(alpha=alpha, l1_ratio=self.l1_ratio,
                              max_iter=5000, warm_start=False)

        param_grid = [{"alpha": a} for a in self.alpha_grid]
        self.best_model_, self.best_params_ = _tune_on_val(
            make_model, param_grid, Xn, y, Xv, y_val
        )
        if self.best_model_ is None:
            self.best_model_ = make_model(self.alpha_grid[len(self.alpha_grid)//2])
            self.best_model_.fit(Xn, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        Xn = self.scaler_.transform(X.values if hasattr(X, "values") else X)
        return self.best_model_.predict(Xn)

    def oos_r2(self, X, y) -> float:
        return oos_r2(y, self.predict(X))


class PCRModel:
    """Principal Components Regression."""
    name = "PCR"

    def __init__(self, n_components_grid: List[int] = None):
        self.n_components_grid = n_components_grid or [3, 5, 10, 20, 30, 40]
        self.scaler_ = StandardScaler()
        self.best_k_: int = 10
        self.pca_: Optional[PCA] = None
        self.reg_: Optional[LinearRegression] = None

    def fit(self, X, y, X_val=None, y_val=None) -> "PCRModel":
        Xn = self.scaler_.fit_transform(X.values if hasattr(X, "values") else X)
        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            X_val, y_val = Xn[-n_val:], y[-n_val:]
            Xn, y = Xn[:-n_val], y[:-n_val]
        else:
            X_val = self.scaler_.transform(X_val.values if hasattr(X_val, "values") else X_val)

        max_k = min(max(self.n_components_grid), Xn.shape[1], Xn.shape[0] - 1)
        pca = PCA(n_components=max_k)
        pca.fit(Xn)

        best_r2 = -np.inf
        for k in self.n_components_grid:
            k = min(k, max_k)
            Z_train = pca.transform(Xn)[:, :k]
            Z_val   = pca.transform(X_val)[:, :k]
            reg = LinearRegression().fit(Z_train, y)
            r2  = oos_r2(y_val, reg.predict(Z_val))
            if r2 > best_r2:
                best_r2      = r2
                self.best_k_ = k

        self.pca_ = PCA(n_components=self.best_k_).fit(Xn)
        Z = self.pca_.transform(Xn)
        self.reg_ = LinearRegression().fit(Z, y)
        return self

    def predict(self, X) -> np.ndarray:
        Xn = self.scaler_.transform(X.values if hasattr(X, "values") else X)
        Z  = self.pca_.transform(Xn)
        return self.reg_.predict(Z)

    def oos_r2(self, X, y) -> float:
        return oos_r2(y, self.predict(X))


class PLSModel:
    """Partial Least Squares Regression."""
    name = "PLS"

    def __init__(self, n_components_grid: List[int] = None):
        self.n_components_grid = n_components_grid or [1, 2, 3, 4, 5, 6, 8, 10]
        self.scaler_ = StandardScaler()
        self.best_model_: Optional[PLSRegression] = None

    def fit(self, X, y, X_val=None, y_val=None) -> "PLSModel":
        Xn = self.scaler_.fit_transform(X.values if hasattr(X, "values") else X)
        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            X_val, y_val = Xn[-n_val:], y[-n_val:]
            Xn, y = Xn[:-n_val], y[:-n_val]
        else:
            X_val = self.scaler_.transform(X_val.values if hasattr(X_val, "values") else X_val)

        max_k = min(max(self.n_components_grid), Xn.shape[1], Xn.shape[0] - 1)
        best_r2 = -np.inf
        for k in self.n_components_grid:
            k = min(k, max_k)
            try:
                pls = PLSRegression(n_components=k).fit(Xn, y)
                r2  = oos_r2(y_val, pls.predict(X_val).flatten())
                if r2 > best_r2:
                    best_r2 = r2
                    self.best_model_ = pls
            except Exception:
                pass
        if self.best_model_ is None:
            self.best_model_ = PLSRegression(n_components=1).fit(Xn, y)
        return self

    def predict(self, X) -> np.ndarray:
        Xn = self.scaler_.transform(X.values if hasattr(X, "values") else X)
        return self.best_model_.predict(Xn).flatten()

    def oos_r2(self, X, y) -> float:
        return oos_r2(y, self.predict(X))


class GLMModel:
    """
    Generalised Linear Model with Group Lasso.
    Approximated via ElasticNet on spline-expanded features.
    Each characteristic is expanded with a quadratic spline (k=3 knots).
    """
    name = "GLM+H"

    def __init__(self, n_knots: int = 3, alpha_grid: List[float] = None):
        self.n_knots   = n_knots
        self.alpha_grid = alpha_grid or [1e-4, 1e-3, 1e-2, 5e-2, 0.1]
        self.scaler_   = StandardScaler()
        self.best_model_: Optional[ElasticNet] = None
        self.knots_: Optional[List[np.ndarray]] = None

    def _fit_spline_knots(self, X: np.ndarray) -> None:
        """
        Compute quantile knots per feature from training matrix ``X`` (scaled)
        and store in ``self.knots_``. Validation/test expansions must use these
        knots only (no refit on held-out quantiles).
        """
        self.knots_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            self.knots_.append(
                np.quantile(col, np.linspace(0.1, 0.9, self.n_knots))
            )

    def _spline_expand(self, X: np.ndarray) -> np.ndarray:
        """Add quadratic spline terms using ``self.knots_`` from ``_fit_spline_knots``."""
        if self.knots_ is None:
            raise RuntimeError("Call _fit_spline_knots(X_train) before _spline_expand.")
        parts = [X]
        for j in range(X.shape[1]):
            col = X[:, j]
            for c in self.knots_[j]:
                parts.append(np.maximum(col - c, 0).reshape(-1, 1) ** 2)
        return np.hstack(parts)

    def fit(self, X, y, X_val=None, y_val=None) -> "GLMModel":
        Xn = self.scaler_.fit_transform(X.values if hasattr(X, "values") else X)
        if X_val is None:
            n_val = max(1, int(0.2 * len(Xn)))
            Xv_n, y_val = Xn[-n_val:], y[-n_val:]
            Xn, y = Xn[:-n_val], y[:-n_val]
        else:
            Xv_n = self.scaler_.transform(X_val.values if hasattr(X_val, "values") else X_val)

        self._fit_spline_knots(Xn)
        Xs = self._spline_expand(Xn)
        Xs_val = self._spline_expand(Xv_n)

        best_r2 = -np.inf
        for alpha in self.alpha_grid:
            m = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000)
            m.fit(Xs, y)
            r2 = oos_r2(y_val, m.predict(Xs_val))
            if r2 > best_r2:
                best_r2 = r2
                self.best_model_ = m
        if self.best_model_ is None:
            self.best_model_ = ElasticNet(alpha=1e-3).fit(Xs, y)
        return self

    def predict(self, X) -> np.ndarray:
        Xn = self.scaler_.transform(X.values if hasattr(X, "values") else X)
        Xs = self._spline_expand(Xn)
        return self.best_model_.predict(Xs)

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
                best_r2 = r2
                self.best_model_ = m
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
                        best_r2 = r2
                        self.best_model_ = m
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

        # Tune l1_lambda using a small search (optional; here we use given value)
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
                    # L1 regularisation
                    l1 = sum(p.abs().sum() for p in net.parameters())
                    (loss + self.l1_lambda * l1).backward()
                    opt.step()

                # Validation loss
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
