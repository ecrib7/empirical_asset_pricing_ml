"""
Gradient boosted trees via LightGBM (``lightgbm.LGBMRegressor``).

Falls back conceptually to ``sklearn.ensemble.HistGradientBoostingRegressor`` if LightGBM
is unavailable in constrained environments.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.models.base import ModelBase

try:
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    lgb = None  # type: ignore


class GBRTModel(ModelBase):
    """Gradient boosted tree ensemble (LightGBM backend when installed)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "GBRTModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
