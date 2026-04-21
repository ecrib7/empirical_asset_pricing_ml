"""
Random Forest regressor wrapper (``sklearn.ensemble.RandomForestRegressor``) with hyper-parameter grids.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.models.base import ModelBase


class RandomForestModel(ModelBase):
    """Random forest with grids over ``max_features`` and ``n_estimators``."""

    def __init__(
        self,
        n_estimators_grid: Optional[List[int]] = None,
        max_features_grid: Optional[List[Any]] = None,
        random_state: int = 42,
    ) -> None:
        self.n_estimators_grid = n_estimators_grid or [100, 300]
        self.max_features_grid = max_features_grid or ["sqrt", 0.3]
        self.random_state = random_state
        self._model: Optional[RandomForestRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "RandomForestModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
