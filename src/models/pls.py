"""
Partial Least Squares regression (``sklearn.cross_decomposition.PLSRegression``).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from src.models.base import ModelBase


class PLSModel(ModelBase):
    """PLS dimension reduction + linear prediction on latent scores."""

    def __init__(self, n_components: int = 5, scale: bool = True) -> None:
        self.n_components = n_components
        self.scale = scale
        self._model: Optional[PLSRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "PLSModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
