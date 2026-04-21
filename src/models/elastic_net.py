"""
Elastic Net regression wrapper (``sklearn.linear_model.ElasticNet``) with alpha / l1_ratio grid.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from sklearn.linear_model import ElasticNet

from src.models.base import ModelBase


class ElasticNetModel(ModelBase):
    """
    Penalised linear regression with hyper-parameter search over ``alphas`` and ``l1_ratios``.
    """

    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        l1_ratios: Optional[List[float]] = None,
        max_iter: int = 2000,
    ) -> None:
        self.alphas = alphas or [1e-4, 1e-3, 1e-2]
        self.l1_ratios = l1_ratios or [0.1, 0.5, 0.9]
        self.max_iter = max_iter
        self._model: Optional[ElasticNet] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "ElasticNetModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
