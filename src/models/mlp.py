"""
Multi-layer perceptron regressor.

Configurable backend: ``sklearn.neural_network.MLPRegressor`` (default stub path) or PyTorch
(``torch.nn``) for larger architectures.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np

from src.models.base import ModelBase

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from sklearn.neural_network import MLPRegressor


class MLPModel(ModelBase):
    """Feed-forward network for return prediction."""

    def __init__(
        self,
        backend: Literal["sklearn", "torch"] = "sklearn",
        hidden_layer_sizes: tuple = (64, 32),
        max_iter: int = 500,
    ) -> None:
        self.backend = backend
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self._sk_model: Optional[MLPRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "MLPModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
