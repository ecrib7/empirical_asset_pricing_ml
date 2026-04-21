"""
Abstract model interface for supervised return prediction.

All concrete models should subclass ``ModelBase`` and implement ``fit`` / ``predict``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class ModelBase(ABC):
    """
    Abstract base class for panel / cross-sectional return forecasters.

    Parameters and hyper-parameters should be returned via ``get_params``
    for logging and experiment tracking.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "ModelBase":
        """Fit the model to training design matrix ``X`` and target ``y``."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return point predictions for rows of ``X``."""
        raise NotImplementedError

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Hyper-parameters and options (sklearn-compatible signature)."""
        return {
            k: getattr(self, k)
            for k in self.__dict__
            if not k.startswith("_") and not callable(getattr(self, k, None))
        }
