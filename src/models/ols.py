"""
OLS-3 benchmark: linear model with three Fama–French style factor exposures as features.

Intended for monthly cross-sectional or panel regressions aligned with GKX (2019).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.base import ModelBase


class OLS3Model(ModelBase):
    """
    Ordinary least squares using exactly three columns of ``X`` (e.g. MKT, SMB, HML).

    This is a stub; wire to ``statsmodels`` or ``sklearn.linear_model.LinearRegression``.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "OLS3Model":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
