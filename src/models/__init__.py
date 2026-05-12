"""
src.models
----------
Model interfaces and estimators for the GKX-style ML pipeline.

Legacy implementations live in ``all_models.py``; this package adds
modular stubs (``base``, ``ols``, …) for a cleaner trading-system layout.
"""

from src.models.base import ModelBase

__all__ = ["ModelBase"]
