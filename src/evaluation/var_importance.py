"""
evaluation/var_importance.py
----------------------------
Compute per-model variable importance for the GKX pipeline.

Two approaches are supported:

  1. ``zero_set_importance``  – GKX (2019) style: zero each feature in
     turn (cross-sectionally rank-normalised features have median 0), and
     measure the drop in OOS R². Strictly model-agnostic, just needs
     ``model.predict(X)``.

  2. ``permutation_importance`` – shuffle each feature column on the
     evaluation set and measure the drop in OOS R². Slightly noisier but
     does not assume features are normalised.

We also provide ``aggregate_to_base_chars`` which collapses
920 Kronecker-product features (e.g. ``mom1m_const``, ``mom1m_dp``,
``mom1m_ep``, …) back to the 94 underlying firm characteristics, by
summing the per-feature importance across the 9 macro blocks that share
the same base name.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import oos_r2

logger = logging.getLogger(__name__)


def zero_set_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: Optional[Iterable[str]] = None,
) -> pd.Series:
    """
    GKX (2019) variable importance: drop in OOS R² when the feature is
    set to its cross-sectional median (= 0 in normalised form).

    Returns a Series indexed by feature name, sorted descending.
    """
    feats = list(feature_names) if feature_names is not None else list(X.columns)
    base_pred = model.predict(X)
    base_r2 = oos_r2(y, base_pred)
    out = {}
    for col in feats:
        if col not in X.columns:
            continue
        original = X[col].copy()
        X[col] = 0.0
        try:
            r2_pert = oos_r2(y, model.predict(X))
        finally:
            X[col] = original
        out[col] = base_r2 - r2_pert
    return pd.Series(out, name="importance").sort_values(ascending=False)


def permutation_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: Optional[Iterable[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """
    Drop in OOS R² after shuffling each feature column.
    """
    feats = list(feature_names) if feature_names is not None else list(X.columns)
    rng = rng or np.random.default_rng(42)
    base_r2 = oos_r2(y, model.predict(X))
    out = {}
    for col in feats:
        if col not in X.columns:
            continue
        original = X[col].copy()
        perm = original.values.copy()
        rng.shuffle(perm)
        X[col] = perm
        try:
            r2_pert = oos_r2(y, model.predict(X))
        finally:
            X[col] = original
        out[col] = base_r2 - r2_pert
    return pd.Series(out, name="importance").sort_values(ascending=False)


def aggregate_to_base_chars(
    importance: pd.Series,
    base_chars: List[str],
) -> pd.Series:
    """
    Collapse Kronecker-product feature importances back to the base
    characteristics. Feature naming convention from
    ``data.characteristics.build_feature_matrix``::

        f"{base}_const"            (constant block)
        f"{base}_{macro}"          (macro × char interaction blocks)
        f"sic2_{code}"             (industry dummies — left as-is)

    For each ``base`` in ``base_chars``, we sum over all features whose
    name starts with ``f"{base}_"``. SIC dummies and any unmatched
    features are appended unchanged.

    Parameters
    ----------
    importance  : Series indexed by feature name (output of
                  ``zero_set_importance`` or ``permutation_importance``).
    base_chars  : the 94 firm characteristic names (e.g. mom1m, bm, ...).
    """
    out: Dict[str, float] = {}
    matched = set()
    for base in base_chars:
        prefix = f"{base}_"
        # Use loc instead of filter() to avoid regex on _ characters
        mask = importance.index.to_series().str.startswith(prefix)
        sub = importance[mask.values]
        if len(sub) > 0:
            out[base] = float(sub.sum())
            matched.update(sub.index.tolist())
    # Append unmatched features (industry dummies, etc.) as-is
    for name, val in importance.items():
        if name in matched:
            continue
        out[name] = float(val)
    return pd.Series(out, name="importance").sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: fit a model on the full training period for variable importance
# ─────────────────────────────────────────────────────────────────────────────

def fit_for_importance(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
):
    """
    Fit the model on (X_train, y_train) — with optional validation set —
    and return the fitted model. Wraps ``model.fit`` and absorbs failures.
    """
    try:
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
    except TypeError:
        # Some models don't accept val args
        model.fit(X_train, y_train)
    return model
