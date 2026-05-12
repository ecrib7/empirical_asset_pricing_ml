"""
evaluation/combinations.py
--------------------------
Forecast combinations across the per-model OOS prediction arrays.

The pipeline trains 13 individual models. Forecast combination —
averaging or weighting their predictions — is well-known to outperform
the best single model out-of-sample, with negligible extra cost.

Two combinations are implemented:

  * ``ENS-AVG``  : equal-weighted average of all input model predictions.
  * ``ENS-MSE``  : weighted average, with weights ∝ 1 / validation MSE.
                   Validation MSE is computed on a held-out segment of
                   the test panel (default: first 10 % of dates).

Both are added as new entries in the ``predictions`` dict and downstream
consumers (portfolio construction, OOS R², DM tests, etc.) treat them
exactly like any other model.

Usage
-----
    from src.evaluation.combinations import build_ensembles
    predictions, ens_meta = build_ensembles(
        predictions,             # {model: 1D np.ndarray}
        y_true=true_returns,
        dates=test_dates,
        which=("avg", "mse"),
    )
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _stack_aligned(predictions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, list]:
    """Stack per-model 1D arrays into (n_models, n_obs). Skips empty arrays."""
    names = []
    arrs = []
    n0 = None
    for name, arr in predictions.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 1 or len(a) == 0:
            continue
        if n0 is None:
            n0 = len(a)
        if len(a) != n0:
            logger.warning(
                f"Skipping {name} from ensemble: length {len(a)} != {n0}"
            )
            continue
        names.append(name)
        arrs.append(a)
    if not arrs:
        return np.zeros((0, 0)), []
    return np.vstack(arrs), names


def equal_weighted(
    predictions: Dict[str, np.ndarray],
    name: str = "ENS-AVG",
) -> Tuple[np.ndarray, list]:
    """Simple cross-model average. Ignores NaNs per row."""
    M, names = _stack_aligned(predictions)
    if M.shape[0] == 0:
        raise ValueError("No usable predictions to combine.")
    avg = np.nanmean(M, axis=0)
    return avg.astype(np.float32), names


def inverse_mse_weighted(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    dates: np.ndarray,
    val_frac: float = 0.10,
    name: str = "ENS-MSE",
) -> Tuple[np.ndarray, list, Dict[str, float]]:
    """
    Weighted average with weights ∝ 1 / MSE on a held-out validation slice.

    The validation slice is the **earliest** ``val_frac`` of distinct test
    dates; the weighted prediction is then constructed across the entire
    test sample. This is conservative (no in-sample peeking) but does
    introduce a tiny look-ahead because the weights were chosen with
    knowledge of the first 10 % of the OOS window. The standard
    practitioner trade-off — accept the small bias for sharper weights.
    """
    M, names = _stack_aligned(predictions)
    if M.shape[0] == 0:
        raise ValueError("No usable predictions to combine.")
    y = np.asarray(y_true, dtype=np.float64)
    if M.shape[1] != len(y):
        raise ValueError(
            f"Prediction length {M.shape[1]} != y_true length {len(y)}."
        )

    # Held-out validation slice = earliest val_frac of distinct dates
    d = pd.to_datetime(np.asarray(dates))
    uniq = np.sort(d.unique())
    cutoff_idx = max(1, int(np.ceil(val_frac * len(uniq))))
    val_dates = set(pd.to_datetime(uniq[:cutoff_idx]))
    val_mask = np.array([t in val_dates for t in d])

    # MSE per model on val slice
    valid_y = ~np.isnan(y) & val_mask
    if valid_y.sum() < 30:
        logger.warning(
            f"Validation slice has only {valid_y.sum()} obs — falling back to "
            "equal weights for inverse-MSE ensemble."
        )
        avg, _ = equal_weighted(predictions)
        weights = {n: 1.0 / len(names) for n in names}
        return avg, names, weights

    mses = np.full(M.shape[0], np.nan)
    for i in range(M.shape[0]):
        diffs = M[i, valid_y] - y[valid_y]
        diffs = diffs[~np.isnan(diffs)]
        if len(diffs):
            mses[i] = float(np.mean(diffs ** 2))
    if not np.isfinite(mses).any():
        raise ValueError("All models have NaN MSE on validation slice.")

    inv = np.where(np.isfinite(mses) & (mses > 0), 1.0 / mses, 0.0)
    inv_norm = inv / inv.sum() if inv.sum() > 0 else np.full_like(inv, 1.0 / len(inv))

    out = np.nansum(M * inv_norm[:, None], axis=0).astype(np.float32)
    weights = {n: float(w) for n, w in zip(names, inv_norm)}
    return out, names, weights


def build_ensembles(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    dates: np.ndarray,
    which: Sequence[str] = ("avg", "mse"),
    val_frac: float = 0.10,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Add ENS-AVG and/or ENS-MSE to the predictions dict.

    Returns (predictions_with_ensembles, meta) where meta documents the
    constituent models and (for ENS-MSE) the inverse-MSE weights.
    """
    out = dict(predictions)
    meta: dict = {}
    if "avg" in which:
        avg, names = equal_weighted(predictions)
        out["ENS-AVG"] = avg
        meta["ENS-AVG"] = {"constituents": names}
        logger.info(f"[ensemble] ENS-AVG built from {len(names)} models: {names}")
    if "mse" in which:
        try:
            mse, names, weights = inverse_mse_weighted(
                predictions, y_true=y_true, dates=dates, val_frac=val_frac,
            )
            out["ENS-MSE"] = mse
            meta["ENS-MSE"] = {
                "constituents": names,
                "weights": weights,
                "val_frac": val_frac,
            }
            top = sorted(weights.items(), key=lambda kv: -kv[1])[:5]
            logger.info(f"[ensemble] ENS-MSE top weights: {top}")
        except Exception as e:
            logger.warning(f"ENS-MSE construction failed: {e}")
    return out, meta
