"""
Monthly portfolio turnover: one-way and two-way measures from weight changes.
"""

from __future__ import annotations

import pandas as pd


def monthly_turnover(
    weights: pd.DataFrame,
    one_way: bool = True,
) -> pd.Series:
    """
    Compute turnover each period from a weight panel (index = dates, columns = assets).

    Parameters
    ----------
    weights
        Portfolio weights before and after rebalance rows.
    one_way
        If True, return Σ|Δw|/2; else return Σ|Δw|.
    """
    raise NotImplementedError
