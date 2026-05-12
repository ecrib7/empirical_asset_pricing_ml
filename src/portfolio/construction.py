"""
Cross-sectional decile sorts and long–short portfolio weight formation.

Maps predicted returns (or signals) to value- or equal-weighted decile portfolios
and top-minus-bottom spreads.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import pandas as pd


def decile_long_short_weights(
    predictions: pd.Series,
    market_caps: Optional[pd.Series] = None,
    n_deciles: int = 10,
    weighting: Literal["value", "equal"] = "value",
) -> Dict[str, pd.Series]:
    """
    Assign stocks to deciles by ``predictions`` and return portfolio weights.

    Parameters
    ----------
    predictions
        Cross-sectional scores at a single date (index = security id).
    market_caps
        Required when ``weighting='value'``.
    n_deciles
        Number of ranked buckets (default 10).
    weighting
        ``value`` or ``equal`` within each decile.

    Returns
    -------
    dict
        Keys include decile labels and ``"H-L"`` for the long-short spread.
    """
    raise NotImplementedError
