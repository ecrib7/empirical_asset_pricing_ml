"""
Transaction cost model: commission + half-spread + market impact (bps).

Converts basis-point charges into per-period portfolio return drags using turnover.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class TransactionCostComponents:
    """One-way cost inputs in basis points."""

    commission_bps: float = 0.0
    half_spread_bps: float = 0.0
    market_impact_bps: float = 0.0

    def total_one_way_bps(self) -> float:
        """Sum of components (one-way)."""
        raise NotImplementedError


def apply_costs_to_returns(
    gross_returns: pd.Series,
    turnover_one_way: pd.Series,
    costs: TransactionCostComponents,
) -> pd.Series:
    """Subtract linear cost ``(bps / 1e4) * turnover`` from ``gross_returns`` (aligned index)."""
    raise NotImplementedError
