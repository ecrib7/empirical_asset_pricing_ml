"""
reporting/portfolio_io.py
-------------------------
Load/save helpers for portfolio return bundles (net / gross / turnover).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd


BUNDLE_VERSION = 1


def unpack_portfolio_bundle(
    data: Any,
) -> Tuple[Dict[str, Dict[str, pd.Series]], Optional[Dict], Optional[Dict], Dict]:
    """
    Normalise pickle contents.

    Returns
    -------
    net, gross, turnover, meta
        ``meta`` includes at least ``{"format": "legacy"|"bundle_v1"}``.
    """
    if isinstance(data, dict) and data.get("_format") == "bundle_v1" and "net" in data:
        meta = {k: v for k, v in data.items() if k in ("_format", "_version")}
        return (
            data["net"],
            data.get("gross"),
            data.get("turnover"),
            {**meta, "format": "bundle_v1"},
        )
    if isinstance(data, dict) and "net" in data and "gross" in data:
        return data["net"], data.get("gross"), data.get("turnover"), {"format": "bundle_v1"}
    return data, None, None, {"format": "legacy"}


def save_portfolio_bundle(
    path: Union[str, Path],
    net: Dict[str, Dict[str, pd.Series]],
    gross: Dict[str, Dict[str, pd.Series]],
    turnover: Dict[str, Dict[str, pd.Series]],
) -> None:
    """Write versioned bundle for dashboards and downstream tools."""
    payload = {
        "_format": "bundle_v1",
        "_version": BUNDLE_VERSION,
        "net": net,
        "gross": gross,
        "turnover": turnover,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def hl_additional_tc_sharpe(
    hl_net: pd.Series,
    turnover_one_way: Optional[pd.Series],
    additional_bps_one_way: float,
) -> float:
    """
    Sharpe of H-L returns after an *additional* one-way TC (bps) on stored turnover.

    When ``additional_bps_one_way`` is 0, returns are unchanged (no double-count
    of the engine's embedded TC in ``hl_net``).
    """
    from src.evaluation.metrics import sharpe_ratio

    hl = hl_net.dropna()
    if len(hl) == 0:
        return float("nan")
    if additional_bps_one_way == 0.0:
        return sharpe_ratio(hl)
    if turnover_one_way is None or len(turnover_one_way.dropna()) == 0:
        return float("nan")
    to = turnover_one_way.reindex(hl.index).fillna(0.0)
    adj = hl - (additional_bps_one_way / 10_000.0) * to
    return sharpe_ratio(adj.dropna())
