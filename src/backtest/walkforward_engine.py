"""
Walk-forward backtesting engine (stub).

Expanding in-sample training window, periodic (e.g. annual) refit, and strictly
out-of-sample predictions for portfolio construction.

**Note:** The production GKX decile pipeline lives in ``engine.py`` (`BacktestEngine`,
`DecilePortfolioBuilder`). This module is the scaffold for a separate walk-forward
experiment runner wired from ``configs/experiment.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

class WalkForwardEngine:
    """
    Orchestrate expanding-window estimation and OOS prediction dates.

    Parameters
    ----------
    config_path
        Optional YAML path; otherwise pass ``config`` dict.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[Path, str]] = None,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path else None

    def run(self) -> Dict[str, Any]:
        """Execute walk-forward schedule and return metrics / predictions (stub)."""
        return {"status": "stub", "class": "WalkForwardEngine"}
