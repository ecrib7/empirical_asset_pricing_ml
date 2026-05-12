"""
src.backtest
------------
Recursive portfolio backtesting (GKX) and walk-forward experiment scaffolding.

* ``engine`` — production decile builder, transaction costs, and ``BacktestEngine``.
* ``walkforward_engine`` — expanding-window training / annual refit stub.
* ``simulator`` — YAML-driven end-to-end run stub.
"""

# Re-export legacy symbols commonly used by the pipeline
from src.backtest.engine import BacktestEngine, DecilePortfolioBuilder  # noqa: F401
from src.backtest.walkforward_engine import WalkForwardEngine  # noqa: F401
from src.backtest.simulator import RunSimulation  # noqa: F401

__all__ = [
    "BacktestEngine",
    "DecilePortfolioBuilder",
    "WalkForwardEngine",
    "RunSimulation",
]
