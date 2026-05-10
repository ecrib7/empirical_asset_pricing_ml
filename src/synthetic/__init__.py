"""
src.synthetic
=============
Skeleton package for synthetic post-2024 regime generation. The actual
panel-generation logic is intentionally not implemented here yet — see
``src/synthetic/regimes.py`` for the configuration surface and the
roadmap docstring.
"""

from src.synthetic.regimes import (
    DEFAULT_SCENARIOS,
    SyntheticRegimeConfig,
    SyntheticScenario,
    list_scenarios,
)

__all__ = [
    "DEFAULT_SCENARIOS",
    "SyntheticRegimeConfig",
    "SyntheticScenario",
    "list_scenarios",
]
