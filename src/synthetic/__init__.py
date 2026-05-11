"""
src.synthetic
=============
Synthetic post-2024 regime generation. Two surfaces:

* ``regimes`` — typed config skeleton + scenario taxonomy for an
  eventual real-WRDS-anchored extension pipeline.
* ``panels``  — stock-level monthly synthetic panels for the
  ``future2026_*`` variants (120 months × 800 permnos, one parquet per
  scenario). These power ``generate_synthetic_results.py`` so the
  future-2026 dashboard variants reflect actual cross-sectional sorts
  on a synthetic universe rather than decile-only shortcuts.
"""

from src.synthetic.regimes import (
    DEFAULT_SCENARIOS,
    SyntheticRegimeConfig,
    SyntheticScenario,
    list_scenarios,
)
from src.synthetic.panels import (
    PANEL_END,
    PANEL_START,
    REQUIRED_COLUMNS,
    SCENARIOS as PANEL_SCENARIOS,
    decile_returns_from_panel,
    generate_all_panels,
    generate_panel,
    load_panel,
    panel_dates,
    panel_path,
    panel_permnos,
    write_panel,
)

__all__ = [
    "DEFAULT_SCENARIOS",
    "SyntheticRegimeConfig",
    "SyntheticScenario",
    "list_scenarios",
    # panels
    "PANEL_END",
    "PANEL_START",
    "REQUIRED_COLUMNS",
    "PANEL_SCENARIOS",
    "decile_returns_from_panel",
    "generate_all_panels",
    "generate_panel",
    "load_panel",
    "panel_dates",
    "panel_path",
    "panel_permnos",
    "write_panel",
]
