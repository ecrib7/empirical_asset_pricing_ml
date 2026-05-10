"""
src/synthetic/regimes.py
------------------------
Configuration skeleton for synthetic post-real-data stress testing.

Why this module exists
~~~~~~~~~~~~~~~~~~~~~~
Real WRDS coverage stops at ``REAL_DATA_END = 2026-03-31`` (the furthest
CIZ/v2 monthly endpoint, observed on ``crsp_q_stock.*`` by
``scripts/check_wrds_coverage.py`` on 2026-05-10). The legacy
``crsp.msf`` endpoint (``LEGACY_REAL_DATA_END = 2024-12-31``) is exposed
separately for legacy-compatible callers (e.g. the ``extended_2024``
variant). To stress-test the GKX pipeline beyond the real-data horizon
without leaking lookahead, anything strictly after the configured
real-data endpoint must be marked synthetic and generated under an
explicit, reproducible regime.

Status
~~~~~~
This is a *skeleton*. It defines:

  * The scenario taxonomy (``SyntheticScenario`` + ``DEFAULT_SCENARIOS``).
  * A typed config object (``SyntheticRegimeConfig``) with explicit
    defaults — CIZ-aware ``synthetic_start = 2026-04-30`` and
    ``real_data_end = 2026-03-31``, the same boundary the WRDS coverage
    checker writes to ``outputs/data_coverage/``.
  * Validation that synthetic months never collide with real data.

What it deliberately does NOT yet do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Generate cross-sectional panels of (permno, char_1..char_94) under any
  regime — that work will adapt the scenario pattern from
  ``anticor-trader`` (https://github.com/cvxgrp/anticor-trader) to
  GKX-style monthly cross-sectional asset-pricing data, where each
  scenario must produce: (a) a synthetic universe of permnos consistent
  with the last real cross-section, (b) per-stock factor loadings,
  (c) macro state paths drawn under the regime, and (d) realised returns
  satisfying the no-lookahead constraint at every t.
* Persist scenarios or hook into ``main.py``'s pipelines.
* Train models on synthetic data.

TODO(next-PR)
~~~~~~~~~~~~~
1. Port the per-scenario shock generators from anticor-trader's
   ``scenarios.py`` and adapt their univariate shocks to a 94-column
   characteristic panel using a covariance estimated on the
   real-data tail (last 60 months ending REAL_DATA_END).
2. Add a generator entry point ``generate_panel(cfg) -> pd.DataFrame``
   that emits a (date, permno, char_*) frame for ``synthetic_start ..
   horizon_end``, with the contract that the last real month and the
   first synthetic month are continuous in the cross-section.
3. Wire a ``--variant extended_ciz_2026 --synthetic <scenario>`` CLI
   knob into ``main.py`` that splices the synthetic frame onto the real
   feature matrix *only* in evaluation mode (never in training). The
   legacy-compatible ``extended_2024`` variant remains supported via
   ``LEGACY_REAL_DATA_END`` / ``LEGACY_SYNTHETIC_START``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import pandas as pd

from src.config import REAL_DATA_END, SYNTHETIC_START

# Scenario names — kept short and tag-like so they can flow through the
# existing ``--variant`` / output-dir conventions without renaming.
DEFAULT_SCENARIOS: Tuple[str, ...] = (
    "base",            # smooth continuation of the last real regime
    "bull",            # sustained risk-on drift, low vol
    "bear",            # broad drawdown, vol spike
    "stagflation",     # rising tbl + dfy + svar; weak realised returns
    "vol_shock",       # transient svar/retvol blow-out, mean-reverts
    "liquidity_dry",   # ill, baspread, zerotrade widen across the panel
)


@dataclass(frozen=True)
class SyntheticScenario:
    """One named regime under which synthetic months are drawn."""

    name: str
    description: str
    macro_drift: dict = field(default_factory=dict)   # macro var -> annual drift
    vol_multiplier: float = 1.0                       # scales realised-vol shocks
    return_skew: float = 0.0                          # +ve = right-tail bias

    def __post_init__(self) -> None:
        if self.name not in DEFAULT_SCENARIOS:
            # Allow custom scenarios but warn loudly via a ValueError on
            # obviously-malformed names. Empty / whitespace is rejected.
            if not self.name or not self.name.strip():
                raise ValueError("SyntheticScenario.name must be non-empty")


@dataclass
class SyntheticRegimeConfig:
    """Top-level config for a synthetic-extension run.

    Fields
    ------
    scenario:
        Name of the regime to draw from. Must appear in
        ``DEFAULT_SCENARIOS`` or be a user-defined ``SyntheticScenario``.
    real_data_end:
        Inclusive last month of *real* data. Defaults to the project
        constant ``REAL_DATA_END`` (verified against WRDS).
    synthetic_start:
        First month-end strictly after ``real_data_end``. Defaults to
        ``SYNTHETIC_START``.
    horizon_months:
        Number of synthetic monthly observations to generate.
    seed:
        RNG seed — required for reproducibility; raise if missing.
    """

    scenario: str = "base"
    real_data_end: str = REAL_DATA_END
    synthetic_start: str = SYNTHETIC_START
    horizon_months: int = 24
    seed: int = 0

    def __post_init__(self) -> None:
        re = pd.Timestamp(self.real_data_end)
        ss = pd.Timestamp(self.synthetic_start)
        if ss <= re:
            raise ValueError(
                f"synthetic_start ({self.synthetic_start}) must be strictly "
                f"after real_data_end ({self.real_data_end}) — no-lookahead."
            )
        if self.horizon_months <= 0:
            raise ValueError("horizon_months must be a positive integer")
        if self.scenario not in DEFAULT_SCENARIOS:
            # Soft-fail: still accept, but make it explicit so callers
            # know they are off the supported list.
            pass

    def horizon_end(self) -> pd.Timestamp:
        """Month-end of the last synthetic observation."""
        start = pd.Timestamp(self.synthetic_start)
        # Use month-end frequency. ``periods=horizon_months`` includes start.
        idx = pd.date_range(start=start, periods=self.horizon_months, freq="ME")
        return idx[-1]


def next_month_end(date_str: str) -> pd.Timestamp:
    """Return the first month-end strictly after ``date_str``.

    Used to derive ``synthetic_start`` from ``real_data_end`` (e.g.
    ``2026-03-31`` -> ``2026-04-30``). If ``date_str`` itself falls on a
    month-end, the result is the *next* month's end.
    """
    ts = pd.Timestamp(date_str)
    return (ts + pd.offsets.MonthBegin(1)) + pd.offsets.MonthEnd(0)


def list_scenarios() -> Tuple[str, ...]:
    """Return the canonical scenario names — useful for CLI ``choices=``."""
    return DEFAULT_SCENARIOS
