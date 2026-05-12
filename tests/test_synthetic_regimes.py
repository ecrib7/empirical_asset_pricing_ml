"""
tests/test_synthetic_regimes.py
-------------------------------
Unit tests for the synthetic-extension config skeleton and the WRDS
coverage manifest builder. Both target run without WRDS credentials.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    LEGACY_REAL_DATA_END,
    LEGACY_SYNTHETIC_START,
    REAL_DATA_END,
    SYNTHETIC_START,
    VARIANT_DEFAULTS,
    get_variant_config,
)
from src.synthetic.regimes import (
    DEFAULT_SCENARIOS,
    SyntheticRegimeConfig,
    SyntheticScenario,
    list_scenarios,
    next_month_end,
)


class TestConfigConstants:
    def test_legacy_real_data_end_is_2024_year_end(self):
        # crsp.msf max(date) on the user's WRDS subscription.
        assert LEGACY_REAL_DATA_END == "2024-12-31"
        assert LEGACY_SYNTHETIC_START == "2025-01-31"

    def test_real_data_end_is_ciz_endpoint(self):
        # Furthest CIZ monthly endpoint (crsp_q_stock.* on 2026-05-10).
        assert REAL_DATA_END == "2026-03-31"

    def test_synthetic_start_strictly_after_real_data_end(self):
        assert pd.Timestamp(SYNTHETIC_START) > pd.Timestamp(REAL_DATA_END)
        assert SYNTHETIC_START == "2026-04-30"

    def test_extended_2024_variant_remains_legacy_compatible(self):
        # Non-breaking: legacy variant keeps the 2024-12-31 endpoint.
        cfg = get_variant_config("extended_2024")
        assert cfg["data_end"] == "2024-12-31"
        assert cfg["test_start"] == "2017-01-01"
        assert cfg["test_end"] == "2024-12-31"
        assert cfg["real_data_end"] == LEGACY_REAL_DATA_END
        assert cfg["synthetic_start"] == LEGACY_SYNTHETIC_START
        assert cfg["synthetic_enabled"] is False

    def test_extended_ciz_2026_variant_uses_ciz_endpoint(self):
        cfg = get_variant_config("extended_ciz_2026")
        assert cfg["data_end"] == "2026-03-31"
        assert cfg["test_start"] == "2017-01-01"
        assert cfg["test_end"] == "2026-03-31"
        assert cfg["real_data_end"] == REAL_DATA_END
        assert cfg["synthetic_start"] == SYNTHETIC_START
        assert cfg["synthetic_enabled"] is False

    def test_existing_variants_are_unchanged(self):
        # Critical: paper / improved must not regress.
        paper = get_variant_config("paper")
        improved = get_variant_config("improved")
        assert paper["data_start"] == "1957-01-01"
        assert paper["data_end"] == "2016-12-31"
        assert improved["data_start"] == "1957-01-01"
        assert improved["data_end"] == "2024-12-31"
        # Sanity: extension variants must not collide with their output dirs
        ext = get_variant_config("extended_2024")
        ext_ciz = get_variant_config("extended_ciz_2026")
        assert ext["output_dir"] != paper["output_dir"]
        assert ext["output_dir"] != improved["output_dir"]
        assert ext_ciz["output_dir"] != ext["output_dir"]
        assert ext_ciz["feature_cache"] != ext["feature_cache"]


class TestSyntheticRegimeConfig:
    def test_defaults_align_with_project_constants(self):
        cfg = SyntheticRegimeConfig()
        assert cfg.real_data_end == REAL_DATA_END
        assert cfg.synthetic_start == SYNTHETIC_START
        assert cfg.scenario == "base"

    def test_synthetic_start_must_be_after_real_data_end(self):
        with pytest.raises(ValueError, match="strictly after"):
            SyntheticRegimeConfig(
                real_data_end="2026-03-31",
                synthetic_start="2026-03-31",
            )
        with pytest.raises(ValueError, match="strictly after"):
            SyntheticRegimeConfig(
                real_data_end="2026-03-31",
                synthetic_start="2026-02-28",
            )

    def test_legacy_endpoints_still_validate(self):
        # Legacy-compatible callers can still build a config pinned to
        # the 2024-12-31 endpoint.
        cfg = SyntheticRegimeConfig(
            real_data_end=LEGACY_REAL_DATA_END,
            synthetic_start=LEGACY_SYNTHETIC_START,
        )
        assert cfg.real_data_end == "2024-12-31"
        assert cfg.synthetic_start == "2025-01-31"

    def test_horizon_months_must_be_positive(self):
        with pytest.raises(ValueError):
            SyntheticRegimeConfig(horizon_months=0)
        with pytest.raises(ValueError):
            SyntheticRegimeConfig(horizon_months=-3)

    def test_horizon_end_advances_correctly(self):
        cfg = SyntheticRegimeConfig(horizon_months=12)
        # Twelve month-ends starting 2026-04-30 -> last is 2027-03-31.
        assert cfg.horizon_end() == pd.Timestamp("2027-03-31")

    def test_default_scenarios_listed(self):
        names = list_scenarios()
        assert "base" in names
        assert set(DEFAULT_SCENARIOS) == set(names)

    def test_scenario_dataclass_rejects_blank_name(self):
        with pytest.raises(ValueError):
            SyntheticScenario(name="   ", description="bad")


class TestNextMonthEnd:
    def test_next_month_end_from_month_end(self):
        # Snap from 2026-03-31 to 2026-04-30.
        assert next_month_end("2026-03-31") == pd.Timestamp("2026-04-30")

    def test_next_month_end_from_legacy_endpoint(self):
        # 2024-12-31 -> 2025-01-31.
        assert next_month_end("2024-12-31") == pd.Timestamp("2025-01-31")

    def test_next_month_end_from_mid_month(self):
        # Mid-month also rolls forward to the *next* month-end.
        assert next_month_end("2026-03-15") == pd.Timestamp("2026-04-30")


class TestCoverageChecker:
    """The script can run dry-run with no WRDS credentials available."""

    def test_dry_run_writes_manifest(self, tmp_path):
        from scripts.check_wrds_coverage import main

        rc = main(["--dry-run", "--out-dir", str(tmp_path)])
        assert rc == 0

        manifest_path = tmp_path / "coverage_latest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        # CIZ-aware endpoint wins when crsp_q_stock.* is present.
        assert manifest["real_data_end"] == "2026-03-31"
        assert manifest["ciz_real_data_end"] == "2026-03-31"
        assert manifest["legacy_real_data_end"] == "2024-12-31"
        assert manifest["synthetic_start"] == "2026-04-30"
        assert "tables" in manifest
        # Legacy CRSP tables still tracked for legacy reproducibility.
        assert manifest["tables"]["crsp.msf"]["max_date"] == "2024-12-31"
        # CIZ/v2 endpoints picked up.
        assert (
            manifest["tables"]["crsp_q_stock.msf_v2"]["max_mthcaldt"]
            == "2026-03-31"
        )
        assert (
            manifest["tables"]["crsp_q_stock.stkmthsecuritydata"]["max_mthcaldt"]
            == "2026-03-31"
        )
        assert manifest["tables"]["comp.funda"]["max_datadate"] == "2026-04-30"
        assert manifest["tables"]["crsp.ccmxpf_linktable"]["max_linkenddt"] == "2026-01-30"
        # Username must never leak into the manifest.
        flat = json.dumps(manifest).lower()
        assert "username" not in flat
        assert "wrds_username" not in flat

    def test_dry_run_legacy_only_falls_back(self, tmp_path):
        """If only legacy CRSP tables are present, real_data_end should
        fall back to the legacy endpoint."""
        from scripts.check_wrds_coverage import build_manifest

        legacy_only = {
            "crsp.msf":      {"max_date":     "2024-12-31"},
            "crsp.msenames": {"max_namedt":   "2024-12-31",
                              "max_nameendt": "2024-12-31"},
        }
        manifest = build_manifest(legacy_only)
        assert manifest["ciz_real_data_end"] is None
        assert manifest["legacy_real_data_end"] == "2024-12-31"
        assert manifest["real_data_end"] == "2024-12-31"
        assert manifest["synthetic_start"] == "2025-01-31"
