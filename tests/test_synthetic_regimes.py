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
)


class TestConfigConstants:
    def test_real_data_end_is_2024_year_end(self):
        assert REAL_DATA_END == "2024-12-31"

    def test_synthetic_start_strictly_after_real_data_end(self):
        assert pd.Timestamp(SYNTHETIC_START) > pd.Timestamp(REAL_DATA_END)
        assert SYNTHETIC_START == "2025-01-31"

    def test_extended_2024_variant_exists_and_carries_metadata(self):
        cfg = get_variant_config("extended_2024")
        assert cfg["data_end"] == "2024-12-31"
        assert cfg["test_start"] == "2017-01-01"
        assert cfg["test_end"] == "2024-12-31"
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
        # Sanity: extended_2024 must not collide with their output dirs
        ext = get_variant_config("extended_2024")
        assert ext["output_dir"] != paper["output_dir"]
        assert ext["output_dir"] != improved["output_dir"]


class TestSyntheticRegimeConfig:
    def test_defaults_align_with_project_constants(self):
        cfg = SyntheticRegimeConfig()
        assert cfg.real_data_end == REAL_DATA_END
        assert cfg.synthetic_start == SYNTHETIC_START
        assert cfg.scenario == "base"

    def test_synthetic_start_must_be_after_real_data_end(self):
        with pytest.raises(ValueError, match="strictly after"):
            SyntheticRegimeConfig(
                real_data_end="2024-12-31",
                synthetic_start="2024-12-31",
            )
        with pytest.raises(ValueError, match="strictly after"):
            SyntheticRegimeConfig(
                real_data_end="2024-12-31",
                synthetic_start="2024-11-30",
            )

    def test_horizon_months_must_be_positive(self):
        with pytest.raises(ValueError):
            SyntheticRegimeConfig(horizon_months=0)
        with pytest.raises(ValueError):
            SyntheticRegimeConfig(horizon_months=-3)

    def test_horizon_end_advances_correctly(self):
        cfg = SyntheticRegimeConfig(horizon_months=12)
        # Twelve month-ends starting 2025-01-31 → last is 2025-12-31.
        assert cfg.horizon_end() == pd.Timestamp("2025-12-31")

    def test_default_scenarios_listed(self):
        names = list_scenarios()
        assert "base" in names
        assert set(DEFAULT_SCENARIOS) == set(names)

    def test_scenario_dataclass_rejects_blank_name(self):
        with pytest.raises(ValueError):
            SyntheticScenario(name="   ", description="bad")


class TestCoverageChecker:
    """The script can run dry-run with no WRDS credentials available."""

    def test_dry_run_writes_manifest(self, tmp_path):
        from scripts.check_wrds_coverage import main

        rc = main(["--dry-run", "--out-dir", str(tmp_path)])
        assert rc == 0

        manifest_path = tmp_path / "coverage_latest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["real_data_end"] == "2024-12-31"
        assert manifest["synthetic_start"] == "2025-01-31"
        assert "tables" in manifest
        assert manifest["tables"]["crsp.msf"]["max_date"] == "2024-12-31"
        assert manifest["tables"]["comp.funda"]["max_datadate"] == "2026-04-30"
        assert manifest["tables"]["crsp.ccmxpf_linktable"]["max_linkenddt"] == "2026-01-30"
        # Username must never leak into the manifest.
        flat = json.dumps(manifest).lower()
        assert "username" not in flat
        assert "wrds_username" not in flat
