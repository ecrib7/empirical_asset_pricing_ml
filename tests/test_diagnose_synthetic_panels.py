"""
tests/test_diagnose_synthetic_panels.py
---------------------------------------
Tests for the synthetic panel diagnostics script
(``scripts/diagnose_synthetic_panels.py``).

These tests generate panels in a per-test ``tmp_path`` directory using
the in-repo ``src.synthetic.panels.generate_all_panels`` helper, run the
diagnostics CLI/library entry points, and verify:

* The output CSV exists and has every column the README references.
* There is exactly one row per generated future2026 scenario.
* Date / row / permno counts match the underlying panels.
* The crisis scenario produces a deep negative market month and a
  drawdown warning matches the documented threshold.
* The choppy scenario has higher market vol than the base panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.synthetic.panels import (  # noqa: E402
    N_MONTHS,
    N_PERMNOS,
    SCENARIOS,
    generate_all_panels,
)

import diagnose_synthetic_panels as diag  # noqa: E402


ALL_VARIANTS = tuple(f"future2026_{s}" for s in SCENARIOS)


@pytest.fixture(scope="module")
def panel_root(tmp_path_factory) -> Path:
    """Generate every panel parquet once per module."""
    root = tmp_path_factory.mktemp("synthetic_panels")
    generate_all_panels(out_root=root)
    return root


@pytest.fixture(scope="module")
def diag_csv(panel_root, tmp_path_factory) -> Path:
    """Run the diagnostics CLI once per module and return the CSV path."""
    out_dir = tmp_path_factory.mktemp("diag_outputs")
    out_csv = out_dir / "synthetic_panel_diagnostics.csv"
    rc = diag.main([
        "--panel-root", str(panel_root),
        "--output", str(out_csv),
    ])
    assert rc == 0
    assert out_csv.exists()
    return out_csv


class TestCSVOutput:
    def test_csv_exists(self, diag_csv):
        assert diag_csv.exists()

    def test_required_columns_present(self, diag_csv):
        df = pd.read_csv(diag_csv)
        missing = set(diag.DIAG_COLUMNS) - set(df.columns)
        assert not missing, f"missing CSV columns: {missing}"

    def test_one_row_per_future_scenario(self, diag_csv):
        df = pd.read_csv(diag_csv)
        assert len(df) == len(ALL_VARIANTS)
        assert set(df["variant"]) == set(ALL_VARIANTS)

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_counts_align_with_panel(self, diag_csv, variant):
        df = pd.read_csv(diag_csv)
        row = df[df["variant"] == variant].iloc[0]
        assert int(row["n_rows"]) == N_MONTHS * N_PERMNOS == 96_000
        assert int(row["n_months"]) == N_MONTHS == 120
        assert int(row["n_permnos"]) == N_PERMNOS == 800
        assert str(row["start_date"]) == "2026-04-30"
        assert str(row["end_date"]) == "2036-03-31"


class TestCrisisDetection:
    def test_crisis_min_market_month_matches_threshold(self, diag_csv):
        df = pd.read_csv(diag_csv)
        row = df[df["variant"] == "future2026_crisis"].iloc[0]
        # The crisis generator injects a ~-22% market shock at month 30.
        assert float(row["crisis_min_market_month"]) < diag.CRISIS_MAX_WORST_MONTH
        assert float(row["market_max_dd_pct"]) < diag.CRISIS_MAX_DD_PCT
        assert float(row["market_worst_month"]) == float(row["crisis_min_market_month"])

    def test_crisis_has_no_warnings_under_default_thresholds(self, diag_csv):
        df = pd.read_csv(diag_csv)
        row = df[df["variant"] == "future2026_crisis"].iloc[0]
        # The crisis panel should *pass* validation, not flag warnings.
        warnings = str(row.get("warnings", "") or "")
        assert warnings == "" or warnings == "nan", warnings

    def test_crisis_warns_when_panel_too_calm(self, tmp_path):
        """Force a 'fake crisis' panel with no shock and confirm we flag it."""
        # Build a flat-market panel and label it as crisis, then diagnose.
        from src.synthetic.panels import generate_panel, write_panel
        base = generate_panel("future2026_base")
        base["scenario"] = "future2026_crisis"
        path = tmp_path / "future2026_crisis.parquet"
        write_panel("future2026_crisis", base, out_path=path)

        d = diag.diagnose_panel(path)
        # No 22% shock present -> crisis check must flag.
        warnings = "; ".join(d.warnings)
        assert "crisis" in warnings.lower(), warnings


class TestChoppyVsBase:
    def test_choppy_market_vol_higher_than_base(self, diag_csv):
        df = pd.read_csv(diag_csv)
        choppy = df[df["variant"] == "future2026_choppy"].iloc[0]
        base = df[df["variant"] == "future2026_base"].iloc[0]
        assert float(choppy["market_vol_monthly"]) > float(base["market_vol_monthly"])
        # And by a clear margin — the scenarios are deliberately distinct.
        ratio = float(choppy["market_vol_monthly"]) / float(base["market_vol_monthly"])
        assert ratio >= diag.CHOPPY_VOL_RATIO_VS_BASE, ratio


class TestScenarioPersistence:
    def test_trending_has_positive_rank_autocorr(self, diag_csv):
        df = pd.read_csv(diag_csv)
        row = df[df["variant"] == "future2026_trending"].iloc[0]
        assert float(row["rank_autocorr_1m"]) >= diag.TRENDING_MIN_RANK_AC

    def test_mean_reversion_has_low_or_negative_rank_autocorr(self, diag_csv):
        df = pd.read_csv(diag_csv)
        row = df[df["variant"] == "future2026_mean_reversion"].iloc[0]
        assert float(row["rank_autocorr_1m"]) <= diag.MEAN_REVERSION_MAX_RANK_AC
        # Reversal spread should be positive: losers bounce back.
        assert float(row["reversal_spread_1m"]) > 0.0


class TestSummaryMarkdown:
    def test_summary_md_written_when_requested(self, panel_root, tmp_path):
        out_csv = tmp_path / "diag.csv"
        out_md = tmp_path / "diag.md"
        rc = diag.main([
            "--panel-root", str(panel_root),
            "--output", str(out_csv),
            "--summary-md", str(out_md),
        ])
        assert rc == 0
        assert out_md.exists()
        text = out_md.read_text()
        # Mentions every variant.
        for v in ALL_VARIANTS:
            assert v in text


class TestSingleVariantFilter:
    def test_variant_filter_returns_single_row(self, panel_root, tmp_path):
        out_csv = tmp_path / "one.csv"
        rc = diag.main([
            "--panel-root", str(panel_root),
            "--variant", "future2026_crisis",
            "--output", str(out_csv),
        ])
        assert rc == 0
        df = pd.read_csv(out_csv)
        assert len(df) == 1
        assert df.iloc[0]["variant"] == "future2026_crisis"

    def test_variant_bare_name_resolves(self, panel_root, tmp_path):
        out_csv = tmp_path / "bare.csv"
        rc = diag.main([
            "--panel-root", str(panel_root),
            "--variant", "trending",
            "--output", str(out_csv),
        ])
        assert rc == 0
        df = pd.read_csv(out_csv)
        assert len(df) == 1
        assert df.iloc[0]["variant"] == "future2026_trending"

    def test_missing_panel_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            diag.find_panel_paths(tmp_path / "does-not-exist")
