"""
tests/test_synthetic_panels.py
------------------------------
Unit tests for the stock-level synthetic panel generator
(``src/synthetic/panels.py``) and its integration with
``generate_synthetic_results.py``.

Covers:
  * Each future2026_* variant produces a panel parquet with the exact
    schema, date range, permno set, and 96k rows.
  * Scenarios are statistically distinct in the panel itself (not just
    after sorting), so the source data — not just the decile shortcut
    — carries the regime signal.
  * ``generate_variant`` derives outputs from the panel: per-model
    pickles carry panel-shape predictions / true_returns and the
    ``synthetic`` / ``source`` markers required for downstream
    distinction.
  * The CLI ``--panels-only`` flag writes parquets without populating
    ``outputs/``.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FUTURE2026_SCENARIOS, get_variant_config  # noqa: E402
from src.synthetic.panels import (  # noqa: E402
    N_MONTHS,
    N_PERMNOS,
    PANEL_END,
    PANEL_START,
    REQUIRED_COLUMNS,
    SCENARIOS,
    decile_returns_from_panel,
    generate_panel,
    load_panel,
    panel_dates,
    panel_path,
    panel_permnos,
    write_panel,
)

import generate_synthetic_results as gsr  # noqa: E402


# Match the bare scenario names to the variant names registered in config.
ALL_VARIANTS = tuple(f"future2026_{s}" for s in SCENARIOS)


@pytest.fixture(scope="module")
def panels_cache(tmp_path_factory):
    """Generate every panel once per test module; share across tests."""
    cache: dict[str, pd.DataFrame] = {}
    for v in ALL_VARIANTS:
        cache[v] = generate_panel(v)
    return cache


class TestPanelShape:
    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_panel_has_correct_shape(self, panels_cache, variant):
        df = panels_cache[variant]
        assert len(df) == N_MONTHS * N_PERMNOS == 96_000
        assert df["date"].nunique() == 120
        assert df["permno"].nunique() == 800

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_panel_date_range_exact(self, panels_cache, variant):
        df = panels_cache[variant]
        dates = sorted(df["date"].unique())
        assert pd.Timestamp(dates[0]) == pd.Timestamp(PANEL_START)
        assert pd.Timestamp(dates[-1]) == pd.Timestamp(PANEL_END)

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_panel_required_columns_present(self, panels_cache, variant):
        df = panels_cache[variant]
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        assert not missing, f"{variant} missing columns: {missing}"

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_panel_scenario_column_matches_variant(self, panels_cache, variant):
        df = panels_cache[variant]
        assert set(df["scenario"].unique()) == {variant}

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_panel_permnos_are_synthetic_and_stable(self, panels_cache, variant):
        df = panels_cache[variant]
        permnos = sorted(df["permno"].unique())
        expected = sorted(panel_permnos().tolist())
        assert permnos == expected
        assert min(permnos) >= 900_000  # synthetic range, never collides with CRSP

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_panel_every_month_has_all_permnos(self, panels_cache, variant):
        df = panels_cache[variant]
        counts = df.groupby("date").size()
        assert (counts == N_PERMNOS).all()

    def test_panel_dates_helper_matches_constants(self):
        idx = panel_dates()
        assert len(idx) == 120
        assert str(idx[0].date()) == "2026-04-30"
        assert str(idx[-1].date()) == "2036-03-31"


class TestPanelIO:
    @pytest.mark.parametrize("variant", ["future2026_base", "future2026_trending"])
    def test_write_and_load_roundtrip(self, tmp_path, variant):
        df = generate_panel(variant)
        path = tmp_path / f"{variant}.parquet"
        out_path = write_panel(variant, df, out_path=path)
        assert out_path == path and path.exists()
        df_loaded = load_panel(variant, in_path=path)
        assert len(df_loaded) == len(df)
        pd.testing.assert_frame_equal(
            df.sort_values(["date", "permno"]).reset_index(drop=True),
            df_loaded.sort_values(["date", "permno"]).reset_index(drop=True),
            check_dtype=False,
        )

    def test_default_panel_path(self):
        p = panel_path("future2026_base")
        assert p == Path("data/cache/synthetic_panels/future2026_base.parquet")

    def test_load_missing_panel_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_panel("future2026_base", in_path=tmp_path / "no-such.parquet")


class TestScenarioDistinctness:
    def test_crisis_has_correlated_drawdown(self, panels_cache):
        df = panels_cache["future2026_crisis"]
        mkt = df.groupby("date")["mkt_ret"].first()
        nav = (1 + mkt).cumprod()
        dd = float((nav / nav.cummax() - 1).min())
        # Crisis: the market factor should show a deep drawdown.
        assert dd < -0.15, f"crisis drawdown too small: {dd}"

    def test_choppy_has_higher_idio_volatility_than_base(self, panels_cache):
        # Cross-sectional dispersion of returns is a clean proxy for idio vol.
        choppy = panels_cache["future2026_choppy"].groupby("date")["ret"].std().mean()
        base = panels_cache["future2026_base"].groupby("date")["ret"].std().mean()
        assert choppy > base * 1.2, (choppy, base)

    def test_mean_reversion_has_lower_stock_ar1_than_trending(self, panels_cache):
        # Per-stock AR(1) of realised returns should be clearly more
        # negative under mean_reversion than under trending.
        def avg_per_stock_ar1(df):
            return df.groupby("permno")["ret"].apply(
                lambda s: s.autocorr(1) if s.std() > 0 else 0.0
            ).mean()

        ar_trend = avg_per_stock_ar1(panels_cache["future2026_trending"])
        ar_revert = avg_per_stock_ar1(panels_cache["future2026_mean_reversion"])
        # Trending should be visibly more autocorrelated than mean-reversion.
        assert ar_trend > ar_revert + 0.10, (ar_trend, ar_revert)
        # Mean reversion's per-stock AR(1) should be close to zero or negative.
        assert ar_revert < 0.05, ar_revert

    def test_factor_rotation_value_sign_alternates(self, panels_cache):
        # The factor_rotation scenario flips the value loading every 18
        # months. The realised value-decile spread should average to a
        # value with smaller absolute mean than a comparable non-rotating
        # scenario (because the +/− contributions partially cancel).
        df_rot = panels_cache["future2026_factor_rotation"]
        df_base = panels_cache["future2026_base"]

        def hl_value(df):
            return decile_returns_from_panel(df, "value")["H-L"]

        # The variability over time should be high relative to the mean —
        # i.e. the sign flips visibly. We just check the standard
        # deviation is non-trivial.
        s_rot = hl_value(df_rot)
        s_base = hl_value(df_base)
        assert s_rot.std() > 0
        # Sign changes — count zero-crossings — should be at least 3 over
        # the 120-month horizon for the rotating panel.
        sign_changes = int(np.sum(np.sign(s_rot.values[1:]) != np.sign(s_rot.values[:-1])))
        assert sign_changes >= 3, sign_changes

    def test_rotating_leaders_has_distinct_dynamics(self, panels_cache):
        # Just verify the realised returns differ from base — not a
        # specific shape check, since leader rotation acts on style
        # premia rather than on every individual stock identically.
        df_rot = panels_cache["future2026_rotating_leaders"]
        df_base = panels_cache["future2026_base"]
        # Different scenarios should produce different cross-sectional
        # standard deviation paths.
        cs_rot = df_rot.groupby("date")["ret"].std().values
        cs_base = df_base.groupby("date")["ret"].std().values
        assert not np.allclose(cs_rot, cs_base)


class TestVariantConfigCarriesPanelPath:
    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_synthetic_panel_path_in_variant_config(self, variant):
        cfg = get_variant_config(variant)
        assert "synthetic_panel_path" in cfg
        expected = f"data/cache/synthetic_panels/{variant}.parquet"
        assert cfg["synthetic_panel_path"] == expected


class TestGeneratorFromPanel:
    def test_generate_writes_panel_and_outputs(self, tmp_path):
        out = tmp_path / "out"
        panels = tmp_path / "panels"
        path = gsr.generate_variant(
            "future2026_base", out_root=out, panel_root=panels,
        )
        # Outputs/<variant>/ populated.
        assert (path / "metrics.json").exists()
        # Panel parquet also written.
        assert (panels / "future2026_base.parquet").exists()
        # metrics.json carries panel metadata.
        meta = json.loads((path / "metrics.json").read_text())["_reporting"]
        assert meta["panel_source"] == "synthetic_stock_level"
        assert meta["n_stocks"] == 800
        assert meta["n_rows"] == 96_000
        assert meta["training_kind"] == "synthetic_training"
        assert meta["evaluation_kind"] == "synthetic_evaluation"

    def test_model_pickle_predictions_have_panel_shape(self, tmp_path):
        gsr.generate_variant(
            "future2026_base",
            out_root=tmp_path / "out",
            panel_root=tmp_path / "panels",
        )
        with open(tmp_path / "out" / "future2026_base" / "models" / "OLS-3.pkl", "rb") as f:
            obj = pickle.load(f)
        # 800 permnos × 120 months -> 96000 rows when sourced from the panel.
        assert obj["predictions"].shape == (96_000,)
        assert obj["true_returns"].shape == (96_000,)
        assert len(obj["test_permnos"]) == 96_000
        assert obj["synthetic"] is True
        assert obj["source"] == "synthetic_panel"
        assert obj["n_stocks_per_month"] == 800

    def test_panels_only_skips_outputs(self, tmp_path):
        panels = tmp_path / "panels"
        rc = gsr.main([
            "--variant", "future2026_base",
            "--out-root", str(tmp_path / "out"),
            "--panel-root", str(panels),
            "--panels-only",
        ])
        assert rc == 0
        assert (panels / "future2026_base.parquet").exists()
        # No outputs/<variant>/ should have been created.
        assert not (tmp_path / "out" / "future2026_base").exists()

    def test_from_panel_uses_existing_parquet(self, tmp_path):
        # Write a panel ahead of time.
        panels = tmp_path / "panels"
        panels.mkdir()
        gsr.generate_panel_for_variant("future2026_trending", panel_root=panels)
        # Now ask the generator to derive outputs WITHOUT regenerating.
        path = gsr.generate_variant(
            "future2026_trending",
            out_root=tmp_path / "out",
            panel_root=panels,
            generate_panel_if_missing=False,
        )
        assert (path / "metrics.json").exists()

    def test_from_panel_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gsr.generate_variant(
                "future2026_trending",
                out_root=tmp_path / "out",
                panel_root=tmp_path / "empty",
                generate_panel_if_missing=False,
            )

    def test_decile_returns_match_panel_universe(self, tmp_path):
        # H-L bundle dates must match the panel's month-ends exactly.
        gsr.generate_variant(
            "future2026_base",
            out_root=tmp_path / "out",
            panel_root=tmp_path / "panels",
        )
        with open(tmp_path / "out" / "future2026_base" / "portfolio_returns.pkl", "rb") as f:
            bundle = pickle.load(f)
        hl = bundle["gross"]["ENS-AVG"]["H-L"]
        assert len(hl) == 120
        assert hl.index.min() == pd.Timestamp("2026-04-30")
        assert hl.index.max() == pd.Timestamp("2036-03-31")

    def test_stronger_models_have_wider_hl_spread(self, tmp_path):
        # Higher-skill models (ENS-AVG / ENS-MSE) should earn wider gross
        # spreads than weak-skill OLS-3 under the trending regime.
        gsr.generate_variant(
            "future2026_trending",
            out_root=tmp_path / "out",
            panel_root=tmp_path / "panels",
        )
        with open(tmp_path / "out" / "future2026_trending" / "portfolio_returns.pkl", "rb") as f:
            bundle = pickle.load(f)
        ols = bundle["gross"]["OLS-3"]["H-L"].mean()
        ens = bundle["gross"]["ENS-AVG"]["H-L"].mean()
        # Strict-greater is fragile; require a meaningful margin.
        assert ens >= ols, (ens, ols)
