"""
tests/test_future2026_synthetic.py
----------------------------------
Unit tests for the future2026 synthetic-scenario generator and the
matching variant config entries.

Covers:
  * Variant registry: every future2026_* name resolves with a config
    dict that has the expected synthetic semantics.
  * Date range: exactly 120 month-ends from 2026-04-30 to 2036-03-31.
  * Output artifact format: every promised file is written and
    metrics.json carries the per-model schema the dashboard expects.
  * Scenario qualitative distinctness: trending and mean_reversion
    produce different H-L return AR(1) signs.
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

from src.config import (  # noqa: E402
    FUTURE2026_END,
    FUTURE2026_SCENARIOS,
    FUTURE2026_START,
    VARIANT_DEFAULTS,
    get_variant_config,
)

import generate_synthetic_results as gsr  # noqa: E402


EXPECTED_ARTIFACTS = (
    "metrics.json",
    "comprehensive.csv",
    "oos_r2.csv",
    "sharpe_table.csv",
    "dm_table.csv",
    "dm_pvalues.csv",
    "regimes.csv",
    "var_importance.csv",
    "portfolio_returns.pkl",
)


class TestFuture2026Config:
    def test_all_scenarios_registered(self):
        expected = {
            "future2026_base",
            "future2026_trending",
            "future2026_mean_reversion",
            "future2026_rotating_leaders",
            "future2026_choppy",
            "future2026_crisis",
            "future2026_factor_rotation",
        }
        assert expected.issubset(set(FUTURE2026_SCENARIOS))
        for name in expected:
            assert name in VARIANT_DEFAULTS, f"missing variant: {name}"

    @pytest.mark.parametrize("variant", list(FUTURE2026_SCENARIOS))
    def test_variant_config_has_expected_fields(self, variant):
        cfg = get_variant_config(variant)
        assert cfg["data_start"] == FUTURE2026_START
        assert cfg["data_end"] == FUTURE2026_END
        assert cfg["test_start"] == FUTURE2026_START
        assert cfg["test_end"] == FUTURE2026_END
        assert cfg["output_dir"] == f"outputs/{variant}"
        assert cfg["model_dir"] == f"outputs/{variant}/models"
        assert cfg["feature_cache"] == f"data/cache/feature_matrix_{variant}.parquet"
        assert cfg["checkpoint_subdir"] == f"backtest_checkpoint_{variant}"
        # Synthetic semantics
        assert cfg["synthetic_enabled"] is True
        assert cfg["is_synthetic_only"] is True
        assert cfg["synthetic_start"] == FUTURE2026_START

    def test_output_dirs_dont_collide(self):
        seen = set()
        for v in FUTURE2026_SCENARIOS:
            out = get_variant_config(v)["output_dir"]
            assert out not in seen, f"collision on {out}"
            seen.add(out)


class TestGenerator:
    def test_arg_choices_include_all_future_and_post2016(self):
        for v in FUTURE2026_SCENARIOS:
            assert v in gsr.ARG_CHOICES
        assert "post2016_ciz" in gsr.ARG_CHOICES
        assert "future2026_all" in gsr.ARG_CHOICES

    def test_date_range_exactly_120_months(self):
        for variant in FUTURE2026_SCENARIOS:
            dates, _ = gsr._resolve_dates(variant)
            assert len(dates) == 120
            assert str(dates[0].date()) == "2026-04-30"
            assert str(dates[-1].date()) == "2036-03-31"

    def test_post2016_date_range(self):
        dates, scenario = gsr._resolve_dates("post2016_ciz")
        assert str(dates[0].date()) == "2017-01-31"
        assert str(dates[-1].date()) == "2026-03-31"
        # 9y + 3m = 111 month-ends
        assert len(dates) == 111
        assert scenario == "post2016_ciz"

    def test_unknown_variant_rejected(self):
        with pytest.raises(ValueError):
            gsr._resolve_dates("not_a_real_variant")

    def test_generate_writes_expected_artifacts(self, tmp_path):
        path = gsr.generate_variant("future2026_base", out_root=tmp_path)
        assert path == tmp_path / "future2026_base"
        for fname in EXPECTED_ARTIFACTS:
            assert (path / fname).exists(), f"missing {fname}"
        # Per-model pickles
        models_dir = path / "models"
        assert models_dir.is_dir()
        pkls = list(models_dir.glob("*.pkl"))
        assert len(pkls) == len(gsr.MODELS)

    def test_metrics_json_schema(self, tmp_path):
        gsr.generate_variant("future2026_base", out_root=tmp_path)
        metrics = json.loads(
            (tmp_path / "future2026_base" / "metrics.json").read_text()
        )
        # Reporting block
        assert "_reporting" in metrics
        assert metrics["_reporting"]["n_months"] == 120
        assert metrics["_reporting"]["synthetic"] is True
        # Per-model block has the same keys as existing improved variant.
        required = {
            "oos_r2_pct", "hl_sharpe", "hl_sharpe_gross",
            "hl_mean_turnover_one_way", "hl_engine_tc_bps",
            "hl_returns_are_net_of_tc",
        }
        for m in gsr.MODELS:
            assert m in metrics, f"missing model {m} in metrics.json"
            assert required.issubset(metrics[m].keys()), m

    def test_portfolio_bundle_format(self, tmp_path):
        gsr.generate_variant("future2026_base", out_root=tmp_path)
        with open(tmp_path / "future2026_base" / "portfolio_returns.pkl", "rb") as f:
            bundle = pickle.load(f)
        assert bundle["_format"] == "bundle_v1"
        assert bundle["_version"] == 1
        for section in ("net", "gross", "turnover"):
            assert section in bundle
            assert set(bundle[section].keys()).issuperset(set(gsr.MODELS))
            # Each model -> dict of decile -> Series
            sample = bundle[section][gsr.MODELS[0]]
            assert "H-L" in sample
            assert isinstance(sample["H-L"], pd.Series)
            assert len(sample["H-L"]) == 120

    def test_model_pickle_schema_matches_existing(self, tmp_path):
        gsr.generate_variant("future2026_base", out_root=tmp_path)
        with open(
            tmp_path / "future2026_base" / "models" / "OLS-3.pkl", "rb"
        ) as f:
            obj = pickle.load(f)
        # Mirror the keys observed in outputs/improved/models/OLS-3.pkl
        for k in (
            "predictions", "true_returns", "test_dates", "test_permnos",
            "portfolio_returns", "portfolio_returns_gross",
            "portfolio_turnover", "metrics", "variant",
        ):
            assert k in obj, f"missing key {k}"
        assert obj["variant"] == "future2026_base"
        assert len(obj["predictions"]) == len(obj["true_returns"])

    def test_scenarios_are_qualitatively_distinct(self, tmp_path):
        # Trending should have positive H-L autocorrelation; mean_reversion
        # should have negative or near-zero autocorrelation. We compare the
        # AR(1) coefficient of the H-L series.
        results = {}
        for variant in ("future2026_trending", "future2026_mean_reversion"):
            gsr.generate_variant(variant, out_root=tmp_path)
            with open(tmp_path / variant / "portfolio_returns.pkl", "rb") as f:
                bundle = pickle.load(f)
            hl = bundle["gross"]["OLS-3"]["H-L"]
            ar1 = hl.autocorr(lag=1)
            results[variant] = ar1
        # Trending must have a strictly larger AR(1) than mean-reversion.
        assert results["future2026_trending"] > results["future2026_mean_reversion"]

    def test_crisis_has_drawdown(self, tmp_path):
        gsr.generate_variant("future2026_crisis", out_root=tmp_path)
        with open(tmp_path / "future2026_crisis" / "portfolio_returns.pkl", "rb") as f:
            bundle = pickle.load(f)
        # The market-factor component drives a broad drawdown around month 30.
        s = bundle["gross"]["OLS-3"]["5"]  # middle decile, closer to market
        nav = (1 + s).cumprod()
        dd = (nav / nav.cummax() - 1).min()
        # Crisis must produce at least a 5% drawdown in the middle decile.
        assert dd < -0.05, f"crisis drawdown too small: {dd}"

    def test_generate_variant_for_post2016_ciz(self, tmp_path):
        path = gsr.generate_variant("post2016_ciz", out_root=tmp_path)
        for fname in EXPECTED_ARTIFACTS:
            assert (path / fname).exists(), f"missing {fname}"

    def test_future2026_all_expands_to_all_scenarios(self, tmp_path, capsys):
        rc = gsr.main([
            "--variant", "future2026_all",
            "--out-root", str(tmp_path),
        ])
        assert rc == 0
        for v in FUTURE2026_SCENARIOS:
            assert (tmp_path / v / "metrics.json").exists()


@pytest.fixture(scope="module")
def all_future2026_outputs(tmp_path_factory):
    """Generate every future2026 variant once per module; share across tests."""
    out_root = tmp_path_factory.mktemp("future2026_outputs")
    panel_root = tmp_path_factory.mktemp("future2026_panels")
    comp_by_variant: dict[str, pd.DataFrame] = {}
    for variant in FUTURE2026_SCENARIOS:
        path = gsr.generate_variant(
            variant, out_root=out_root, panel_root=panel_root,
        )
        comp_by_variant[variant] = pd.read_csv(path / "comprehensive.csv")
    return comp_by_variant


class TestOutputPlausibility:
    """Calibration guardrails: panel-derived outputs must be plausible.

    These do NOT check exact numbers — they check upper/lower bounds and
    scenario ordering. Tight numeric assertions would break every reseed.
    """

    SHARPE_HARD_CAP = 3.0  # no future2026 model may exceed this gross-Sharpe
    SHARPE_NET_HARD_CAP = 3.0
    SHARPE_LOWER_BOUND = -2.0  # avoid catastrophic mis-calibration sign flips

    def test_no_model_sharpe_above_hard_cap(self, all_future2026_outputs):
        for variant, comp in all_future2026_outputs.items():
            max_net = float(comp["Sharpe (net)"].max())
            max_gross = float(comp["Sharpe (gross)"].max())
            assert max_net < self.SHARPE_NET_HARD_CAP, (
                f"{variant}: net Sharpe {max_net:.2f} >= {self.SHARPE_NET_HARD_CAP}"
            )
            assert max_gross < self.SHARPE_HARD_CAP, (
                f"{variant}: gross Sharpe {max_gross:.2f} >= {self.SHARPE_HARD_CAP}"
            )

    def test_no_model_sharpe_below_lower_bound(self, all_future2026_outputs):
        for variant, comp in all_future2026_outputs.items():
            min_net = float(comp["Sharpe (net)"].min())
            assert min_net > self.SHARPE_LOWER_BOUND, (
                f"{variant}: net Sharpe {min_net:.2f} <= {self.SHARPE_LOWER_BOUND}"
            )

    def test_choppy_avg_sharpe_lower_than_trending(self, all_future2026_outputs):
        choppy = all_future2026_outputs["future2026_choppy"]["Sharpe (net)"].mean()
        trending = all_future2026_outputs["future2026_trending"]["Sharpe (net)"].mean()
        assert choppy < trending - 0.3, (choppy, trending)

    def test_crisis_avg_sharpe_lower_than_base(self, all_future2026_outputs):
        crisis = all_future2026_outputs["future2026_crisis"]["Sharpe (net)"].mean()
        base = all_future2026_outputs["future2026_base"]["Sharpe (net)"].mean()
        assert crisis < base, (crisis, base)

    def test_trending_best_model_strongest_overall(self, all_future2026_outputs):
        # Trending should have the highest "best ensemble" Sharpe among the
        # major regimes (trending vs choppy vs crisis).
        best_by_scn = {
            v: all_future2026_outputs[v]["Sharpe (net)"].max()
            for v in (
                "future2026_trending",
                "future2026_base",
                "future2026_choppy",
                "future2026_crisis",
            )
        }
        assert best_by_scn["future2026_trending"] > best_by_scn["future2026_choppy"]
        assert best_by_scn["future2026_trending"] > best_by_scn["future2026_crisis"]

    def test_at_least_one_scenario_has_meaningful_drawdown(self, all_future2026_outputs):
        # Across all variants, at least one model's Max DD must exceed 8%.
        worst_dd_overall = max(
            float(comp["Max DD (%)"].max())
            for comp in all_future2026_outputs.values()
        )
        assert worst_dd_overall > 8.0, worst_dd_overall

    def test_crisis_drawdown_propagates_to_models(self, all_future2026_outputs):
        # Crisis should be the (or among the) worst drawdown regimes.
        crisis_max_dd = float(
            all_future2026_outputs["future2026_crisis"]["Max DD (%)"].max()
        )
        trending_max_dd = float(
            all_future2026_outputs["future2026_trending"]["Max DD (%)"].max()
        )
        assert crisis_max_dd > trending_max_dd, (crisis_max_dd, trending_max_dd)
        # Crisis worst-model DD must be material (> 8%).
        assert crisis_max_dd > 8.0, crisis_max_dd

    def test_ensemble_sharpe_not_absurdly_high(self, all_future2026_outputs):
        # No ENS-* model in any scenario should exceed Sharpe 2.7.
        for variant, comp in all_future2026_outputs.items():
            ens_rows = comp[comp["Model"].isin(["ENS-AVG", "ENS-MSE"])]
            if not len(ens_rows):
                continue
            ens_max = float(ens_rows["Sharpe (net)"].max())
            assert ens_max < 2.7, f"{variant} ENS max Sharpe {ens_max:.2f}"

    def test_no_zero_drawdown_for_weaker_models(self, all_future2026_outputs):
        # The H-L noise overlay should keep DD non-zero for weak models in
        # most regimes — otherwise the synthetic is overly optimistic.
        non_trivial_dd_variants = 0
        for variant, comp in all_future2026_outputs.items():
            if float(comp["Max DD (%)"].max()) >= 5.0:
                non_trivial_dd_variants += 1
        assert non_trivial_dd_variants >= 5, (
            f"only {non_trivial_dd_variants}/7 variants have meaningful DD"
        )

    def test_panel_metrics_schema_preserved(self, all_future2026_outputs):
        # Schema check: comprehensive.csv must still carry the canonical columns.
        expected_cols = {
            "Model", "Sharpe (net)", "Sharpe (gross)", "Max DD (%)",
            "OOS R² (%)", "Mean TO (1-way)", "Alpha (% / yr)", "t(alpha)",
        }
        for variant, comp in all_future2026_outputs.items():
            missing = expected_cols - set(comp.columns)
            assert not missing, f"{variant} missing comp columns: {missing}"
            # Every canonical model must appear.
            assert set(gsr.MODELS).issubset(set(comp["Model"])), variant


class TestArgparseChoices:
    def test_main_argparse_accepts_future_variants(self):
        import main as main_mod
        # Re-parse with each future2026 variant; choices=... will reject
        # anything not in the whitelist.
        for v in FUTURE2026_SCENARIOS:
            argv = ["--variant", v, "--mode", "test"]
            ns = main_mod.parse_args.__wrapped__() if hasattr(
                main_mod.parse_args, "__wrapped__"
            ) else None
            # Easier: just exercise the parser directly via the module.
            import argparse
            # Reuse the parser construction code by calling parse_args
            # with monkeypatched sys.argv.
            old = sys.argv
            try:
                sys.argv = ["main.py"] + argv
                parsed = main_mod.parse_args()
            finally:
                sys.argv = old
            assert parsed.variant == v
