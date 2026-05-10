"""
tests/test_post2016_ciz_predict.py
----------------------------------
Cover the config, CLI and pickle-slicing logic for the new
``post2016_ciz`` scoring variant. Does not require WRDS, model
training, or heavy feature matrices — just exercises the bookkeeping
in ``src.config`` and ``main.run_predict``.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

from src.config import (
    LEGACY_REAL_DATA_END,
    REAL_DATA_END,
    VARIANT_DEFAULTS,
    get_variant_config,
)


# ────────────────────────────────────────────────────────────────────
# Config tests
# ────────────────────────────────────────────────────────────────────

def test_post2016_ciz_variant_registered():
    assert "post2016_ciz" in VARIANT_DEFAULTS
    # Existing variants are preserved
    for name in ("paper", "improved", "extended_2024", "extended_ciz_2026"):
        assert name in VARIANT_DEFAULTS


def test_post2016_ciz_dates_and_paths():
    cfg = get_variant_config("post2016_ciz")
    assert cfg["data_start"] == "2015-01-01"
    assert cfg["data_end"] == "2026-03-31"
    assert cfg["test_start"] == "2017-01-01"
    assert cfg["test_end"] == "2026-03-31"
    assert cfg["output_dir"] == "outputs/post2016_ciz"
    assert cfg["model_dir"] == "outputs/post2016_ciz/models"
    assert cfg["feature_cache"] == "data/cache/feature_matrix_post2016_ciz.parquet"
    assert cfg["checkpoint_subdir"] == "backtest_checkpoint_post2016_ciz"
    assert cfg["real_data_end"] == REAL_DATA_END
    assert cfg.get("is_scoring_variant") is True


def test_post2016_ciz_warmup_predates_test_start():
    cfg = get_variant_config("post2016_ciz")
    # 12-month warmup required for momentum / rolling features
    assert pd.Timestamp(cfg["data_start"]) <= (
        pd.Timestamp(cfg["test_start"]) - pd.DateOffset(months=12)
    )


def test_post2016_ciz_is_ciz_aware():
    from src.data.wrds_loader import CIZ_AWARE_VARIANTS
    assert "post2016_ciz" in CIZ_AWARE_VARIANTS
    # And the helper in main agrees:
    import argparse
    import main as m
    ns = argparse.Namespace(variant="post2016_ciz")
    cfg = m._resolve_variant(ns)
    assert m._crsp_data_source(cfg) == "ciz"


# ────────────────────────────────────────────────────────────────────
# CLI tests
# ────────────────────────────────────────────────────────────────────

def test_cli_accepts_post2016_ciz_variant_and_predict_mode(monkeypatch):
    import main as m
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--mode", "predict",
            "--variant", "post2016_ciz",
            "--source-model-variant", "improved",
            "--models", "OLS-3", "ENet+H",
        ],
    )
    args = m.parse_args()
    assert args.mode == "predict"
    assert args.variant == "post2016_ciz"
    assert args.source_model_variant == "improved"
    assert args.models == ["OLS-3", "ENet+H"]


def test_cli_predict_requires_source_model_variant(monkeypatch, tmp_path):
    import main as m
    monkeypatch.chdir(tmp_path)
    ns = type("NS", (), {})()
    ns.variant = "post2016_ciz"
    ns.source_model_variant = None
    ns.models = None
    with pytest.raises(ValueError, match="source-model-variant"):
        m.run_predict(ns)


def test_cli_predict_rejects_same_source_and_target(monkeypatch, tmp_path):
    import main as m
    monkeypatch.chdir(tmp_path)
    ns = type("NS", (), {})()
    ns.variant = "post2016_ciz"
    ns.source_model_variant = "post2016_ciz"
    ns.models = None
    with pytest.raises(ValueError, match="cannot equal"):
        m.run_predict(ns)


# ────────────────────────────────────────────────────────────────────
# Slicing helper tests
# ────────────────────────────────────────────────────────────────────

def _make_fake_pickle(n_dates=24, start="2015-01-31", n_permnos=3, variant="improved"):
    dates = pd.date_range(start, periods=n_dates, freq="ME")
    # Build a (date × permno) Cartesian frame
    rows = []
    for d in dates:
        for pn in range(1, n_permnos + 1):
            rows.append((d, pn))
    df = pd.DataFrame(rows, columns=["date", "permno"])
    n = len(df)
    rng = np.random.default_rng(0)
    return {
        "predictions":  rng.standard_normal(n).astype(np.float32),
        "true_returns": rng.standard_normal(n).astype(np.float32) * 0.02,
        "test_dates":   pd.DatetimeIndex(df["date"]),
        "test_permnos": list(df["permno"].values),
        "portfolio_returns":       {"H-L": pd.Series(rng.standard_normal(n_dates), index=dates)},
        "portfolio_returns_gross": {"H-L": pd.Series(rng.standard_normal(n_dates), index=dates)},
        "portfolio_turnover":      {"H-L": pd.Series(rng.random(n_dates), index=dates)},
        "metrics":                 {"oos_r2_pct": 1.23},
        "variant":                 variant,
    }


def test_slice_pickle_to_window_basic():
    import main as m
    src = _make_fake_pickle(n_dates=24, start="2015-01-31", n_permnos=2)
    out = m._slice_pickle_to_window(
        src, pd.Timestamp("2016-01-01"), pd.Timestamp("2016-12-31"),
    )
    assert out is not None
    # 12 months × 2 permnos = 24 rows
    assert len(out["predictions"]) == 24
    assert len(out["true_returns"]) == 24
    assert out["test_dates"].min() >= pd.Timestamp("2016-01-01")
    assert out["test_dates"].max() <= pd.Timestamp("2016-12-31")
    # portfolio series are also sliced
    hl = out["portfolio_returns"]["H-L"]
    assert hl.index.min() >= pd.Timestamp("2016-01-01")
    assert hl.index.max() <= pd.Timestamp("2016-12-31")
    # Provenance metadata
    assert out["_source_pickle_variant"] == "improved"
    assert out["_sliced_to"]["test_start"] == "2016-01-01"
    assert out["_sliced_to"]["test_end"] == "2016-12-31"


def test_slice_pickle_to_window_empty():
    import main as m
    src = _make_fake_pickle(n_dates=12, start="2015-01-31")
    # Window entirely after source coverage
    out = m._slice_pickle_to_window(
        src, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"),
    )
    assert out is None


def test_run_predict_writes_sliced_pickles(monkeypatch, tmp_path):
    import main as m
    # Operate from a scratch CWD so we don't touch the real outputs/
    monkeypatch.chdir(tmp_path)

    # Build source pickles in the 'improved' layout, covering 2016-2018
    src_dir = tmp_path / "outputs" / "improved" / "models"
    src_dir.mkdir(parents=True)
    for name in ("OLS-3", "ENet+H"):
        pkl = _make_fake_pickle(n_dates=36, start="2016-01-31", n_permnos=2)
        with open(src_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(pkl, f)

    # Build a CLI args namespace
    ns = type("NS", (), {})()
    ns.variant = "post2016_ciz"
    ns.source_model_variant = "improved"
    ns.models = ["OLS-3", "ENet+H"]
    for opt in ("data_start", "data_end", "train_start", "val_start",
                "val_end", "test_start", "test_end", "tc_bps"):
        setattr(ns, opt, None)

    result = m.run_predict(ns)
    assert sorted(result["written"]) == ["ENet+H", "OLS-3"]
    assert result["source_variant"] == "improved"
    assert result["target_variant"] == "post2016_ciz"

    # Verify outputs exist and have the correct sliced window
    dst_dir = tmp_path / "outputs" / "post2016_ciz" / "models"
    for name in ("OLS-3", "ENet+H"):
        out_path = dst_dir / f"{name}.pkl"
        assert out_path.exists()
        with open(out_path, "rb") as f:
            sliced = pickle.load(f)
        # Window restricted to test_start = 2017-01-01
        assert pd.DatetimeIndex(sliced["test_dates"]).min() >= pd.Timestamp("2017-01-01")
        assert sliced["_source_pickle_variant"] == "improved"
        assert sliced["_sliced_to"]["test_start"] == "2017-01-01"


def test_run_predict_warns_when_source_short_of_test_end(
    monkeypatch, tmp_path, caplog,
):
    import logging as _l
    import main as m
    monkeypatch.chdir(tmp_path)

    src_dir = tmp_path / "outputs" / "improved" / "models"
    src_dir.mkdir(parents=True)
    # Source pickle only covers up to 2018 — well short of 2026-03 test_end
    pkl = _make_fake_pickle(n_dates=24, start="2017-01-31", n_permnos=1)
    with open(src_dir / "OLS-3.pkl", "wb") as f:
        pickle.dump(pkl, f)

    ns = type("NS", (), {})()
    ns.variant = "post2016_ciz"
    ns.source_model_variant = "improved"
    ns.models = ["OLS-3"]
    for opt in ("data_start", "data_end", "train_start", "val_start",
                "val_end", "test_start", "test_end", "tc_bps"):
        setattr(ns, opt, None)

    caplog.set_level(_l.WARNING, logger="main")
    m.run_predict(ns)
    assert any(
        "source predictions end at" in rec.message
        for rec in caplog.records
    )
