"""
Reporting layer: portfolio bundle I/O and incremental TC (no double-count).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import DecilePortfolioBuilder, TransactionCostModel
from src.evaluation.metrics import sharpe_ratio
from src.reporting.portfolio_io import (
    hl_additional_tc_sharpe,
    save_portfolio_bundle,
    unpack_portfolio_bundle,
)


class TestUnpackPortfolioBundle:
    def test_legacy_flat_dict(self):
        net = {"M": {"H-L": pd.Series([0.01, -0.02], index=pd.date_range("2020-01", periods=2, freq="ME"))}}
        n, g, t, meta = unpack_portfolio_bundle(net)
        assert n is net
        assert g is None and t is None
        assert meta["format"] == "legacy"

    def test_bundle_v1_roundtrip(self, tmp_path):
        idx = pd.date_range("2020-01", periods=4, freq="ME")
        net = {"RF": {"H-L": pd.Series(np.linspace(0, 0.03, 4), index=idx)}}
        gross = {"RF": {"H-L": pd.Series(np.linspace(0.01, 0.04, 4), index=idx)}}
        turnover = {"RF": {"H-L": pd.Series([0.5, 0.2, 0.2, 0.1], index=idx)}}
        p = tmp_path / "b.pkl"
        save_portfolio_bundle(p, net, gross, turnover)
        import pickle

        with open(p, "rb") as f:
            raw = pickle.load(f)
        n, gr, to, meta = unpack_portfolio_bundle(raw)
        assert meta["format"] == "bundle_v1"
        pd.testing.assert_frame_equal(
            pd.DataFrame(n["RF"]["H-L"]),
            pd.DataFrame(net["RF"]["H-L"]),
            check_names=False,
        )
        assert gr is not None and to is not None


class TestHlAdditionalTcSharpe:
    def test_zero_additional_matches_sharpe_of_net_series(self):
        idx = pd.date_range("2018-01", periods=24, freq="ME")
        hl = pd.Series(np.random.default_rng(0).normal(0.005, 0.04, len(idx)), index=idx)
        to = pd.Series(np.abs(np.random.default_rng(1).normal(0.3, 0.1, len(idx))), index=idx)
        assert hl_additional_tc_sharpe(hl, to, 0.0) == pytest.approx(sharpe_ratio(hl.dropna()))

    def test_missing_turnover_returns_nan_for_positive_extra(self):
        hl = pd.Series([0.01, 0.02], index=pd.date_range("2020-01", periods=2, freq="ME"))
        assert np.isnan(hl_additional_tc_sharpe(hl, None, 10.0))

    def test_incremental_tc_matches_manual_series(self):
        idx = pd.date_range("2019-01", periods=36, freq="ME")
        rng = np.random.default_rng(42)
        hl_net = pd.Series(rng.normal(0.004, 0.03, len(idx)), index=idx)
        to = pd.Series(np.clip(rng.uniform(0.1, 0.9, len(idx)), 0.05, 1.0), index=idx)
        extra_bps = 15.0
        adj = hl_net - (extra_bps / 10_000.0) * to.reindex(hl_net.index).fillna(0.0)
        got = hl_additional_tc_sharpe(hl_net, to, extra_bps)
        assert got == pytest.approx(sharpe_ratio(adj.dropna()))


class TestNoDoubleCountVsEngine:
    def test_reporting_sharpe_at_zero_extra_equals_engine_net_sharpe(self):
        """UI path: additional bps=0 must not alter stored net H-L (already net of engine TC)."""
        rng = np.random.default_rng(7)
        dates = pd.date_range("2015-01", periods=48, freq="ME")
        n = 40
        perms = np.arange(1, n + 1)
        pred = pd.DataFrame(rng.standard_normal((len(dates), n)), index=dates, columns=perms)
        ret = pd.DataFrame(rng.normal(0.003, 0.025, (len(dates), n)), index=dates, columns=perms)
        tc = TransactionCostModel(cost_bps=12.0)
        b = DecilePortfolioBuilder(n_deciles=5, weighting="equal", tc_model=tc)
        net, gross, turn = b.build(pred, ret)
        hl = net["H-L"].dropna()
        to_hl = turn["H-L"].reindex(hl.index)
        sr_engine = sharpe_ratio(hl)
        sr_ui = hl_additional_tc_sharpe(hl, to_hl, 0.0)
        assert sr_ui == pytest.approx(sr_engine)
        g = gross["H-L"].reindex(hl.index).dropna()
        h = net["H-L"].reindex(g.index)
        assert (g.values >= h.values - 1e-12).all()
