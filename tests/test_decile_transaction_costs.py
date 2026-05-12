"""
tests/test_decile_transaction_costs.py
----------------------------------------
Decile portfolio transaction costs (month-over-month weight changes).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import DecilePortfolioBuilder, TransactionCostModel


class TestTransactionCostModelPeriod:
    def test_zero_turnover_implies_zero_cost(self):
        tc = TransactionCostModel(cost_bps=25.0)
        w = pd.Series([0.25, 0.25, 0.5], index=[1, 2, 3])
        assert tc.period_turnover_cost(w, w) == pytest.approx(0.0)

    def test_higher_turnover_implies_higher_cost(self):
        tc = TransactionCostModel(cost_bps=10.0)
        w0 = pd.Series([1.0, 0.0], index=["a", "b"])
        w1 = pd.Series([0.0, 1.0], index=["a", "b"])
        w_half = pd.Series([0.5, 0.5], index=["a", "b"])
        c_flip = tc.period_turnover_cost(w1, w0)
        c_half = tc.period_turnover_cost(w_half, w0)
        assert c_flip > c_half > 0.0

    def test_net_return_le_gross_for_identical_gross_series(self):
        """One row: gross minus cost <= gross when cost non-negative."""
        tc = TransactionCostModel(cost_bps=50.0)
        gross = pd.Series([0.02], index=[pd.Timestamp("2020-01-31")])
        w = pd.DataFrame([[1.0, 0.0]], index=gross.index, columns=["a", "b"])
        w0 = pd.DataFrame([[0.0, 1.0]], index=gross.index, columns=["a", "b"])
        net = tc.net_return(gross, w, w0)
        assert (net <= gross).all()


class TestDecileBuilderWithTC:
    def test_decile_tc_zero_when_weights_unchanged(self, monkeypatch):
        """Flat predictions → stable decile membership → zero turnover cost each month."""
        rng = np.random.default_rng(1)
        dates = pd.date_range("2020-01", periods=6, freq="ME")
        n = 20
        perms = np.arange(1, n + 1)
        # Identical cross-section of predictions each month → same deciles, same weights
        base = np.linspace(-1.0, 1.0, n)
        pred = pd.DataFrame([base] * len(dates), index=dates, columns=perms)
        ret = pd.DataFrame(rng.normal(0.01, 0.02, (len(dates), n)), index=dates, columns=perms)
        me = pd.DataFrame(1e6, index=dates, columns=perms)

        tc = TransactionCostModel(cost_bps=100.0)
        b = DecilePortfolioBuilder(n_deciles=5, weighting="equal", tc_model=tc)
        out, _, _ = b.build(pred, ret)

        hl = out["H-L"].dropna()
        gross_b = DecilePortfolioBuilder(n_deciles=5, weighting="equal", tc_model=None)
        ghl = gross_b.build(pred, ret)[0]["H-L"].dropna()
        # First month: turnover from zero weights → TC > 0. Later months: identical
        # deciles and equal weights → no further rebalancing cost.
        assert hl.iloc[0] < ghl.iloc[0]
        pd.testing.assert_series_equal(hl.iloc[1:], ghl.iloc[1:], check_names=False)

    def test_decile_tc_reduces_vs_gross_when_predictions_shift(self):
        """Changing ranks forces rebalancing → strictly lower net than gross on average."""
        rng = np.random.default_rng(2)
        dates = pd.date_range("2021-01", periods=12, freq="ME")
        n = 30
        perms = np.arange(1, n + 1)
        pred = pd.DataFrame(
            rng.standard_normal((len(dates), n)),
            index=dates,
            columns=perms,
        )
        ret = pd.DataFrame(rng.normal(0.005, 0.03, (len(dates), n)), index=dates, columns=perms)

        gross_b = DecilePortfolioBuilder(n_deciles=5, weighting="equal", tc_model=None)
        net_b = DecilePortfolioBuilder(
            n_deciles=5, weighting="equal", tc_model=TransactionCostModel(cost_bps=40.0)
        )
        g = gross_b.build(pred, ret)[0]["H-L"].dropna()
        nret = net_b.build(pred, ret)[0]["H-L"].dropna()
        assert (nret <= g + 1e-15).all()
        assert nret.sum() < g.sum()
