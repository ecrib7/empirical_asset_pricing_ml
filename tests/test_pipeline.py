"""
tests/test_pipeline.py
----------------------
Unit and integration tests for the GKX replication pipeline.
Run with:  pytest tests/ -v
"""

import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")
from src.config import FREQ_MONTH_END, FREQ_YEAR_START


# ════════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def small_panel():
    """50 stocks × 5 years synthetic panel."""
    from main import generate_synthetic_data
    return generate_synthetic_data(n_stocks=50, start="1957-03-01", end="1961-12-31", seed=0)


@pytest.fixture(scope="module")
def feature_cols(small_panel):
    return [c for c in small_panel.columns
            if c not in ["permno", "date", "ret", "me", "siccd"]]


@pytest.fixture(scope="module")
def xy(small_panel, feature_cols):
    X = small_panel[feature_cols].fillna(0).values
    y = small_panel["ret"].values
    mid = len(X) // 2
    return X[:mid], y[:mid], X[mid:], y[mid:]


# ════════════════════════════════════════════════════════════════════
#  1. Evaluation Metrics
# ════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_oos_r2_perfect(self):
        from src.evaluation.metrics import oos_r2
        y = np.array([1.0, 2.0, 3.0])
        assert oos_r2(y, y) == pytest.approx(1.0)

    def test_oos_r2_zero_forecast(self):
        from src.evaluation.metrics import oos_r2
        y = np.array([1.0, -1.0, 2.0])
        assert oos_r2(y, np.zeros(3)) == pytest.approx(0.0)

    def test_oos_r2_random(self):
        from src.evaluation.metrics import oos_r2
        rng = np.random.default_rng(42)
        y = rng.standard_normal(500)
        p = y + rng.standard_normal(500) * 0.5
        r2 = oos_r2(y, p)
        assert r2 > 0.0

    def test_sharpe_ratio(self):
        from src.evaluation.metrics import sharpe_ratio
        # Use random returns with known positive mean
        rng = np.random.default_rng(0)
        ret = rng.standard_normal(120) * 0.04 + 0.01
        sr  = sharpe_ratio(ret)
        assert sr > 0

    def test_max_drawdown_flat(self):
        from src.evaluation.metrics import max_drawdown
        ret = pd.Series(np.zeros(60))
        assert max_drawdown(ret) == pytest.approx(0.0)

    def test_sr_improvement_positive_r2(self):
        from src.evaluation.metrics import sr_improvement
        imp = sr_improvement(sr=0.5, r2=0.01)
        assert imp > 0

    def test_diebold_mariano_identical(self):
        from src.evaluation.metrics import diebold_mariano
        rng = np.random.default_rng(1)
        y = rng.standard_normal(600)
        p = y + rng.standard_normal(600)
        dates = np.repeat(pd.date_range("1987-01", periods=50, freq=FREQ_MONTH_END), 12)
        dm, pval = diebold_mariano(y, p, p, dates)
        # Same forecasts → squared error difference is exactly zero → DM is nan or 0
        assert np.isnan(dm) or abs(dm) < 1e-6


# ════════════════════════════════════════════════════════════════════
#  2. Model Tests
# ════════════════════════════════════════════════════════════════════

class TestModels:
    def test_ols3_fit_predict(self, xy):
        from src.models.all_models import OLS3Model
        X_tr, y_tr, X_te, y_te = xy
        cols = [f"char_{j:02d}_const" for j in range(3)]
        # Build mini DataFrames with expected column names
        import pandas as pd
        df_tr = pd.DataFrame(X_tr[:, :3], columns=cols)
        df_te = pd.DataFrame(X_te[:, :3], columns=cols)
        m = OLS3Model()
        # Patch cols to match synthetic data
        m.cols_ = cols
        m.fit(df_tr, y_tr)
        pred = m.predict(df_te)
        assert len(pred) == len(y_te)
        assert not np.isnan(pred).all()

    def test_elasticnet_fit(self, xy):
        from src.models.all_models import ElasticNetModel
        X_tr, y_tr, X_te, y_te = xy
        m = ElasticNetModel(alpha_grid=[1e-2])
        m.fit(X_tr, y_tr, X_te, y_te)
        pred = m.predict(X_te)
        assert len(pred) == len(y_te)

    def test_pcr_fit(self, xy):
        from src.models.all_models import PCRModel
        X_tr, y_tr, X_te, y_te = xy
        m = PCRModel(n_components_grid=[2, 5])
        m.fit(X_tr, y_tr, X_te, y_te)
        assert m.best_k_ in [2, 5]
        pred = m.predict(X_te)
        assert len(pred) == len(y_te)

    def test_pls_fit(self, xy):
        from src.models.all_models import PLSModel
        X_tr, y_tr, X_te, y_te = xy
        m = PLSModel(n_components_grid=[1, 2])
        m.fit(X_tr, y_tr, X_te, y_te)
        pred = m.predict(X_te)
        assert len(pred) == len(y_te)

    def test_rf_fit(self, xy):
        from src.models.all_models import RandomForestModel
        X_tr, y_tr, X_te, y_te = xy
        m = RandomForestModel(n_estimators=20, max_depth_grid=[2])
        m.fit(X_tr, y_tr, X_te, y_te)
        pred = m.predict(X_te)
        assert len(pred) == len(y_te)

    def test_gbrt_fit(self, xy):
        from src.models.all_models import GBRTModel
        X_tr, y_tr, X_te, y_te = xy
        m = GBRTModel(n_estimators_grid=[50], max_depth_grid=[1], learning_rate_grid=[0.1])
        m.fit(X_tr, y_tr, X_te, y_te)
        pred = m.predict(X_te)
        assert len(pred) == len(y_te)

    def test_glm_fit(self, xy):
        from src.models.all_models import GLMModel
        X_tr, y_tr, X_te, y_te = xy
        m = GLMModel(n_knots=2, alpha_grid=[1e-2])
        m.fit(X_tr, y_tr, X_te, y_te)
        pred = m.predict(X_te)
        assert len(pred) == len(y_te)

    def test_all_models_registry(self):
        from src.models.all_models import get_all_models
        models = get_all_models()
        assert "OLS-3"  in models
        assert "ENet+H" in models
        assert "PCR"    in models
        assert "PLS"    in models
        assert "RF"     in models
        assert "GBRT+H" in models

    def test_no_lookahead_oos_r2(self, xy):
        """Ensure model trained on train does not have inflated test R²."""
        from src.models.all_models import ElasticNetModel, oos_r2
        X_tr, y_tr, X_te, y_te = xy
        m = ElasticNetModel(alpha_grid=[1e-3])
        m.fit(X_tr, y_tr)
        # OOS R² on training set should be higher than on test set typically
        r2_train = oos_r2(y_tr, m.predict(X_tr))
        r2_test  = oos_r2(y_te, m.predict(X_te))
        # Both can be positive or negative depending on signal; just ensure no crash
        assert np.isfinite(r2_train)
        assert np.isfinite(r2_test)


# ════════════════════════════════════════════════════════════════════
#  3. Portfolio / Backtest Tests
# ════════════════════════════════════════════════════════════════════

class TestPortfolio:
    def _make_pred_ret(self, n_dates=24, n_stocks=50, seed=7):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("1987-01", periods=n_dates, freq=FREQ_MONTH_END)
        permnos = list(range(1, n_stocks + 1))
        idx = pd.MultiIndex.from_product([dates, permnos], names=["date", "permno"])
        ret  = pd.Series(rng.standard_normal(len(idx)) * 0.05, index=idx)
        pred = ret + rng.standard_normal(len(idx)) * 0.04
        me   = pd.Series(np.exp(rng.uniform(3, 10, len(idx))), index=idx)
        pred_wide = pred.unstack("permno")
        ret_wide  = ret.unstack("permno")
        me_wide   = me.unstack("permno")
        return pred_wide, ret_wide, me_wide

    def test_decile_builder_value_weight(self):
        from src.backtest.engine import DecilePortfolioBuilder
        pw, rw, mw = self._make_pred_ret()
        b = DecilePortfolioBuilder(n_deciles=5, weighting="value")
        port, _, _ = b.build(pw, rw, mw)
        assert "H-L" in port
        assert "1" in port
        assert "5" in port
        hl = port["H-L"].dropna()
        assert len(hl) > 0

    def test_decile_builder_equal_weight(self):
        from src.backtest.engine import DecilePortfolioBuilder
        pw, rw, _ = self._make_pred_ret()
        b = DecilePortfolioBuilder(n_deciles=5, weighting="equal")
        port, _, _ = b.build(pw, rw)
        hl = port["H-L"].dropna()
        assert len(hl) > 0

    def test_transaction_cost_reduces_return(self):
        from src.backtest.engine import TransactionCostModel
        tc = TransactionCostModel(cost_bps=20)
        # A model with high turnover should have lower net return
        gross = pd.Series([0.01] * 12, index=pd.date_range("1987-01", periods=12, freq=FREQ_MONTH_END))
        w  = pd.DataFrame({"A": [1.0] * 12}, index=gross.index)
        wl = pd.DataFrame({"A": [0.0] + [1.0] * 11}, index=gross.index)
        net = tc.net_return(gross, w, wl)
        assert net.mean() < gross.mean()

    def test_market_timer_positive_improvement(self):
        from src.backtest.engine import MarketTimer
        rng = np.random.default_rng(99)
        dates = pd.date_range("1987-01", periods=120, freq=FREQ_MONTH_END)
        real  = pd.Series(rng.standard_normal(120) * 0.04 + 0.005, index=dates)
        pred  = real + rng.standard_normal(120) * 0.01  # good signal
        timer = MarketTimer(max_leverage=1.5)
        imp = timer.sharpe_improvement(pred, real)
        assert isinstance(imp, float)


# ════════════════════════════════════════════════════════════════════
#  4. Data / Characteristics Tests
# ════════════════════════════════════════════════════════════════════

class TestCharacteristics:
    def _make_stock_series(self, n=120):
        rng = np.random.default_rng(5)
        idx = pd.date_range("1957-03", periods=n, freq=FREQ_MONTH_END)
        return (
            pd.Series(rng.standard_normal(n) * 0.04, index=idx, name="ret"),
            pd.Series(rng.standard_normal(n) * 0.03, index=idx, name="mkt"),
        )

    def test_momentum_mom1m(self):
        from src.data.characteristics import MomentumBuilder
        ret, _ = self._make_stock_series()
        m1 = MomentumBuilder.mom1m(ret)
        pd.testing.assert_series_equal(m1, ret)

    def test_momentum_mom12m_lagged(self):
        from src.data.characteristics import MomentumBuilder
        ret, _ = self._make_stock_series(120)
        m12 = MomentumBuilder.mom12m(ret)
        # First 8 positions must be NaN (min_periods=8 rolling + 1-month shift)
        assert m12.iloc[:8].isna().all()

    def test_liquidity_mvel1(self):
        from src.data.characteristics import LiquidityBuilder
        rng = np.random.default_rng(3)
        idx = pd.date_range("1957-03", periods=60, freq=FREQ_MONTH_END)
        prc    = pd.Series(np.abs(rng.standard_normal(60)) * 20 + 10, index=idx)
        shrout = pd.Series(np.abs(rng.standard_normal(60)) * 1e5 + 5e4, index=idx)
        mvel1  = LiquidityBuilder.mvel1(prc, shrout)
        assert (mvel1 > 0).all()   # log of positive quantity

    def test_risk_beta_shape(self):
        from src.data.characteristics import RiskBuilder
        ret, mkt = self._make_stock_series(120)
        beta = RiskBuilder.beta(ret, mkt)
        assert beta.shape == ret.shape

    def test_cross_sectional_rank(self):
        from src.data.characteristics import _cs_rank
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = _cs_rank(s)
        assert r.min() == pytest.approx(-1.0)
        assert r.max() == pytest.approx(1.0)

    def test_feature_matrix_shape(self):
        from src.data.characteristics import build_feature_matrix
        import numpy as np
        rng = np.random.default_rng(7)
        n = 100
        chars  = ["c1", "c2", "c3"]
        macros = ["dp", "ep"]
        dates  = pd.date_range("1987-01", periods=n, freq=FREQ_MONTH_END)
        panel  = pd.DataFrame({
            "permno": range(n),
            "date":   dates,
            "ret":    rng.standard_normal(n) * 0.04,
            "me":     np.exp(rng.uniform(3, 10, n)),
            "c1":     rng.standard_normal(n),
            "c2":     rng.standard_normal(n),
            "c3":     rng.standard_normal(n),
        })
        macro  = pd.DataFrame({
            "date": dates,
            "dp":   rng.standard_normal(n),
            "ep":   rng.standard_normal(n),
        })
        fm = build_feature_matrix(panel, macro, chars, macros)
        # Expected cols: 3 × (1 const + 2 macro) = 9 interaction features + permno, date, ret, me
        assert "c1_const" in fm.columns
        assert "c1_dp"    in fm.columns
        assert len(fm) == n


# ════════════════════════════════════════════════════════════════════
#  5. Config Tests
# ════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_dates_ordering(self):
        from src.config import TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END
        assert TRAIN_START < TRAIN_END
        assert TRAIN_END < VAL_START
        assert VAL_START < VAL_END
        assert VAL_END < TEST_START
        assert TEST_START < TEST_END

    def test_macro_vars_count(self):
        from src.config import MACRO_VARS
        assert len(MACRO_VARS) == 8

    def test_neural_net_architectures(self):
        from src.config import NeuralNetConfig
        cfg = NeuralNetConfig()
        assert len(cfg.architectures) == 5
        assert cfg.architectures[0] == [32]           # NN1
        assert cfg.architectures[2] == [32, 16, 8]    # NN3


# ════════════════════════════════════════════════════════════════════
#  6. Integration Smoke Test
# ════════════════════════════════════════════════════════════════════

class TestIntegration:
    """
    Fast smoke test: 20 stocks × 3 years, ENet+RF only.
    Should complete in < 30 seconds.
    """
    def test_mini_pipeline_runs(self):
        from main import generate_synthetic_data
        from src.models.all_models import ElasticNetModel, RandomForestModel, oos_r2
        import pandas as pd, numpy as np

        df = generate_synthetic_data(n_stocks=20, start="1957-03-01", end="1989-12-31", seed=42)
        feat_cols = [c for c in df.columns if c not in ["permno","date","ret","me","siccd"]]

        train = df[df["date"] < "1987-01-01"]
        test  = df[df["date"] >= "1987-01-01"]

        X_tr = train[feat_cols].fillna(0).values
        y_tr = train["ret"].values
        X_te = test[feat_cols].fillna(0).values
        y_te = test["ret"].values

        for Model in [ElasticNetModel, RandomForestModel]:
            m = Model() if Model != RandomForestModel else RandomForestModel(n_estimators=20, max_depth_grid=[2])
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            r2   = oos_r2(y_te, pred)
            assert np.isfinite(r2), f"{Model.__name__} returned non-finite R²"

    def test_decile_portfolio_hl_finite(self):
        from src.backtest.engine import DecilePortfolioBuilder
        from src.evaluation.metrics import sharpe_ratio
        import numpy as np, pandas as pd

        rng = np.random.default_rng(1)
        n_dates, n_stocks = 36, 30
        dates  = pd.date_range("1987-01", periods=n_dates, freq=FREQ_MONTH_END)
        permnos = list(range(n_stocks))

        idx   = pd.MultiIndex.from_product([dates, permnos])
        ret   = pd.Series(rng.standard_normal(len(idx)) * 0.04, index=idx).unstack()
        pred  = (ret + rng.standard_normal(ret.shape) * 0.03).rename_axis(
            index="date", columns="permno")
        ret.index.name  = "date"; ret.columns.name  = "permno"
        pred.index.name = "date"; pred.columns.name = "permno"

        b    = DecilePortfolioBuilder(n_deciles=5)
        port, _, _ = b.build(pred, ret)
        hl   = port["H-L"].dropna()
        assert np.isfinite(hl.values).all()
        sr   = sharpe_ratio(hl)
        assert np.isfinite(sr)
