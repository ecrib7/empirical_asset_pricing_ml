"""
tests/test_forward_return.py
------------------------------
Forward return target alignment (no same-row ret as supervised y).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import (
    add_forward_return_target,
    feature_columns_for_training,
)


def test_add_forward_return_matches_next_month_realized():
    df = pd.DataFrame(
        {
            "permno": [1, 1, 1, 2, 2],
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-29",
                    "2020-03-31",
                    "2020-01-31",
                    "2020-02-29",
                ]
            ),
            "ret": [0.01, 0.02, 0.03, -0.01, -0.02],
            "me": [1.0] * 5,
        }
    )
    out = add_forward_return_target(df)
    row_jan = out[(out["permno"] == 1) & (out["date"] == pd.Timestamp("2020-01-31"))]
    assert float(row_jan["ret_fwd"].iloc[0]) == pytest.approx(0.02)
    row_feb = out[(out["permno"] == 1) & (out["date"] == pd.Timestamp("2020-02-29"))]
    assert float(row_feb["ret_fwd"].iloc[0]) == pytest.approx(0.03)
    row_mar = out[(out["permno"] == 1) & (out["date"] == pd.Timestamp("2020-03-31"))]
    assert pd.isna(row_mar["ret_fwd"].iloc[0])


def test_mom1m_style_feature_not_identical_to_forward_target():
    """
    Under same-row leakage, contemporaneous return (mom1m definition) equals y.
    With ret_fwd as y, they differ except for degenerate series.
    """
    rng = np.random.default_rng(42)
    n = 48
    dates = pd.date_range("2018-01", periods=n, freq="ME")
    r = rng.normal(0.0, 0.02, size=n)
    df = pd.DataFrame({"permno": 7, "date": dates, "ret": r, "me": 1.0})
    df = add_forward_return_target(df)
    mom1m = df["ret"]  # MomentumBuilder.mom1m(ret) returns ret unchanged
    mask = df["ret_fwd"].notna()
    # Same-row y = ret would equal mom1m; forward target differs for random returns
    assert not np.allclose(mom1m[mask].values, df.loc[mask, "ret_fwd"].values)


def test_raw_ret_excluded_from_feature_list_when_target_is_forward():
    """
    Regression: if ``ret`` stayed in X while y = ret_fwd, same-period return would
    leak into supervised learning.  feature_columns_for_training must drop ``ret``.
    """
    rng = np.random.default_rng(0)
    fm = pd.DataFrame(
        {
            "permno": [1] * 10,
            "date": pd.date_range("2019-01", periods=10, freq="ME"),
            "ret": rng.standard_normal(10) * 0.01,
            "ret_fwd": rng.standard_normal(10) * 0.01,
            "me": [1.0] * 10,
            "x_signal": rng.standard_normal(10),
        }
    )
    cols = feature_columns_for_training(fm, "ret_fwd")
    assert "ret" not in cols
    assert "ret_fwd" not in cols
    assert "x_signal" in cols


def test_old_leaky_feature_list_would_include_ret():
    """Sanity: naive exclusion (target + ids only) wrongly keeps ``ret`` in X."""
    fm = pd.DataFrame(
        {
            "permno": [1],
            "date": [pd.Timestamp("2020-01-31")],
            "ret": [0.05],
            "ret_fwd": [0.03],
            "me": [1.0],
            "x_signal": [1.0],
        }
    )
    naive = [c for c in fm.columns if c not in ("permno", "date", "ret_fwd", "me")]
    assert "ret" in naive  # would allow leakage
    safe = feature_columns_for_training(fm, "ret_fwd")
    assert "ret" not in safe
