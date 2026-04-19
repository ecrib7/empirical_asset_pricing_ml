"""
tests/test_wrds_loader.py
-------------------------
Regression tests for WRDSLoader data utilities (no live WRDS required).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import wrds_loader
from src.data.wrds_loader import WRDSLoader, merge_crsp_compustat


class TestMergeCrspCompustat:
    """merge_asof must be sorted by (permno, date) within each permno group."""

    def test_point_in_time_merge_per_permno(self):
        link = pd.DataFrame(
            {
                "gvkey": ["G1", "G2"],
                "permno": [1, 2],
                "linkdt": pd.to_datetime(["2000-01-01"] * 2),
                "linkenddt": pd.to_datetime(["2099-12-31"] * 2),
            }
        )
        comp = pd.DataFrame(
            {
                "gvkey": ["G1", "G2"],
                "datadate": pd.to_datetime(["2018-06-30", "2018-06-30"]),
                "at": [1000.0, 2000.0],
            }
        )
        # CRSP rows deliberately not sorted by (permno, date)
        crsp = pd.DataFrame(
            {
                "permno": [2, 1, 1],
                "date": pd.to_datetime(
                    ["2020-01-31", "2020-01-31", "2020-03-31"]
                ),
                "ret": [0.01, 0.02, 0.03],
            }
        )
        merged = merge_crsp_compustat(crsp, comp, link, lag_months=6)
        assert merged.groupby("permno")["date"].apply(
            lambda s: s.is_monotonic_increasing
        ).all()

        m1 = merged[merged["permno"] == 1].set_index("date")["at"]
        m2 = merged[merged["permno"] == 2].set_index("date")["at"]
        # Both January rows should pick the same lagged annual report (at 1000 / 2000)
        assert m1.loc[pd.Timestamp("2020-01-31")] == pytest.approx(1000.0)
        assert m2.loc[pd.Timestamp("2020-01-31")] == pytest.approx(2000.0)
        assert m1.loc[pd.Timestamp("2020-03-31")] == pytest.approx(1000.0)

    def test_merge_keeps_left_row_count_and_keys(self):
        """Left CRSP rows are 1:1 with merged output: no drops, no duplicate (permno, date)."""
        link = pd.DataFrame(
            {
                "gvkey": ["GA", "GB"],
                "permno": [10, 20],
                "linkdt": pd.to_datetime(["1990-01-01"] * 2),
                "linkenddt": pd.to_datetime(["2099-12-31"] * 2),
            }
        )
        comp = pd.DataFrame(
            {
                "gvkey": ["GA", "GB"],
                "datadate": pd.to_datetime(["2015-12-31", "2015-12-31"]),
                "at": [50.0, 60.0],
            }
        )
        crsp = pd.DataFrame(
            {
                "permno": [10, 20, 10, 20],
                "date": pd.to_datetime(
                    ["2019-01-31", "2019-01-31", "2019-02-28", "2019-02-28"]
                ),
                "ret": [0.01, -0.02, 0.03, -0.04],
            }
        )
        n_left = len(crsp)
        keys_left = crsp[["permno", "date"]].drop_duplicates()
        assert len(keys_left) == n_left, "fixture must use unique (permno, date) pairs"

        merged = merge_crsp_compustat(crsp, comp, link, lag_months=6)

        assert len(merged) == n_left, "merge must preserve every CRSP row"
        dup = merged.duplicated(subset=["permno", "date"], keep=False)
        assert not dup.any(), "merge must not introduce duplicate (permno, date) keys"

        left_keys = set(zip(crsp["permno"].astype(int), crsp["date"]))
        out_keys = set(zip(merged["permno"].astype(int), merged["date"]))
        assert left_keys == out_keys, "(permno, date) keys must match left exactly"


class TestMacroStubPolicy:
    def test_get_macro_predictors_refuses_stub_by_default(self, tmp_path, monkeypatch):
        loader = WRDSLoader(
            wrds_username="",
            cache_dir=str(tmp_path) + "/",
            start_date="2020-01-01",
            end_date="2020-03-31",
        )

        def _fail_wrds():
            raise RuntimeError("no wrds")

        monkeypatch.setattr(loader, "_fetch_macro_from_wrds", _fail_wrds)

        with pytest.raises(RuntimeError, match="Refusing to use silent all-zero"):
            loader.get_macro_predictors(
                goyal_csv_path=None,
                force_refresh=True,
                allow_macro_stub=False,
            )

    def test_get_macro_predictors_stub_when_opt_in(self, tmp_path, monkeypatch):
        loader = WRDSLoader(
            wrds_username="",
            cache_dir=str(tmp_path) + "/",
            start_date="2020-01-01",
            end_date="2020-03-31",
        )

        def _boom():
            raise RuntimeError("no wrds")

        monkeypatch.setattr(loader, "_fetch_macro_from_wrds", _boom)

        df = loader.get_macro_predictors(
            goyal_csv_path=None,
            force_refresh=True,
            allow_macro_stub=True,
        )
        assert "dp" in df.columns
        assert float(df["dp"].abs().sum()) == 0.0

    def test_cached_zero_macro_raises_without_opt_in(self, tmp_path):
        loader = WRDSLoader(
            wrds_username="",
            cache_dir=str(tmp_path) + "/",
            start_date="2020-01-01",
            end_date="2020-03-31",
        )
        cache_name = "macro_predictors_2020_2020.parquet"
        stub = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", "2020-03-31", freq="ME"),
                "dp": 0.0,
                "ep": 0.0,
                "bm": 0.0,
                "ntis": 0.0,
                "tbl": 0.0,
                "tms": 0.0,
                "dfy": 0.0,
                "svar": 0.0,
            }
        )
        stub.to_parquet(tmp_path / cache_name, index=False)

        with pytest.raises(RuntimeError, match="Cached macro file appears to be an all-zero stub"):
            loader.get_macro_predictors(force_refresh=False, allow_macro_stub=False)

    def test_macro_stub_detection_helper(self):
        df = pd.DataFrame(
            {
                "dp": [0.0, 0.0],
                "ep": [0.0, 0.0],
                "bm": [0.0, 0.0],
                "ntis": [0.0, 0.0],
                "tbl": [0.0, 0.0],
                "tms": [0.0, 0.0],
                "dfy": [0.0, 0.0],
                "svar": [0.0, 0.0],
            }
        )
        assert wrds_loader._macro_frame_looks_like_zero_stub(df) is True
        df2 = df.copy()
        df2.loc[0, "dp"] = 0.01
        assert wrds_loader._macro_frame_looks_like_zero_stub(df2) is False
