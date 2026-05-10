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
from src.data.wrds_loader import (
    CIZ_AWARE_VARIANTS,
    CIZ_COLUMN_MAP,
    WRDSLoader,
    _rename_ciz_to_legacy,
    merge_crsp_compustat,
)


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


class TestCIZColumnMapping:
    """The CIZ -> legacy rename must cover the columns downstream code reads."""

    def test_rename_maps_known_ciz_columns(self):
        df = pd.DataFrame(
            {
                "permno":     [1, 1],
                "mthcaldt":   pd.to_datetime(["2026-01-31", "2026-02-28"]),
                "mthret":     [0.01, -0.02],
                "mthretx":    [0.01, -0.02],
                "mthprc":     [10.0, 9.8],
                "mthvol":     [1000.0, 1100.0],
                "mthcfacpr":  [1.0, 1.0],
                "mthcfacshr": [1.0, 1.0],
                "siccd":      ["3711", "3711"],
            }
        )
        out = _rename_ciz_to_legacy(df)
        assert set(out.columns) >= {
            "permno", "date", "ret", "retx", "prc", "vol",
            "cfacpr", "cfacshr", "siccd",
        }
        # Legacy names absent in CIZ source must not be invented.
        assert "mthcaldt" not in out.columns
        assert "mthret" not in out.columns

    def test_rename_is_no_op_for_non_ciz_columns(self):
        df = pd.DataFrame({"permno": [1], "date": pd.to_datetime(["2020-01-31"])})
        out = _rename_ciz_to_legacy(df)
        assert list(out.columns) == ["permno", "date"]

    def test_column_map_is_complete_for_downstream_schema(self):
        # Every legacy column the loader's docstring promises must have a
        # CIZ source — guards against accidental schema drift.
        legacy_required = {"date", "ret", "retx", "prc", "vol", "cfacpr", "cfacshr"}
        assert legacy_required.issubset(set(CIZ_COLUMN_MAP.values()))


class TestCIZSourceSelection:
    def test_default_data_source_is_legacy(self, tmp_path):
        loader = WRDSLoader(
            wrds_username="",
            cache_dir=str(tmp_path) + "/",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        assert loader.data_source == "legacy"

    def test_ciz_data_source_accepted(self, tmp_path):
        loader = WRDSLoader(
            wrds_username="",
            cache_dir=str(tmp_path) + "/",
            start_date="2026-01-01",
            end_date="2026-03-31",
            data_source="ciz",
        )
        assert loader.data_source == "ciz"

    def test_invalid_data_source_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="data_source"):
            WRDSLoader(
                wrds_username="",
                cache_dir=str(tmp_path) + "/",
                data_source="bogus",
            )

    def test_ciz_cache_path_is_distinct_from_legacy(self, tmp_path, monkeypatch):
        """Legacy and CIZ loads must not collide on the same parquet."""
        legacy = WRDSLoader(
            wrds_username="",
            cache_dir=str(tmp_path) + "/",
            start_date="2020-01-01",
            end_date="2024-12-31",
            data_source="legacy",
        )
        ciz = WRDSLoader(
            wrds_username="",
            cache_dir=str(tmp_path) + "/",
            start_date="2020-01-01",
            end_date="2026-03-31",
            data_source="ciz",
        )
        # Trigger fetch with monkeypatched _fetch_* returning empty frames so
        # the cache files land on disk; assert their paths differ.
        empty = pd.DataFrame(
            {"permno": [], "date": pd.to_datetime([]), "ret": [], "retx": [],
             "prc": [], "vol": [], "shrout": []}
        )
        monkeypatch.setattr(legacy, "_fetch_crsp_monthly_legacy", lambda: empty.copy())
        monkeypatch.setattr(ciz, "_fetch_crsp_monthly_ciz", lambda: empty.copy())
        legacy.get_crsp_monthly()
        ciz.get_crsp_monthly()
        legacy_path = legacy._cache_path("crsp_monthly_2020_2024")
        ciz_path = ciz._cache_path("crsp_monthly_ciz_2020_2026")
        assert legacy_path.exists()
        assert ciz_path.exists()
        assert legacy_path != ciz_path

    def test_extended_ciz_2026_is_in_ciz_aware_variants(self):
        assert "extended_ciz_2026" in CIZ_AWARE_VARIANTS


class TestCIZRouter:
    """get_crsp_monthly must dispatch on data_source without touching WRDS."""

    def test_get_crsp_monthly_routes_to_ciz_fetcher(self, tmp_path, monkeypatch):
        loader = WRDSLoader(
            wrds_username="",
            cache_dir=str(tmp_path) + "/",
            start_date="2026-01-01",
            end_date="2026-03-31",
            data_source="ciz",
        )
        called = {"ciz": 0, "legacy": 0}

        def fake_ciz():
            called["ciz"] += 1
            return pd.DataFrame(
                {"permno": [1], "date": pd.to_datetime(["2026-01-31"]),
                 "ret": [0.01], "retx": [0.01], "prc": [10.0], "vol": [100.0],
                 "shrout": [1000.0]}
            )

        def fake_legacy():
            called["legacy"] += 1
            return pd.DataFrame()

        monkeypatch.setattr(loader, "_fetch_crsp_monthly_ciz", fake_ciz)
        monkeypatch.setattr(loader, "_fetch_crsp_monthly_legacy", fake_legacy)

        out = loader.get_crsp_monthly()
        assert called == {"ciz": 1, "legacy": 0}
        assert "date" in out.columns
        assert len(out) == 1

    def test_get_crsp_monthly_routes_to_legacy_fetcher(self, tmp_path, monkeypatch):
        loader = WRDSLoader(
            wrds_username="",
            cache_dir=str(tmp_path) + "/",
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
        called = {"ciz": 0, "legacy": 0}

        def fake_ciz():
            called["ciz"] += 1
            return pd.DataFrame()

        def fake_legacy():
            called["legacy"] += 1
            return pd.DataFrame(
                {"permno": [1], "date": pd.to_datetime(["2024-12-31"]),
                 "ret": [0.01], "retx": [0.01], "prc": [10.0], "vol": [100.0],
                 "shrout": [1000.0]}
            )

        monkeypatch.setattr(loader, "_fetch_crsp_monthly_ciz", fake_ciz)
        monkeypatch.setattr(loader, "_fetch_crsp_monthly_legacy", fake_legacy)

        loader.get_crsp_monthly()
        assert called == {"ciz": 0, "legacy": 1}
