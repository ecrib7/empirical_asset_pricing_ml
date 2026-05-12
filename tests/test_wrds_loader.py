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
    _attach_siccd_from_history,
    _build_ciz_msf_sql,
    _build_ciz_siccd_sql,
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


class _FakeWRDSConn:
    """
    Minimal stand-in for ``wrds.Connection`` used by the CIZ schema-aware
    tests. ``raw_sql`` distinguishes information_schema introspection from
    payload queries via a per-table fixture.

    ``tables`` maps "schema.table" → {"columns": set[str], "rows": DataFrame}.
    The introspection query returns a frame with a ``column_name`` column;
    payload queries return the configured ``rows``.
    """

    def __init__(self, tables: dict[str, dict]):
        self.tables = tables
        self.last_payload_sql: str | None = None
        self.payload_calls: list[tuple[str, str]] = []  # (table, sql)

    def raw_sql(self, sql: str, date_cols=None):
        s = sql.strip()
        # information_schema.columns lookups
        if "information_schema.columns" in s:
            schema = _between(s, "table_schema = '", "'")
            name = _between(s, "table_name = '", "'")
            key = f"{schema}.{name}"
            cols = self.tables.get(key, {}).get("columns")
            if cols is None:
                return pd.DataFrame({"column_name": []})
            return pd.DataFrame({"column_name": sorted(cols)})
        # Payload SELECT … FROM <schema.table> …
        for key, spec in self.tables.items():
            if f"FROM {key}" in s:
                self.last_payload_sql = s
                self.payload_calls.append((key, s))
                return spec["rows"].copy()
        raise RuntimeError(f"Unexpected SQL: {sql}")

    def close(self):
        pass


def _between(text: str, start: str, end: str) -> str:
    i = text.index(start) + len(start)
    j = text.index(end, i)
    return text[i:j]


class TestCIZSchemaAwareSelect:
    """`_build_ciz_msf_sql` projects only the columns the table exposes."""

    def test_required_columns_only(self):
        sql = _build_ciz_msf_sql(
            "crsp_q_stock.stkmthsecuritydata",
            columns={"permno", "mthcaldt", "mthret", "mthprc"},
            start_date="2026-01-01",
            end_date="2026-03-31",
        )
        # Required columns appear; optional ones do not.
        assert "permno" in sql and "mthcaldt" in sql
        assert "mthret" in sql and "mthprc" in sql
        for absent in (
            "shrout", "mthcap", "mthcfacpr", "mthcumfacpr",
            "mthcfacshr", "mthcumfacshr", "mthvol",
        ):
            assert absent not in sql, f"{absent} unexpectedly selected"
        # All optional WHERE filters skipped because columns are absent.
        for absent in ("sharetype", "securitytype", "issuertype", "primaryexch"):
            assert absent not in sql, f"{absent} predicate unexpectedly emitted"

    def test_stkmth_without_shrout_uses_mthcap(self):
        # Real WRDS error: stkmthsecuritydata lacks `shrout`. With `mthcap`
        # available the SELECT must still succeed.
        cols = {
            "permno", "mthcaldt", "mthret", "mthretx", "mthprc",
            "mthvol", "mthcap",
            "primaryexch", "sharetype", "securitytype", "issuertype", "siccd",
        }
        sql = _build_ciz_msf_sql(
            "crsp_q_stock.stkmthsecuritydata",
            columns=cols,
            start_date="1957-01-01",
            end_date="2026-03-31",
        )
        assert "mthcap" in sql
        assert "shrout" not in sql  # absent on this table — must not be selected
        assert "sharetype = 'NS'" in sql
        assert "primaryexch IN ('N','A','Q')" in sql

    def test_msf_v2_with_cumulative_factor_alias(self):
        # Real WRDS error: msf_v2 lacks `mthcfacpr` / `mthcfacshr` but exposes
        # the cumulative-factor aliases. The SQL must select those instead.
        cols = {
            "permno", "mthcaldt", "mthret", "mthretx", "mthprc",
            "mthvol", "shrout",
            "mthcumfacpr", "mthcumfacshr",
            "primaryexch", "sharetype", "securitytype", "issuertype", "siccd",
        }
        sql = _build_ciz_msf_sql(
            "crsp_q_stock.msf_v2", cols, "2020-01-01", "2026-03-31",
        )
        assert "mthcumfacpr" in sql
        assert "mthcumfacshr" in sql
        # The period-factor names must NOT appear if absent on the table.
        assert "mthcfacpr," not in sql and " mthcfacpr " not in sql
        assert "mthcfacshr," not in sql and " mthcfacshr " not in sql

    def test_optional_filter_skipped_when_column_absent(self):
        # primaryexch and sharetype absent → predicates must be omitted.
        cols = {"permno", "mthcaldt", "mthret", "mthprc", "siccd"}
        sql = _build_ciz_msf_sql(
            "crsp.msf_v2", cols, "2026-01-01", "2026-03-31",
        )
        assert "primaryexch" not in sql
        assert "sharetype" not in sql
        # Date filter still present.
        assert "mthcaldt BETWEEN" in sql

    def test_alias_collision_drops_cumulative_when_period_present(self):
        # When both adjustment-factor names are present, both are selected
        # (the rename collision is handled at the dataframe level).
        cols = {
            "permno", "mthcaldt", "mthret", "mthprc",
            "mthcfacpr", "mthcumfacpr",
        }
        sql = _build_ciz_msf_sql("crsp.msf_v2", cols, "2026-01-01", "2026-03-31")
        assert "mthcfacpr" in sql
        assert "mthcumfacpr" in sql


class TestCIZRenameAliasCollision:
    def test_period_factor_wins_over_cumulative(self):
        df = pd.DataFrame(
            {
                "permno": [1],
                "mthcaldt": pd.to_datetime(["2026-01-31"]),
                "mthret": [0.01],
                "mthprc": [10.0],
                "mthcfacpr": [1.0],
                "mthcumfacpr": [1.5],
                "mthcfacshr": [1.0],
                "mthcumfacshr": [1.5],
            }
        )
        out = _rename_ciz_to_legacy(df)
        # Each legacy adjustment column appears exactly once and carries the
        # period-factor (mthcfac*) value, not the cumulative one.
        assert list(out.columns).count("cfacpr") == 1
        assert list(out.columns).count("cfacshr") == 1
        assert float(out["cfacpr"].iloc[0]) == 1.0
        assert float(out["cfacshr"].iloc[0]) == 1.0

    def test_cumulative_alias_used_when_period_absent(self):
        df = pd.DataFrame(
            {
                "permno": [1],
                "mthcaldt": pd.to_datetime(["2026-01-31"]),
                "mthret": [0.01],
                "mthprc": [10.0],
                "mthcumfacpr": [1.5],
                "mthcumfacshr": [1.5],
            }
        )
        out = _rename_ciz_to_legacy(df)
        assert "cfacpr" in out.columns
        assert "cfacshr" in out.columns
        assert float(out["cfacpr"].iloc[0]) == 1.5

    def test_mthcap_renamed_to_me(self):
        df = pd.DataFrame(
            {
                "permno": [1],
                "mthcaldt": pd.to_datetime(["2026-01-31"]),
                "mthret": [0.01],
                "mthprc": [10.0],
                "mthcap": [123456.0],
            }
        )
        out = _rename_ciz_to_legacy(df)
        assert "me" in out.columns
        assert float(out["me"].iloc[0]) == 123456.0


class TestCIZFetchEndToEnd:
    """End-to-end: _fetch_crsp_monthly_ciz must accept tables that are
    missing optional columns or use alias names, derive shrout from mthcap
    when absent, and not crash on optional filter columns being missing."""

    @staticmethod
    def _row(payload: dict, n: int = 1) -> pd.DataFrame:
        return pd.DataFrame({k: [v] * n for k, v in payload.items()})

    def _make_loader(self, tmp_path, fake_db):
        loader = WRDSLoader(
            wrds_username="",
            cache_dir=str(tmp_path) + "/",
            start_date="2026-01-01",
            end_date="2026-03-31",
            data_source="ciz",
        )
        loader._db = fake_db  # bypass real wrds.Connection
        # Bypass HAS_WRDS guard in _connect — the fake connection is already
        # attached, so we just want _connect to return self._db.
        loader._connect = lambda: loader._db  # type: ignore[method-assign]
        return loader

    def test_stkmth_without_shrout_succeeds_and_derives_shrout(
        self, tmp_path, monkeypatch
    ):
        # stkmthsecuritydata: no `shrout`, but `mthcap` and `mthprc` present.
        rows = self._row(
            {
                "permno": 10,
                "mthcaldt": pd.Timestamp("2026-01-31"),
                "mthret": 0.05,
                "mthretx": 0.05,
                "mthprc": 50.0,
                "mthvol": 1000.0,
                "mthcap": 500_000.0,  # → shrout = 500000 / 50 = 10000
                "primaryexch": "N",
                "sharetype": "NS",
                "securitytype": "EQTY",
                "issuertype": "CORP",
                "siccd": "3711",
            }
        )
        fake = _FakeWRDSConn(
            {
                "crsp_q_stock.stkmthsecuritydata": {
                    "columns": set(rows.columns) | {"permno"},
                    "rows": rows,
                },
            }
        )
        loader = self._make_loader(tmp_path, fake)
        # msedelist branch must fail-soft, not raise.
        df = loader._fetch_crsp_monthly_ciz()
        assert "date" in df.columns
        assert "ret" in df.columns and "prc" in df.columns
        assert "me" in df.columns
        assert "shrout" in df.columns
        # Derived shrout = me / |prc|
        assert float(df["shrout"].iloc[0]) == pytest.approx(10_000.0)
        assert float(df["me"].iloc[0]) == pytest.approx(500_000.0)

    def test_msf_v2_with_cumfac_alias_succeeds(self, tmp_path):
        rows = self._row(
            {
                "permno": 11,
                "mthcaldt": pd.Timestamp("2026-02-28"),
                "mthret": -0.01,
                "mthretx": -0.01,
                "mthprc": 25.0,
                "mthvol": 800.0,
                "shrout": 4_000.0,  # already present; shrout-derivation skipped
                "mthcumfacpr": 1.25,
                "mthcumfacshr": 1.25,
                "primaryexch": "Q",
                "sharetype": "NS",
                "securitytype": "EQTY",
                "issuertype": "CORP",
                "siccd": "7372",
            }
        )
        # First-preference table missing entirely (no columns); second
        # preference (msf_v2) succeeds.
        fake = _FakeWRDSConn(
            {
                "crsp_q_stock.stkmthsecuritydata": {"columns": set(), "rows": rows.iloc[0:0]},
                "crsp_q_stock.msf_v2": {
                    "columns": set(rows.columns) | {"permno"},
                    "rows": rows,
                },
            }
        )
        loader = self._make_loader(tmp_path, fake)
        df = loader._fetch_crsp_monthly_ciz()
        # cfacpr / cfacshr were aliased from cumulative-factor names.
        assert "cfacpr" in df.columns
        assert "cfacshr" in df.columns
        assert float(df["cfacpr"].iloc[0]) == pytest.approx(1.25)
        # me derived from prc * shrout because mthcap absent.
        assert "me" in df.columns
        assert float(df["me"].iloc[0]) == pytest.approx(25.0 * 4_000.0)

    def test_optional_filter_columns_missing_does_not_crash(self, tmp_path):
        # Bare-minimum CIZ table: no sharetype / primaryexch / securitytype /
        # issuertype. The query must still succeed without those predicates.
        rows = self._row(
            {
                "permno": 12,
                "mthcaldt": pd.Timestamp("2026-03-31"),
                "mthret": 0.02,
                "mthprc": 12.0,
                "mthcap": 240_000.0,  # → shrout = 20000
            }
        )
        fake = _FakeWRDSConn(
            {
                "crsp_q_stock.stkmthsecuritydata": {
                    "columns": set(rows.columns),
                    "rows": rows,
                },
            }
        )
        loader = self._make_loader(tmp_path, fake)
        df = loader._fetch_crsp_monthly_ciz()
        assert len(df) == 1
        assert float(df["shrout"].iloc[0]) == pytest.approx(20_000.0)
        # SQL emitted skipped the missing optional filters.
        last_sql = fake.last_payload_sql
        for absent in ("sharetype", "securitytype", "issuertype", "primaryexch"):
            assert absent not in last_sql, f"{absent} predicate emitted unexpectedly"


class TestCIZSiccdEnrichment:
    """When the chosen CIZ monthly table lacks ``siccd``, the loader must
    join a security-info history table to attach it. Downstream code keys
    industry signals off ``siccd``."""

    def test_build_ciz_siccd_sql_uses_present_window_columns(self):
        sql = _build_ciz_siccd_sql(
            "crsp_q_stock.stksecurityinfohist",
            cols={"permno", "siccd", "secinfostartdt", "secinfoenddt", "junk"},
        )
        assert sql is not None
        assert "secinfostartdt AS dt_start" in sql
        assert "secinfoenddt AS dt_end" in sql
        assert "FROM crsp_q_stock.stksecurityinfohist" in sql

    def test_build_ciz_siccd_sql_falls_back_to_namedt(self):
        sql = _build_ciz_siccd_sql(
            "crsp.stocknames_v2",
            cols={"permno", "siccd", "namedt", "nameenddt"},
        )
        assert sql is not None
        assert "namedt AS dt_start" in sql
        assert "nameenddt AS dt_end" in sql

    def test_build_ciz_siccd_sql_returns_none_without_siccd(self):
        sql = _build_ciz_siccd_sql(
            "crsp.stksecurityinfohdr",
            cols={"permno", "namedt", "nameenddt"},  # no siccd
        )
        assert sql is None

    def test_attach_siccd_from_history_uses_window(self):
        panel = pd.DataFrame(
            {
                "permno": [1, 1, 2],
                "date": pd.to_datetime(["2026-01-31", "2026-04-30", "2026-01-31"]),
                "ret": [0.01, 0.02, -0.01],
            }
        )
        hist = pd.DataFrame(
            {
                "permno": [1, 1, 2],
                "siccd": ["3711", "3714", "7372"],
                "dt_start": pd.to_datetime(["1990-01-01", "2026-03-01", "1990-01-01"]),
                "dt_end": pd.to_datetime(["2026-02-28", "2099-12-31", "2099-12-31"]),
            }
        )
        out = _attach_siccd_from_history(panel, hist)
        assert "siccd" in out.columns
        # permno 1 in Jan-26 → first window; in Apr-26 → second window
        assert out.loc[0, "siccd"] == "3711"
        assert out.loc[1, "siccd"] == "3714"
        assert out.loc[2, "siccd"] == "7372"

    def test_attach_siccd_from_history_handles_open_ended_window(self):
        panel = pd.DataFrame(
            {"permno": [42], "date": pd.to_datetime(["2026-05-31"])}
        )
        hist = pd.DataFrame(
            {
                "permno": [42],
                "siccd": ["6020"],
                "dt_start": [pd.NaT],
                "dt_end": [pd.NaT],
            }
        )
        out = _attach_siccd_from_history(panel, hist)
        assert out.loc[0, "siccd"] == "6020"

    def test_attach_siccd_preserves_existing_when_present(self):
        panel = pd.DataFrame(
            {
                "permno": [1, 2],
                "date": pd.to_datetime(["2026-01-31", "2026-01-31"]),
                "siccd": ["3711", None],
            }
        )
        hist = pd.DataFrame(
            {
                "permno": [1, 2],
                "siccd": ["9999", "7372"],  # would overwrite 1's existing if buggy
                "dt_start": pd.to_datetime(["1990-01-01", "1990-01-01"]),
                "dt_end": pd.to_datetime(["2099-12-31", "2099-12-31"]),
            }
        )
        out = _attach_siccd_from_history(panel, hist)
        # Existing siccd preserved for permno 1; permno 2 backfilled.
        assert out.loc[0, "siccd"] == "3711"
        assert out.loc[1, "siccd"] == "7372"

    def test_fetch_ciz_enriches_siccd_via_security_info_hist(self, tmp_path):
        """End-to-end: stkmthsecuritydata exposes no siccd → loader joins
        stksecurityinfohist and the returned panel carries siccd."""
        # CIZ monthly rows without siccd
        rows = pd.DataFrame(
            {
                "permno": [10, 10],
                "mthcaldt": pd.to_datetime(["2026-01-31", "2026-02-28"]),
                "mthret": [0.01, -0.02],
                "mthprc": [50.0, 49.0],
                "mthcap": [500_000.0, 490_000.0],
            }
        )
        # History table with a single window covering both months
        sec_hist = pd.DataFrame(
            {
                "permno": [10],
                "siccd": ["3711"],
                "dt_start": pd.to_datetime(["1990-01-01"]),
                "dt_end": pd.to_datetime(["2099-12-31"]),
            }
        )
        fake = _FakeWRDSConn(
            {
                "crsp_q_stock.stkmthsecuritydata": {
                    "columns": set(rows.columns),
                    "rows": rows,
                },
                "crsp_q_stock.stksecurityinfohist": {
                    "columns": {"permno", "siccd", "secinfostartdt", "secinfoenddt"},
                    "rows": sec_hist.rename(
                        columns={
                            "dt_start": "dt_start",
                            "dt_end": "dt_end",
                        }
                    ),
                },
            }
        )
        loader = WRDSLoader(
            wrds_username="",
            cache_dir=str(tmp_path) + "/",
            start_date="2026-01-01",
            end_date="2026-03-31",
            data_source="ciz",
        )
        loader._db = fake
        loader._connect = lambda: loader._db  # type: ignore[method-assign]
        df = loader._fetch_crsp_monthly_ciz()
        assert "siccd" in df.columns
        assert (df["siccd"] == "3711").all()


class TestMergeBackfillsSiccdFromSich:
    def test_merge_backfills_siccd_from_compustat_sich(self):
        link = pd.DataFrame(
            {
                "gvkey": ["G1"],
                "permno": [1],
                "linkdt": pd.to_datetime(["1990-01-01"]),
                "linkenddt": pd.to_datetime(["2099-12-31"]),
            }
        )
        comp = pd.DataFrame(
            {
                "gvkey": ["G1"],
                "datadate": pd.to_datetime(["2025-06-30"]),
                "at": [1000.0],
                "sich": [3711.0],
            }
        )
        # CRSP side has NO siccd at all (real-world CIZ stkmth case)
        crsp = pd.DataFrame(
            {
                "permno": [1, 1],
                "date": pd.to_datetime(["2026-01-31", "2026-02-28"]),
                "ret": [0.01, 0.02],
            }
        )
        merged = merge_crsp_compustat(crsp, comp, link, lag_months=6)
        assert "siccd" in merged.columns
        assert merged["siccd"].notna().all()
        assert (merged["siccd"] == 3711.0).all()

    def test_merge_preserves_existing_siccd_when_present(self):
        link = pd.DataFrame(
            {
                "gvkey": ["G1"],
                "permno": [1],
                "linkdt": pd.to_datetime(["1990-01-01"]),
                "linkenddt": pd.to_datetime(["2099-12-31"]),
            }
        )
        comp = pd.DataFrame(
            {
                "gvkey": ["G1"],
                "datadate": pd.to_datetime(["2025-06-30"]),
                "at": [1000.0],
                "sich": [9999.0],  # Compustat code that would overwrite if buggy
            }
        )
        crsp = pd.DataFrame(
            {
                "permno": [1],
                "date": pd.to_datetime(["2026-01-31"]),
                "ret": [0.01],
                "siccd": ["3711"],  # CRSP-supplied
            }
        )
        merged = merge_crsp_compustat(crsp, comp, link, lag_months=6)
        # Existing CRSP siccd not overwritten by Compustat sich.
        assert merged.loc[0, "siccd"] == "3711"


class TestIndmomPanelMissingSiccd:
    """IndustryBuilder.indmom_panel must not crash when siccd is unavailable."""

    def test_returns_nan_when_siccd_missing_and_no_sich(self):
        from src.data.characteristics import IndustryBuilder

        panel = pd.DataFrame(
            {
                "permno": [1, 1, 2, 2],
                "date": pd.to_datetime(
                    ["2026-01-31", "2026-02-28", "2026-01-31", "2026-02-28"]
                ),
                "ret": [0.01, 0.02, -0.01, 0.0],
            }
        )
        out = IndustryBuilder.indmom_panel(panel)
        assert isinstance(out, pd.Series)
        assert len(out) == len(panel)
        assert out.isna().all()

    def test_returns_nan_when_siccd_all_nan(self):
        from src.data.characteristics import IndustryBuilder

        panel = pd.DataFrame(
            {
                "permno": [1, 2],
                "date": pd.to_datetime(["2026-01-31", "2026-01-31"]),
                "ret": [0.01, -0.01],
                "siccd": [np.nan, np.nan],
            }
        )
        out = IndustryBuilder.indmom_panel(panel)
        assert out.isna().all()

    def test_falls_back_to_sich_when_siccd_missing(self):
        from src.data.characteristics import IndustryBuilder

        # Two stocks in same 2-digit SIC at the same date; with enough
        # history, indmom should return a non-null number for the cross
        # section. We just check it does NOT crash and that non-null
        # entries appear when sich provides the industry code.
        dates = pd.date_range("2024-01-31", "2026-02-28", freq="ME")
        permnos = [1, 2]
        rows = []
        for p in permnos:
            for t in dates:
                rows.append({"permno": p, "date": t, "ret": 0.01, "sich": 3711.0})
        panel = pd.DataFrame(rows)
        out = IndustryBuilder.indmom_panel(panel)
        assert isinstance(out, pd.Series)
        assert len(out) == len(panel)
        # At least the most-recent row, with full history, should be finite.
        assert out.notna().any()

    def test_unchanged_behavior_when_siccd_present(self):
        from src.data.characteristics import IndustryBuilder

        dates = pd.date_range("2024-01-31", "2026-02-28", freq="ME")
        rows = []
        for p in [1, 2]:
            for t in dates:
                rows.append(
                    {"permno": p, "date": t, "ret": 0.01, "siccd": "3711"}
                )
        panel = pd.DataFrame(rows)
        out = IndustryBuilder.indmom_panel(panel)
        assert out.notna().any()
