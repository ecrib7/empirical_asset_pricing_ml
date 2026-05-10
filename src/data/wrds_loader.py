"""
data/wrds_loader.py
-------------------
Connects to WRDS and pulls:
  • CRSP monthly stock file (returns, prices, shares, volume, bid-ask)
  • Compustat annual fundamentals (for accounting characteristics)
  • Compustat quarterly fundamentals
  • Welch & Goyal (2008) macro predictors

Usage
-----
    from src.data.wrds_loader import WRDSLoader
    loader = WRDSLoader(wrds_username="your_username")
    crsp   = loader.get_crsp_monthly()
    comp_a = loader.get_compustat_annual()
    comp_q = loader.get_compustat_quarterly()
    macro  = loader.get_macro_predictors(goyal_csv_path="PredictorData2023.xlsx")
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import wrds
    HAS_WRDS = True
except ImportError:
    HAS_WRDS = False
    warnings.warn("wrds package not installed. Install with: pip install wrds")

from src.config import FREQ_MONTH_END, MACRO_VARS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  CIZ/v2 column mapping (CRSP "common stock file" 2024+ schema)
#  ----------------------------------------------------------------------------
#  The CIZ schema renames the legacy crsp.msf columns. Downstream code in this
#  repo expects the legacy names (date, ret, retx, prc, vol, shrout, …), so we
#  translate at load time. Only columns we actually use are listed.
#
#  legacy crsp.msf      ↔   CIZ crsp.msf_v2 / stkmthsecuritydata
#  ─────────────────────────────────────────────────────────────────
#  date                 ↔   mthcaldt
#  ret                  ↔   mthret
#  retx                 ↔   mthretx
#  prc                  ↔   mthprc
#  vol                  ↔   mthvol
#  shrout               ↔   shrout            (same name)
#  cfacpr / cfacshr     ↔   mthcfacpr / mthcfacshr
#  bid / ask            ↔   mthbid / mthask
#  siccd / shrcd        ↔   siccd / sharetype (sharetype carries a code list,
#                                              but tests below only verify the
#                                              column-name mapping)
# ─────────────────────────────────────────────────────────────────────────────
CIZ_COLUMN_MAP: dict[str, str] = {
    "mthcaldt":     "date",
    "mthret":       "ret",
    "mthretx":      "retx",
    "mthprc":       "prc",
    "mthvol":       "vol",
    "mthbid":       "bid",
    "mthask":       "ask",
    "mthcfacpr":    "cfacpr",
    "mthcfacshr":   "cfacshr",
    # CIZ alias variants observed across WRDS subscriptions: some snapshots
    # expose only the cumulative-factor names (mthcumfacpr / mthcumfacshr)
    # rather than the period-factor names. Both map to the same legacy
    # adjustment columns; whichever is present in the source table wins.
    "mthcumfacpr":  "cfacpr",
    "mthcumfacshr": "cfacshr",
    # mthcap (market cap, in $) is CIZ's pre-computed market equity. We
    # surface it as "me" so downstream code that already has a ``me``
    # column from the legacy path keeps working.
    "mthcap":       "me",
}

# Optional columns we project from CIZ tables in addition to the required
# minimum (permno, mthcaldt, mthret, mthprc). Each is included in the SELECT
# only if information_schema confirms it exists on the chosen table.
_CIZ_OPTIONAL_SELECT_COLUMNS: tuple[str, ...] = (
    "mthretx", "mthvol", "mthbid", "mthask",
    "shrout", "mthcap",
    "mthcfacpr", "mthcfacshr", "mthcumfacpr", "mthcumfacshr",
    "siccd",
    "primaryexch", "sharetype", "securitytype", "issuertype",
)

# Optional WHERE-clause filters on CIZ tables. Each predicate is added only
# if its column is present in information_schema; missing columns trigger a
# logged warning and the predicate is skipped instead of failing the query.
_CIZ_OPTIONAL_FILTERS: tuple[tuple[str, str], ...] = (
    ("sharetype",    "sharetype = 'NS'"),
    ("securitytype", "securitytype = 'EQTY'"),
    ("issuertype",   "issuertype = 'CORP'"),
    ("primaryexch",  "primaryexch IN ('N','A','Q')"),
)

# Required minimum columns. If any of these is missing on a candidate table
# we skip the candidate and try the next one in the preference list.
_CIZ_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"permno", "mthcaldt", "mthret", "mthprc"}
)

# Variants whose data_end exceeds LEGACY_REAL_DATA_END (2024-12-31) require
# a CIZ-aware loader. Anything else stays on legacy crsp.msf.
CIZ_AWARE_VARIANTS: frozenset[str] = frozenset({"extended_ciz_2026", "post2016_ciz"})

# CIZ source preference: furthest endpoint first (matches
# scripts/check_wrds_coverage.py::_CIZ_PREFERENCE).
_CIZ_SOURCE_PREFERENCE: tuple[str, ...] = (
    "crsp_q_stock.stkmthsecuritydata",
    "crsp_q_stock.msf_v2",
    "crsp.stkmthsecuritydata",
    "crsp.msf_v2",
)

# Tables that carry a historical (permno, siccd, namedt, nameendt) view in
# the CIZ schema. Used to enrich the monthly panel with siccd when the
# chosen monthly table doesn't expose it directly. Tried in order; the
# first table whose schema looks usable wins.
_CIZ_SICCD_SOURCE_PREFERENCE: tuple[str, ...] = (
    "crsp_q_stock.stksecurityinfohist",
    "crsp_q_stock.stocknames_v2",
    "crsp_q_stock.stksecurityinfohdr",
    "crsp.stksecurityinfohist",
    "crsp.stocknames_v2",
    "crsp.stksecurityinfohdr",
)

# Candidate column names across CIZ siccd-source tables. We accept the
# first one present.
_CIZ_SICCD_DATE_START_CANDIDATES: tuple[str, ...] = (
    "secinfostartdt", "namedt", "secinfodtbeg", "histdt",
)
_CIZ_SICCD_DATE_END_CANDIDATES: tuple[str, ...] = (
    "secinfoenddt", "nameenddt", "secinfodtend",
)


def _rename_ciz_to_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename CIZ columns to the legacy schema downstream code expects.

    Only known CIZ columns are renamed; everything else is left untouched so
    that callers can opt into additional CIZ-only fields without surprises.

    If a row contains both a "primary" alias and a fallback alias mapping to
    the same legacy column (e.g. both ``mthcfacpr`` and ``mthcumfacpr``),
    the primary wins — the fallback is dropped before renaming so we never
    produce duplicate column labels.
    """
    # Resolve alias collisions: when two CIZ source columns map to the same
    # legacy name, prefer the period-factor name (mthcfacpr / mthcfacshr)
    # over the cumulative-factor alias.
    df = df.copy()
    if "mthcfacpr" in df.columns and "mthcumfacpr" in df.columns:
        df = df.drop(columns=["mthcumfacpr"])
    if "mthcfacshr" in df.columns and "mthcumfacshr" in df.columns:
        df = df.drop(columns=["mthcumfacshr"])
    present = {src: dst for src, dst in CIZ_COLUMN_MAP.items() if src in df.columns}
    return df.rename(columns=present)


def _ciz_table_columns(db, table: str) -> set[str]:
    """
    Return the set of column names available for a CIZ table, queried via
    information_schema. ``table`` is "schema.table"; an unqualified name is
    treated as falling in the default schema.
    """
    if "." in table:
        schema, name = table.split(".", 1)
    else:
        schema, name = "public", table
    sql = (
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_schema = '{schema}' AND table_name = '{name}'"
    )
    df = db.raw_sql(sql)
    if df is None or len(df) == 0:
        return set()
    return {str(c).lower() for c in df["column_name"].tolist()}


def _pick_first_present(candidates: tuple[str, ...], cols: set[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _build_ciz_siccd_sql(table: str, cols: set[str]) -> Optional[str]:
    """
    Build a SELECT against a CIZ siccd-source table that returns one row
    per (permno, validity-window) with columns ``permno, siccd, dt_start,
    dt_end``. Returns None if the table lacks ``permno`` or ``siccd``.

    Validity-window columns are best-effort: if the table only carries a
    single ``namedt``/``secinfostartdt`` we still emit it and treat the end
    as open (NULL ↔ far future at join time). If neither boundary is
    present (e.g. a header-only table), we project NULLs and fall back to
    a permno-only join (most-recent siccd per permno).
    """
    if "permno" not in cols or "siccd" not in cols:
        return None
    start_col = _pick_first_present(_CIZ_SICCD_DATE_START_CANDIDATES, cols)
    end_col = _pick_first_present(_CIZ_SICCD_DATE_END_CANDIDATES, cols)
    start_sql = f"{start_col} AS dt_start" if start_col else "NULL::date AS dt_start"
    end_sql = f"{end_col} AS dt_end" if end_col else "NULL::date AS dt_end"
    return (
        f"SELECT permno, siccd, {start_sql}, {end_sql} "
        f"FROM {table} WHERE siccd IS NOT NULL"
    )


def _attach_siccd_from_history(panel: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """
    Attach ``siccd`` to ``panel`` (which already carries ``permno`` and
    ``date``) using a (permno, namedt..nameenddt) history. Open-ended
    windows (NaT in dt_end) are treated as +infinity. If neither dt_start
    nor dt_end is populated for a permno, the most-recent record wins.

    Pure pandas — no WRDS calls — so it is straightforward to unit-test.
    """
    if hist is None or hist.empty:
        return panel
    if "permno" not in panel.columns or "date" not in panel.columns:
        return panel
    h = hist.copy()
    h["permno"] = pd.to_numeric(h["permno"], errors="coerce").astype("Int64")
    h = h.dropna(subset=["permno", "siccd"])
    h["dt_start"] = pd.to_datetime(h.get("dt_start"), errors="coerce")
    h["dt_end"] = pd.to_datetime(h.get("dt_end"), errors="coerce")
    h["dt_start"] = h["dt_start"].fillna(pd.Timestamp("1900-01-01"))
    h["dt_end"] = h["dt_end"].fillna(pd.Timestamp("2099-12-31"))

    p = panel.copy()
    p["permno"] = pd.to_numeric(p["permno"], errors="coerce").astype("Int64")
    p["date"] = pd.to_datetime(p["date"])
    # Drop the panel's siccd before the cross-join to avoid suffix collisions
    # (panel-side NA must not mask history-side hits). We re-merge the
    # original siccd back at the end so existing values win over history.
    p_keys = p[["permno", "date"]].reset_index()
    merged = p_keys.merge(
        h[["permno", "siccd", "dt_start", "dt_end"]],
        on="permno",
        how="left",
    )
    in_window = (
        (merged["date"] >= merged["dt_start"])
        & (merged["date"] <= merged["dt_end"])
    )
    merged.loc[~in_window, "siccd"] = np.nan
    # Pick the latest valid siccd per (original index): orders by dt_start
    # so deterministic when histories overlap.
    merged = merged.sort_values(["index", "dt_start"])
    sic = (
        merged.dropna(subset=["siccd"])
        .drop_duplicates(subset=["index"], keep="last")
        .set_index("index")["siccd"]
    )
    out = panel.copy()
    if "siccd" in out.columns:
        out["siccd"] = out["siccd"].where(out["siccd"].notna(), out.index.map(sic))
    else:
        out["siccd"] = out.index.map(sic)
    return out


def _build_ciz_msf_sql(
    table: str,
    columns: set[str],
    start_date: str,
    end_date: str,
) -> str:
    """
    Build a schema-aware SELECT for ``table`` given the columns the table
    actually exposes (per information_schema). Optional columns and filters
    are dropped silently when absent; missing optional filter columns emit a
    warning. Required columns are assumed already validated by the caller.
    """
    select_cols = ["permno", "mthcaldt", "mthret", "mthprc"]
    for col in _CIZ_OPTIONAL_SELECT_COLUMNS:
        if col in columns and col not in select_cols:
            select_cols.append(col)

    where_clauses = [f"mthcaldt BETWEEN '{start_date}' AND '{end_date}'"]
    for col, predicate in _CIZ_OPTIONAL_FILTERS:
        if col in columns:
            where_clauses.append(predicate)
        else:
            logger.warning(
                "CIZ loader: %s lacks optional filter column '%s'; "
                "skipping predicate %r.",
                table, col, predicate,
            )

    select_sql = ", ".join(select_cols)
    where_sql = " AND ".join(where_clauses)
    return f"SELECT {select_sql} FROM {table} WHERE {where_sql}"


def _macro_frame_looks_like_zero_stub(df: pd.DataFrame) -> bool:
    """
    Detect legacy all-zero synthetic macro files (older code cached stubs
    to the same path as real macro data).
    """
    cols = list(MACRO_VARS)
    if len(df) == 0 or not all(c in df.columns for c in cols):
        return False
    return float(df[cols].fillna(0.0).abs().sum().sum()) == 0.0


class WRDSLoader:
    """
    Handles all WRDS database connections and raw data extraction.
    Results are cached locally as Parquet files to avoid redundant queries.
    """

    def __init__(
        self,
        wrds_username: Optional[str] = None,
        cache_dir: str = "data/cache/",
        start_date: str = "1957-01-01",
        end_date: str = "2016-12-31",
        data_source: str = "legacy",
    ):
        """
        Parameters
        ----------
        data_source : {"legacy", "ciz"}
            Which CRSP monthly schema to use.

            * ``"legacy"`` (default) — pull from ``crsp.msf`` + ``crsp.msenames``.
              Reproduces the existing behaviour for the ``paper``, ``improved``
              and ``extended_2024`` variants.
            * ``"ciz"`` — pull from the CIZ/v2 monthly tables, preferring
              ``crsp_q_stock.stkmthsecuritydata`` → ``crsp_q_stock.msf_v2`` →
              ``crsp.stkmthsecuritydata`` → ``crsp.msf_v2`` (first that
              succeeds wins). Columns are renamed to the legacy schema so
              downstream code is unchanged. Used by the
              ``extended_ciz_2026`` variant.
        """
        self.username  = wrds_username or os.environ.get("WRDS_USERNAME", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.end_date   = end_date
        if data_source not in {"legacy", "ciz"}:
            raise ValueError(
                f"data_source must be 'legacy' or 'ciz', got {data_source!r}"
            )
        self.data_source = data_source
        self._db: Optional["wrds.Connection"] = None

    # ─── connection ───────────────────────────────────────────────────────
    def _connect(self):
        if not HAS_WRDS:
            raise ImportError("Install wrds: pip install wrds")
        if self._db is None:
            logger.info("Connecting to WRDS...")
            self._db = wrds.Connection(wrds_username=self.username)
        return self._db

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    # ─── cache helpers ─────────────────────────────────────────────────────
    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.parquet"

    def _load_or_fetch(self, name: str, fetch_fn) -> pd.DataFrame:
        path = self._cache_path(name)
        if path.exists():
            logger.info(f"Loading cached {name}...")
            return pd.read_parquet(path)
        logger.info(f"Fetching {name} from WRDS...")
        df = fetch_fn()
        df.to_parquet(path, index=False)
        logger.info(f"Cached {name} → {path}")
        return df

    # ─── CRSP Monthly ──────────────────────────────────────────────────────
    def get_crsp_monthly(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Pull CRSP monthly stock file.

        For ``data_source="legacy"`` uses ``crsp.msf`` + ``crsp.msenames``
        (max date 2024-12-31 on the user's WRDS subscription).
        For ``data_source="ciz"`` prefers the CIZ/v2 monthly tables, which
        extend further (up to 2026-03-31 via ``crsp_q_stock.*``); CIZ
        columns are mapped back to the legacy schema so downstream code is
        unchanged.

        Columns returned (legacy schema)
        --------------------------------
        permno, date, ret, retx, shrout, prc, vol, bid, ask,
        siccd, exchcd, shrcd, cfacpr, cfacshr
        """
        suffix = "_ciz" if self.data_source == "ciz" else ""
        cache_name = (
            f"crsp_monthly{suffix}_"
            f"{self.start_date[:4]}_{self.end_date[:4]}"
        )
        if force_refresh:
            self._cache_path(cache_name).unlink(missing_ok=True)

        if self.data_source == "ciz":
            return self._load_or_fetch(cache_name, self._fetch_crsp_monthly_ciz)
        return self._load_or_fetch(cache_name, self._fetch_crsp_monthly_legacy)

    # ─── CRSP Monthly: legacy crsp.msf path ────────────────────────────────
    def _fetch_crsp_monthly_legacy(self) -> pd.DataFrame:
        db = self._connect()
        # Main monthly security file
        msf = db.raw_sql(f"""
            SELECT a.permno, a.date, a.ret, a.retx,
                   a.shrout, a.prc, a.vol,
                   a.bid, a.ask,
                   b.siccd, b.exchcd, b.shrcd
            FROM crsp.msf  AS a
            JOIN crsp.msenames AS b
              ON a.permno = b.permno
             AND b.namedt  <= a.date
             AND a.date    <= b.nameendt
            WHERE a.date BETWEEN '{self.start_date}' AND '{self.end_date}'
              AND b.shrcd IN (10, 11)
              AND b.exchcd IN (1, 2, 3)
        """, date_cols=["date"])

        # Delisting returns (to avoid survivorship bias)
        msedelist = db.raw_sql(f"""
            SELECT permno, dlstdt AS date, dlret
            FROM crsp.msedelist
            WHERE dlstdt BETWEEN '{self.start_date}' AND '{self.end_date}'
              AND dlret IS NOT NULL
        """, date_cols=["date"])

        # Merge delisting returns
        msf = msf.merge(
            msedelist[["permno", "date", "dlret"]],
            on=["permno", "date"], how="left"
        )
        msf["dlret"] = msf["dlret"].fillna(0.0)
        # Adjust return for delisting
        msf["ret"] = np.where(
            msf["ret"].isna(),
            msf["dlret"],
            (1 + msf["ret"]) * (1 + msf["dlret"]) - 1
        )
        msf.drop(columns=["dlret"], inplace=True)
        msf["date"] = pd.to_datetime(msf["date"]).dt.to_period("M").dt.to_timestamp("M")
        msf["prc"] = msf["prc"].abs()   # negative price = average bid-ask
        msf["me"]  = msf["prc"] * msf["shrout"]   # market equity ($K)
        return msf.reset_index(drop=True)

    # ─── CRSP Monthly: CIZ/v2 path ─────────────────────────────────────────
    def _fetch_crsp_monthly_ciz(self) -> pd.DataFrame:
        """
        Pull the CRSP monthly file from the CIZ/v2 schema, falling back
        through the four candidate tables in preference order. Columns are
        translated to the legacy schema (mthcaldt → date, mthret → ret, …)
        so downstream merges and characteristics builders are unchanged.

        Filters mirror the legacy query: U.S. common stock listed on the
        primary three exchanges only (sharetype IN ('NS') with
        ``primaryexch`` ∈ {'N','A','Q'} and ``securitytype`` = 'EQTY' /
        ``issuertype`` = 'CORP'). Where CIZ does not surface a column we
        leave it null rather than fabricating a value.
        """
        db = self._connect()
        last_exc: Optional[Exception] = None
        msf: Optional[pd.DataFrame] = None
        chosen: Optional[str] = None
        for table in _CIZ_SOURCE_PREFERENCE:
            try:
                logger.info("CIZ loader: inspecting schema of %s", table)
                cols = _ciz_table_columns(db, table)
                if not cols:
                    logger.warning(
                        "CIZ loader: %s not visible in information_schema; "
                        "skipping.", table,
                    )
                    continue
                missing_required = _CIZ_REQUIRED_COLUMNS - cols
                if missing_required:
                    logger.warning(
                        "CIZ loader: %s missing required columns %s; "
                        "skipping.", table, sorted(missing_required),
                    )
                    continue
                sql = _build_ciz_msf_sql(table, cols, self.start_date, self.end_date)
                logger.info("CIZ loader: querying %s", table)
                msf = db.raw_sql(sql, date_cols=["mthcaldt"])
                chosen = table
                break
            except Exception as exc:  # subscription / table-missing / runtime
                last_exc = exc
                logger.warning("CIZ loader: %s unavailable (%s)", table, exc)
        if msf is None:
            raise RuntimeError(
                "No CIZ monthly table accessible on this WRDS subscription "
                f"(last error: {last_exc})."
            )
        logger.info("CIZ loader: using %s (%d rows)", chosen, len(msf))

        # CIZ → legacy column rename. Note: we keep `permno` / `permco` as-is.
        msf = _rename_ciz_to_legacy(msf)

        # Delisting return: CIZ encodes delisting inline via mthretx vs mthret.
        # The v2 stkmthsecuritydata exposes a ``mthretdt`` / ``dlstcd``-style
        # field on some snapshots; we fall back to crsp.msedelist when the
        # subscription still has it, so survivorship-bias handling matches
        # the legacy path. If msedelist is unavailable (CIZ-only subs) we
        # leave ret unmodified — the inline ``mthret`` already incorporates
        # the delisting return per CRSP's CIZ documentation.
        try:
            msedelist = db.raw_sql(f"""
                SELECT permno, dlstdt AS date, dlret
                FROM crsp.msedelist
                WHERE dlstdt BETWEEN '{self.start_date}' AND '{self.end_date}'
                  AND dlret IS NOT NULL
            """, date_cols=["date"])
            msf = msf.merge(
                msedelist[["permno", "date", "dlret"]],
                on=["permno", "date"], how="left",
            )
            msf["dlret"] = msf["dlret"].fillna(0.0)
            msf["ret"] = np.where(
                msf["ret"].isna(),
                msf["dlret"],
                (1 + msf["ret"]) * (1 + msf["dlret"]) - 1,
            )
            msf.drop(columns=["dlret"], inplace=True)
        except Exception as exc:
            logger.info(
                "CIZ loader: msedelist unavailable (%s); relying on inline "
                "CIZ delisting handling.",
                exc,
            )

        msf["date"] = (
            pd.to_datetime(msf["date"]).dt.to_period("M").dt.to_timestamp("M")
        )
        if "prc" in msf.columns:
            msf["prc"] = msf["prc"].abs()
        # Market equity: prefer CIZ's pre-computed mthcap (renamed to "me"),
        # else derive from prc*shrout. If shrout is missing but mthcap and
        # prc are present, recover shrout = me / prc so downstream code that
        # still reads ``shrout`` keeps working. CRSP's mthcap is in dollars
        # and shrout in thousands → we keep the legacy-equivalent units by
        # matching the legacy formula (prc*shrout).
        if "shrout" not in msf.columns and "me" in msf.columns and "prc" in msf.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                msf["shrout"] = np.where(
                    (msf["prc"].abs() > 0) & msf["me"].notna(),
                    msf["me"] / msf["prc"].abs(),
                    np.nan,
                )
        if "me" not in msf.columns and "prc" in msf.columns and "shrout" in msf.columns:
            msf["me"] = msf["prc"] * msf["shrout"]

        # Synthesise legacy-compatible exchcd / shrcd for any downstream
        # consumer that still reads them. Mapping is intentionally narrow
        # because the CIZ filter above already restricted the universe.
        if "primaryexch" in msf.columns:
            msf["exchcd"] = msf["primaryexch"].map(
                {"N": 1, "A": 2, "Q": 3}
            ).astype("Int64")
        if "sharetype" in msf.columns:
            msf["shrcd"] = np.where(msf["sharetype"] == "NS", 11, np.nan)

        # Enrich with siccd if the chosen monthly CIZ table didn't expose it
        # (e.g. crsp_q_stock.stkmthsecuritydata carries pricing+cap only and
        # leaves industry codes to the security-info history tables).
        needs_siccd = "siccd" not in msf.columns or msf["siccd"].isna().all()
        if needs_siccd:
            msf = self._enrich_ciz_siccd(db, msf)
        return msf.reset_index(drop=True)

    def _enrich_ciz_siccd(self, db, msf: pd.DataFrame) -> pd.DataFrame:
        """
        Best-effort enrichment of a CIZ monthly panel with ``siccd`` by
        joining a security-info history table from
        ``_CIZ_SICCD_SOURCE_PREFERENCE``. Falls through silently with a
        warning when no candidate is accessible — downstream code is
        expected to handle missing ``siccd`` gracefully.
        """
        for table in _CIZ_SICCD_SOURCE_PREFERENCE:
            try:
                cols = _ciz_table_columns(db, table)
                sql = _build_ciz_siccd_sql(table, cols)
                if sql is None:
                    continue
                logger.info("CIZ loader: enriching siccd from %s", table)
                hist = db.raw_sql(sql, date_cols=["dt_start", "dt_end"])
                if hist is None or hist.empty:
                    continue
                out = _attach_siccd_from_history(msf, hist)
                logger.info(
                    "CIZ loader: siccd populated for %d/%d rows from %s",
                    int(out["siccd"].notna().sum()) if "siccd" in out.columns else 0,
                    len(out),
                    table,
                )
                return out
            except Exception as exc:
                logger.warning(
                    "CIZ loader: siccd enrichment via %s failed (%s)",
                    table, exc,
                )
        logger.warning(
            "CIZ loader: no siccd source available; downstream characteristics "
            "will fall back to Compustat sich or NaN."
        )
        return msf

    # ─── Compustat Annual ─────────────────────────────────────────────────
    def get_compustat_annual(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Pull Compustat annual fundamentals (funda).
        Returns the key variables needed for accounting characteristics.
        """
        cache_name = f"compustat_annual_{self.start_date[:4]}_{self.end_date[:4]}"
        if force_refresh:
            self._cache_path(cache_name).unlink(missing_ok=True)

        def _fetch():
            db = self._connect()
            df = db.raw_sql(f"""
                SELECT gvkey, datadate, fyear,
                       at, lt, seq, ceq, pstk, pstkrv, pstkl, txditc, txdb,
                       revt, cogs, xsga, dp AS depr_a, xrd, capx, act, lct,
                       dltt, dlc, che, ib, ni, oiadp, sale, csho,
                       prcc_f, sich, indfmt, datafmt, popsrc, consol,
                       ajex, mkvalt, ebitda, oancf, ivncf, fincf,
                       re, dvc, dvp, txfo, pifo, pi, nopi, spi, xi, "do",
                       wcap, rect, invt, ap, lco, lo, lcox, ppent, ppegt
                FROM comp.funda
                WHERE datadate BETWEEN '{self.start_date}' AND '{self.end_date}'
                  AND indfmt  = 'INDL'
                  AND datafmt = 'STD'
                  AND popsrc  = 'D'
                  AND consol  = 'C'
                  AND at > 0
            """, date_cols=["datadate"])
            df["datadate"] = pd.to_datetime(df["datadate"])
            return df.reset_index(drop=True)

        return self._load_or_fetch(cache_name, _fetch)

    # ─── Compustat Quarterly ──────────────────────────────────────────────
    def get_compustat_quarterly(self, force_refresh: bool = False) -> pd.DataFrame:
        """Pull Compustat quarterly fundamentals (fundq)."""
        cache_name = f"compustat_quarterly_{self.start_date[:4]}_{self.end_date[:4]}"
        if force_refresh:
            self._cache_path(cache_name).unlink(missing_ok=True)

        def _fetch():
            db = self._connect()
            df = db.raw_sql(f"""
                SELECT gvkey, datadate, fqtr, fyearq,
                       atq, ltq, ceqq, req, seqq, pstkq, pstkrq,
                       ibq, niq, saleq, cogsq, xsgaq, xrdq, capsq,
                       actq, lctq, cheq, dlttq, dlcq, rectq, invtq,
                       epspxq, cshoq, prccq, ajexq, txditcq, txdbq,
                       rdq
                FROM comp.fundq
                WHERE datadate BETWEEN '{self.start_date}' AND '{self.end_date}'
                  AND indfmt  = 'INDL'
                  AND datafmt = 'STD'
                  AND popsrc  = 'D'
                  AND consol  = 'C'
            """, date_cols=["datadate"])
            df["datadate"] = pd.to_datetime(df["datadate"])
            return df.reset_index(drop=True)

        return self._load_or_fetch(cache_name, _fetch)

    # ─── CRSP/Compustat Link Table ────────────────────────────────────────
    def get_crsp_compustat_link(self, force_refresh: bool = False) -> pd.DataFrame:
        """Pull CRSP-Compustat linking table (ccmxpf_linktable)."""
        cache_name = "ccm_link"
        if force_refresh:
            self._cache_path(cache_name).unlink(missing_ok=True)

        def _fetch():
            db = self._connect()
            df = db.raw_sql("""
                SELECT gvkey, lpermno AS permno,
                       linktype, linkprim, liid,
                       linkdt, linkenddt
                FROM crsp.ccmxpf_linktable
                WHERE substr(linktype,1,1) = 'L'
                  AND linkprim IN ('P','C')
            """, date_cols=["linkdt", "linkenddt"])
            df["linkdt"]    = pd.to_datetime(df["linkdt"])
            df["linkenddt"] = pd.to_datetime(df["linkenddt"].fillna("2099-12-31"))
            return df.reset_index(drop=True)

        return self._load_or_fetch(cache_name, _fetch)

    # ─── Welch & Goyal Macro Predictors ──────────────────────────────────
    def get_macro_predictors(
        self,
        goyal_csv_path: Optional[str] = None,
        force_refresh: bool = False,
        allow_macro_stub: bool = False,
    ) -> pd.DataFrame:
        """
        Load Welch & Goyal (2008) macro predictors.

        Priority:
        1. goyal_csv_path  – local CSV downloaded from Amit Goyal's website
           (https://sites.google.com/view/agoyal145)
        2. WRDS macro table (if available)
        3. Synthetic stub (only if ``allow_macro_stub`` is True, e.g. dev/CI)

        By default (``allow_macro_stub=False``) the loader **does not** fall
        back to silent all-zero stubs: real pipelines must supply Goyal data or
        a working WRDS macro table. Set env ``GKX_ALLOW_MACRO_STUB=1`` from
        callers that intentionally need stubs.

        Returns monthly DataFrame with columns:
            date, dp, ep, bm, ntis, tbl, tms, dfy, svar
        """
        cache_name = f"macro_predictors_{self.start_date[:4]}_{self.end_date[:4]}"
        if force_refresh:
            self._cache_path(cache_name).unlink(missing_ok=True)
        path = self._cache_path(cache_name)
        if path.exists() and not force_refresh:
            df = pd.read_parquet(path)
            if (not allow_macro_stub) and _macro_frame_looks_like_zero_stub(df):
                raise RuntimeError(
                    f"Cached macro file appears to be an all-zero stub from an older "
                    f"run: {path}\n"
                    "Delete that file (or use force_refresh), supply --goyal-csv, or "
                    "set environment variable GKX_ALLOW_MACRO_STUB=1 to allow stubs."
                )
            return df

        # ── Option 1: user-provided CSV ──
        if goyal_csv_path and Path(goyal_csv_path).exists():
            df = self._parse_goyal_csv(goyal_csv_path)
            df.to_parquet(path, index=False)
            return df

        # ── Option 2: try WRDS predictor table ──
        try:
            df = self._fetch_macro_from_wrds()
            df.to_parquet(path, index=False)
            return df
        except Exception as e:
            logger.warning(f"Could not fetch macro from WRDS: {e}")

        # ── Option 3: synthetic stub (explicit opt-in only) ──
        if not allow_macro_stub:
            raise RuntimeError(
                "Macro predictors unavailable: no valid --goyal-csv path, and WRDS "
                "macro fetch failed (see log). Refusing to use silent all-zero stubs.\n"
                "Fix: download PredictorData2023.xlsx from Amit Goyal's site and pass "
                "--goyal-csv, fix WRDS access to goyal.macro_predictors, or for "
                "development-only set environment variable GKX_ALLOW_MACRO_STUB=1."
            )
        logger.warning(
            "allow_macro_stub=True: using synthetic macro predictor stubs (all zeros). "
            "Not valid for research replication."
        )
        dates = pd.date_range(self.start_date, self.end_date, freq=FREQ_MONTH_END)
        df = pd.DataFrame({"date": dates})
        for col in ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]:
            df[col] = 0.0
        df.to_parquet(path, index=False)
        return df

    def _parse_goyal_csv(self, path: str) -> pd.DataFrame:
        """Parse Goyal's PredictorData Excel/CSV file."""
        if path.endswith(".xlsx") or path.endswith(".xls"):
            raw = pd.read_excel(path)
        else:
            raw = pd.read_csv(path)
        # Goyal's file uses 'yyyymm' format
        import logging as _logging
        _logging.debug("Goyal CSV columns: %s", list(raw.columns))
        _candidates = [c for c in raw.columns if any(
            k in c.lower() for k in ("date","yyyymm","year","ym","yyyy","period","month"))]
        if _candidates:
            date_col = _candidates[0]
        else:
            date_col = raw.columns[0]
            _logging.warning("No date column found; using first column: %s", date_col)
        # Parse date robustly — Goyal file may have floats like 197001.0
        _val_str = str(raw[date_col].iloc[0]).split(".")[0]
        if len(_val_str) == 6:
            raw["date"] = pd.to_datetime(
                raw[date_col].astype(float).astype(int).astype(str), format="%Y%m")
        elif len(_val_str) == 8:
            raw["date"] = pd.to_datetime(
                raw[date_col].astype(float).astype(int).astype(str), format="%Y%m%d")
        else:
            raw["date"] = pd.to_datetime(raw[date_col], format="mixed")
        raw["date"] = raw["date"] + pd.offsets.MonthEnd(0)

        # Standardise column names (Goyal uses exact names)
        rename_map = {
            "D/P": "dp", "E/P": "ep", "B/M": "bm",
            "NTIS": "ntis", "Rfree": "tbl",
            "TMS": "tms", "DFY": "dfy", "SVAR": "svar",
            "dp": "dp", "ep": "ep", "bm": "bm",
            "ntis": "ntis", "tbl": "tbl",
            "tms": "tms", "dfy": "dfy", "svar": "svar",
        }
        raw = raw.rename(columns=rename_map)
        needed = ["date"] + [c for c in ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]
                             if c in raw.columns]
        raw = raw[needed].dropna(subset=["date"])
        mask = (raw["date"] >= self.start_date) & (raw["date"] <= self.end_date)
        return raw.loc[mask].reset_index(drop=True)

    def _fetch_macro_from_wrds(self) -> pd.DataFrame:
        """Try to pull macro predictors from WRDS (Welch-Goyal dataset)."""
        db = self._connect()
        df = db.raw_sql(f"""
            SELECT date, dp, ep, bm, ntis, tbl, lty, baa, aaa, svar
            FROM goyal.macro_predictors
            WHERE date BETWEEN '{self.start_date}' AND '{self.end_date}'
        """, date_cols=["date"])
        df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
        df["tms"]  = df.get("lty", np.nan) - df["tbl"]      # term spread
        df["dfy"]  = df.get("baa", np.nan) - df.get("aaa", np.nan)  # default spread
        return df[["date", "dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Utility: merge Compustat onto CRSP via CCM link
# ─────────────────────────────────────────────────────────────────────────────
def merge_crsp_compustat(
    crsp: pd.DataFrame,
    comp: pd.DataFrame,
    link: pd.DataFrame,
    comp_date_col: str = "datadate",
    lag_months: int = 6,
) -> pd.DataFrame:
    """
    Left-join Compustat fundamentals to CRSP using the CCM link table.

    Parameters
    ----------
    crsp          : CRSP monthly panel (permno, date, …)
    comp          : Compustat fundamentals (gvkey, datadate, …)
    link          : CCM link table (gvkey, permno, linkdt, linkenddt)
    comp_date_col : date column in comp
    lag_months    : minimum publication lag (6 months for annual, 4 for quarterly)
    """
    # Attach permno to comp via link
    comp = comp.merge(link[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey", how="left")
    comp = comp.dropna(subset=["permno"])

    # Apply availability date with the publication lag
    comp["avail_date"] = comp[comp_date_col] + pd.DateOffset(months=lag_months)
    comp["avail_date"] = comp["avail_date"] + pd.offsets.MonthEnd(0)

    # For each CRSP observation, find the most recent available Compustat record
    crsp = crsp.copy()
    crsp["permno"] = crsp["permno"].astype(int)
    comp["permno"] = comp["permno"].astype(int)

    # Point-in-time join per permno.  A single merge_asof(..., by="permno") also
    # requires the ``on`` key to be globally sorted across the whole left frame,
    # which a multi-stock monthly panel does not satisfy.  Merging each permno
    # separately avoids that constraint and matches correct PIT semantics.
    right = (
        comp.drop(columns=[comp_date_col, "gvkey"], errors="ignore")
        .rename(columns={"avail_date": "date"})
    )
    left = crsp.sort_values(["permno", "date"])
    right = right.sort_values(["permno", "date"])
    fund_cols = [c for c in right.columns if c not in ("permno", "date")]

    chunks = []
    for permno, Lg in left.groupby("permno", sort=False):
        Lg = Lg.sort_values("date")
        Rg = right.loc[right["permno"] == permno].drop(columns=["permno"]).sort_values("date")
        if Rg.empty:
            out = Lg.copy()
            for c in fund_cols:
                if c not in out.columns:
                    out[c] = np.nan
            chunks.append(out)
            continue
        chunks.append(pd.merge_asof(Lg, Rg, on="date", direction="backward"))

    out = pd.concat(chunks, ignore_index=True)

    # Backfill siccd from Compustat's sich when the CRSP side did not
    # supply it (CIZ ``stkmthsecuritydata`` lacks siccd; if loader-level
    # enrichment also failed, sich is the next-best industry code).
    if "sich" in out.columns:
        if "siccd" not in out.columns:
            out["siccd"] = out["sich"]
        else:
            need = out["siccd"].isna()
            if need.any():
                out.loc[need, "siccd"] = out.loc[need, "sich"]
    return out
