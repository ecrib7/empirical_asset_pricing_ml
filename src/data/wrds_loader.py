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
    ):
        self.username  = wrds_username or os.environ.get("WRDS_USERNAME", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.end_date   = end_date
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
        Pull CRSP monthly stock file (msf + msenames + msedelist).

        Columns returned
        ----------------
        permno, date, ret, retx, shrout, prc, vol, bid, ask,
        siccd, exchcd, shrcd, cfacpr, cfacshr
        """
        cache_name = f"crsp_monthly_{self.start_date[:4]}_{self.end_date[:4]}"
        if force_refresh:
            self._cache_path(cache_name).unlink(missing_ok=True)

        def _fetch():
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

        return self._load_or_fetch(cache_name, _fetch)

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
        date_col = [c for c in raw.columns if "date" in c.lower() or "yyyymm" in c.lower()][0]
        raw = raw.rename(columns={date_col: "yyyymm"})
        raw["date"] = pd.to_datetime(raw["yyyymm"].astype(str), format="%Y%m")
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

    return pd.concat(chunks, ignore_index=True)
