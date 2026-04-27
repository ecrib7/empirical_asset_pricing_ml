"""
gwz_macro.py
------------
Builds the macro predictor parquet(s) the rest of the pipeline consumes.

Two sources, two outputs:

  data/Data2024.xlsx               (or PredictorData2024.xlsx)
    -> data/cache/macro.parquet
    Columns: date, dp, ep, bm, ntis, tbl, tms, dfy, svar
    These are the 8 base Goyal-Welch (2008) predictors GKX uses.

  data/gwz_data_csv_2024.zip
    -> data/cache/macro_extra.parquet
    Columns: date + one column per extra predictor (accrul, avgcor, ...)
    These are the additional predictors introduced in Goyal-Welch-Zafirov
    (2024). Not consumed by the current pipeline; produced as a side artifact
    for future extensions.

Each GWZ csv is a wide N x N matrix where rows = forecast date and
columns = data-as-of date. We take the diagonal (data revealed exactly on the
forecast date) as the "as-of-then" time series — this is what a real-time
forecaster would have seen.

Usage from main.py:

    from pathlib import Path
    from gwz_macro import build_macro_parquet

    macro = build_macro_parquet(
        data_dir=Path('data'),
        cache_path=Path('data/cache/macro.parquet'),
        start_date='1957-01-01',
        end_date='2016-12-31',
    )
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_MACRO_COLS = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']

GWZ_ZIP_NAME = 'gwz_data_csv_2024.zip'
EXCEL_CANDIDATES = ('Data2024.xlsx', 'PredictorData2024.xlsx')


# ── shared helpers ───────────────────────────────────────────────────────────

def _parse_yyyymm(s: pd.Series) -> pd.Series:
    """Parse a Series of yyyymm or yyyymmdd values (int / float / str) -> month-end Timestamps."""
    raw = s.astype(str).str.split('.').str[0]
    sample = raw.dropna().iloc[0]
    if len(sample) == 6:
        out = pd.to_datetime(raw, format='%Y%m', errors='coerce')
    elif len(sample) == 8:
        out = pd.to_datetime(raw, format='%Y%m%d', errors='coerce')
    elif len(sample) == 4:
        out = pd.to_datetime(raw + '12', format='%Y%m', errors='coerce')  # annual -> Dec
    else:
        out = pd.to_datetime(raw, format='mixed', errors='coerce')
    return out + pd.offsets.MonthEnd(0)


# ── base predictors from Data2024.xlsx ───────────────────────────────────────

def _find_excel(data_dir: Path) -> Path:
    for name in EXCEL_CANDIDATES:
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f'No base macro Excel found in {data_dir}. Expected one of: {", ".join(EXCEL_CANDIDATES)}'
    )


def _parse_base_excel(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Parse the Goyal-Welch Excel file. The standard format has columns
    yyyymm, Index, D12, E12, b/m, tbl, lty, ntis, Rfree, infl, ltr, corpr, svar, ...
    Some recent versions ship dp / ep / bm directly; older versions require us
    to compute them from D12, E12, Index.
    """
    raw = pd.read_excel(path)

    # Normalize column names for robust lookup
    cols_lower = {c.lower().strip(): c for c in raw.columns}

    def col(*names):
        for n in names:
            if n.lower() in cols_lower:
                return cols_lower[n.lower()]
        return None

    # Date
    date_src = col('yyyymm', 'date', 'month')
    if date_src is None:
        date_src = raw.columns[0]
    out = pd.DataFrame({'date': _parse_yyyymm(raw[date_src])})

    # dp = log(D12) - log(Index); ep = log(E12) - log(Index); fall back to direct cols
    import numpy as np

    direct_dp = col('dp', 'd/p')
    direct_ep = col('ep', 'e/p')
    direct_bm = col('bm', 'b/m')

    if direct_dp is not None:
        out['dp'] = pd.to_numeric(raw[direct_dp], errors='coerce')
    else:
        d12 = col('d12'); idx = col('index')
        if d12 is None or idx is None:
            raise RuntimeError(f'Excel missing both dp and (D12, Index): {path}')
        out['dp'] = np.log(pd.to_numeric(raw[d12], errors='coerce')) - \
                   np.log(pd.to_numeric(raw[idx], errors='coerce'))

    if direct_ep is not None:
        out['ep'] = pd.to_numeric(raw[direct_ep], errors='coerce')
    else:
        e12 = col('e12'); idx = col('index')
        if e12 is None or idx is None:
            raise RuntimeError(f'Excel missing both ep and (E12, Index): {path}')
        out['ep'] = np.log(pd.to_numeric(raw[e12], errors='coerce')) - \
                   np.log(pd.to_numeric(raw[idx], errors='coerce'))

    if direct_bm is not None:
        out['bm'] = pd.to_numeric(raw[direct_bm], errors='coerce')
    else:
        raise RuntimeError(f'Excel missing bm column (looked for: bm, b/m): {path}')

    # ntis, tbl, svar — usually present directly
    for tgt, *aliases in [
        ('ntis', 'ntis'),
        ('tbl', 'tbl'),
        ('svar', 'svar'),
    ]:
        src = col(*aliases)
        if src is None:
            raise RuntimeError(f'Excel missing column {tgt} (aliases tried: {aliases}): {path}')
        out[tgt] = pd.to_numeric(raw[src], errors='coerce')

    # tms = lty - tbl; dfy = baa - aaa (or use direct cols if present)
    direct_tms = col('tms')
    direct_dfy = col('dfy')

    if direct_tms is not None:
        out['tms'] = pd.to_numeric(raw[direct_tms], errors='coerce')
    else:
        lty = col('lty')
        if lty is None:
            raise RuntimeError(f'Excel missing tms and lty (cannot derive tms): {path}')
        out['tms'] = pd.to_numeric(raw[lty], errors='coerce') - out['tbl']

    if direct_dfy is not None:
        out['dfy'] = pd.to_numeric(raw[direct_dfy], errors='coerce')
    else:
        baa = col('baa'); aaa = col('aaa')
        if baa is None or aaa is None:
            raise RuntimeError(f'Excel missing dfy and (baa, aaa): {path}')
        out['dfy'] = pd.to_numeric(raw[baa], errors='coerce') - \
                    pd.to_numeric(raw[aaa], errors='coerce')

    out = out.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    mask = (out['date'] >= pd.Timestamp(start_date)) & (out['date'] <= pd.Timestamp(end_date))
    return out.loc[mask, ['date'] + REQUIRED_MACRO_COLS].reset_index(drop=True)


# ── extra predictors from the GWZ csv zip ────────────────────────────────────

def _extract_zip(data_dir: Path) -> Optional[Path]:
    gwz_zip = data_dir / GWZ_ZIP_NAME
    if not gwz_zip.exists():
        return None
    extract_root = data_dir / 'gwz_extracted'
    extract_root.mkdir(parents=True, exist_ok=True)
    # Skip extraction if it's already been done
    if not any(extract_root.rglob('*.csv')):
        with zipfile.ZipFile(gwz_zip, 'r') as z:
            z.extractall(str(extract_root))
        logger.info(f'[gwz] extracted {gwz_zip.name} -> {extract_root}')
    return extract_root


def _wide_csv_to_diagonal_series(path: Path, name: str) -> Optional[pd.DataFrame]:
    """
    Read a GWZ wide-matrix CSV and return its diagonal as a time series.
      rows    = forecast date  (column 0)
      columns = data-as-of date
    The diagonal value [t, t] is the "real-time" reading at date t.
    Returns DataFrame['date', name].
    """
    raw = pd.read_csv(path)
    if raw.empty:
        return None

    # Row labels = first column, column labels = remaining columns
    row_labels = raw.iloc[:, 0]
    col_labels = raw.columns[1:]
    values = raw.iloc[:, 1:]

    # Frequency from suffix: _M monthly, _Q quarterly, _A annual, _S semi-annual, _D daily
    suffix = path.stem.split('_')[-1]

    row_dates = _parse_yyyymm(row_labels)
    col_dates = _parse_yyyymm(pd.Series(col_labels))

    # Build a lookup from column-date -> column position
    col_pos = {d: i for i, d in enumerate(col_dates) if pd.notna(d)}

    diag = []
    dates = []
    for i, rd in enumerate(row_dates):
        if pd.isna(rd):
            continue
        # Use the column whose date == row date (real-time / as-of-t value)
        if rd in col_pos:
            diag.append(values.iloc[i, col_pos[rd]])
            dates.append(rd)

    if not dates:
        return None

    out = pd.DataFrame({'date': dates, name: pd.to_numeric(diag, errors='coerce')})
    return out.dropna(subset=['date']).reset_index(drop=True)


def _build_extra_parquet(extract_root: Path, cache_path: Path,
                         start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    csvs = sorted(extract_root.rglob('*.csv'))
    if not csvs:
        return None

    frames = []
    for csv in csvs:
        # Strip frequency suffix from name (e.g. avgcor_M -> avgcor)
        stem = csv.stem
        name = stem.rsplit('_', 1)[0] if stem.rsplit('_', 1)[-1] in ('M', 'Q', 'A', 'S', 'D') else stem
        try:
            df = _wide_csv_to_diagonal_series(csv, name)
        except Exception as e:
            logger.warning(f'[gwz] could not parse {csv.name}: {e}')
            continue
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return None

    extra = frames[0]
    for df in frames[1:]:
        extra = extra.merge(df, on='date', how='outer')

    extra = extra.sort_values('date').reset_index(drop=True)
    mask = (extra['date'] >= pd.Timestamp(start_date)) & (extra['date'] <= pd.Timestamp(end_date))
    extra = extra.loc[mask].reset_index(drop=True)

    extra.to_parquet(cache_path, index=False)
    logger.info(f'[gwz] wrote extras parquet: {cache_path} ({len(extra)} rows, '
                f'{len(extra.columns) - 1} predictors)')
    return extra


# ── main entry point ─────────────────────────────────────────────────────────

def _macro_parquet_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:
        return False
    if not set(REQUIRED_MACRO_COLS).issubset(df.columns):
        return False
    return float(df[REQUIRED_MACRO_COLS].fillna(0.0).abs().sum().sum()) > 0.0


def build_macro_parquet(
    data_dir: Path,
    cache_path: Path,
    start_date: str = '1957-01-01',
    end_date: str = '2016-12-31',
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build the base macro parquet (8 GW predictors) from Data2024.xlsx.
    Also build a side-car extras parquet from the GWZ zip if present.
    Returns the base DataFrame.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    # ─ Base predictors (required) ─────────────────────────────────────────
    if not force_refresh and _macro_parquet_is_valid(cache_path):
        logger.info(f'[gwz] base cache OK: {cache_path}')
        macro = pd.read_parquet(cache_path)
    else:
        excel = _find_excel(data_dir)
        macro = _parse_base_excel(excel, start_date, end_date)
        macro.to_parquet(cache_path, index=False)
        logger.info(f'[gwz] wrote base parquet: {cache_path} ({len(macro)} rows)')

    # ─ Extras (required) ──────────────────────────────────────────────────
    extras_cache = cache_path.parent / 'macro_extra.parquet'
    if force_refresh or not extras_cache.exists():
        extract_root = _extract_zip(data_dir)
        if extract_root is None:
            raise FileNotFoundError(
                f'GWZ zip not found: expected {data_dir / GWZ_ZIP_NAME}'
            )
        extras = _build_extra_parquet(extract_root, extras_cache, start_date, end_date)
        if extras is None or extras.empty:
            raise RuntimeError(
                f'GWZ zip extracted at {extract_root} but no usable predictor '
                f'CSVs were parsed. Check the zip contents.'
            )

    return macro
