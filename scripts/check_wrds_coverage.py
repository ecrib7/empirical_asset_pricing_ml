"""
scripts/check_wrds_coverage.py
------------------------------
Query WRDS for the latest available date in each table we depend on
and persist a coverage manifest under ``outputs/data_coverage/``.

Tables checked (legacy CRSP + CIZ/v2):
  - crsp.msf                       (legacy monthly stock file: max date)
  - crsp.msenames                  (legacy name history: max namedt, max nameendt)
  - crsp.msf_v2                    (CIZ monthly stock file: max mthcaldt)
  - crsp.stkmthsecuritydata        (CIZ monthly security data: max mthcaldt)
  - crsp_q_stock.msf_v2            (quarterly-refreshed CIZ msf_v2: max mthcaldt)
  - crsp_q_stock.stkmthsecuritydata(quarterly-refreshed CIZ stkmthsecuritydata)
  - comp.funda                     (annual fundamentals: max datadate)
  - comp.fundq                     (quarterly fundamentals: max datadate)
  - crsp.ccmxpf_linktable          (CCM link: max linkdt, max linkenddt)

The script is read-only on WRDS, never logs the username, and writes:
  - outputs/data_coverage/coverage_<UTC-timestamp>.json
  - outputs/data_coverage/coverage_latest.json   (symlink-style copy)

Real-data endpoint computation
------------------------------
The user's WRDS subscription receives the legacy ``crsp.msf`` only through
2024-12-31, while the CIZ/v2 monthly tables (especially the
quarterly-refreshed ``crsp_q_stock`` schema) extend further. We therefore
compute two endpoints and prefer the CIZ one when present:

  - ``legacy_real_data_end`` = min(crsp.msf.max_date,
                                   crsp.msenames.max_nameendt)
  - ``ciz_real_data_end``    = max available CIZ monthly endpoint, in
                               preference order
                                 crsp_q_stock.stkmthsecuritydata,
                                 crsp_q_stock.msf_v2,
                                 crsp.stkmthsecuritydata,
                                 crsp.msf_v2.
  - ``real_data_end``        = ``ciz_real_data_end`` when present,
                               else ``legacy_real_data_end``.
  - ``synthetic_start``      = next month-end strictly after ``real_data_end``.

Anything strictly after ``real_data_end`` is synthetic.

Usage
-----
    python scripts/check_wrds_coverage.py --wrds-username <user>
    python scripts/check_wrds_coverage.py --dry-run            # no WRDS call

The ``--dry-run`` flag short-circuits the WRDS connection and writes a
manifest from cached/known values (the ones observed 2026-05-10). It is
intended for environments without WRDS credentials and for CI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow ``python scripts/check_wrds_coverage.py`` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("wrds_coverage")

# Last manually verified WRDS coverage (used by --dry-run).
# Observed by the user on 2026-05-10.
KNOWN_COVERAGE = {
    "crsp.msf":                       {"max_date":      "2024-12-31"},
    "crsp.msenames":                  {"max_namedt":    "2024-12-31",
                                       "max_nameendt":  "2024-12-31"},
    "crsp.msf_v2":                    {"max_mthcaldt":  "2025-12-31"},
    "crsp.stkmthsecuritydata":        {"max_mthcaldt":  "2025-12-31"},
    "crsp_q_stock.msf_v2":            {"max_mthcaldt":  "2026-03-31"},
    "crsp_q_stock.stkmthsecuritydata":{"max_mthcaldt":  "2026-03-31"},
    "comp.funda":                     {"max_datadate":  "2026-04-30"},
    "comp.fundq":                     {"max_datadate":  "2026-04-30"},
    "crsp.ccmxpf_linktable":          {"max_linkdt":    "2025-12-31",
                                       "max_linkenddt": "2026-01-30"},
}

QUERIES = {
    "crsp.msf":                       "SELECT MAX(date) AS max_date FROM crsp.msf",
    "crsp.msenames":                  (
        "SELECT MAX(namedt) AS max_namedt, "
        "MAX(nameendt) AS max_nameendt FROM crsp.msenames"
    ),
    "crsp.msf_v2":                    (
        "SELECT MAX(mthcaldt) AS max_mthcaldt FROM crsp.msf_v2"
    ),
    "crsp.stkmthsecuritydata":        (
        "SELECT MAX(mthcaldt) AS max_mthcaldt FROM crsp.stkmthsecuritydata"
    ),
    "crsp_q_stock.msf_v2":            (
        "SELECT MAX(mthcaldt) AS max_mthcaldt FROM crsp_q_stock.msf_v2"
    ),
    "crsp_q_stock.stkmthsecuritydata":(
        "SELECT MAX(mthcaldt) AS max_mthcaldt FROM crsp_q_stock.stkmthsecuritydata"
    ),
    "comp.funda":                     "SELECT MAX(datadate) AS max_datadate FROM comp.funda",
    "comp.fundq":                     "SELECT MAX(datadate) AS max_datadate FROM comp.fundq",
    "crsp.ccmxpf_linktable":          (
        "SELECT MAX(linkdt) AS max_linkdt, "
        "MAX(linkenddt) AS max_linkenddt FROM crsp.ccmxpf_linktable"
    ),
}

# Order in which CIZ endpoints are preferred when computing the
# CIZ real-data end. The quarterly-refreshed ``crsp_q_stock`` schema
# typically reaches the furthest, so we check it first.
_CIZ_PREFERENCE = (
    ("crsp_q_stock.stkmthsecuritydata", "max_mthcaldt"),
    ("crsp_q_stock.msf_v2",             "max_mthcaldt"),
    ("crsp.stkmthsecuritydata",         "max_mthcaldt"),
    ("crsp.msf_v2",                     "max_mthcaldt"),
)


def _query_live(username: str | None) -> dict[str, dict[str, str]]:
    import wrds  # imported lazily so --dry-run works without the package
    import pandas as pd

    db = wrds.Connection(wrds_username=username or os.environ.get("WRDS_USERNAME"))
    out: dict[str, dict[str, str]] = {}
    try:
        for table, sql in QUERIES.items():
            try:
                row = db.raw_sql(sql).iloc[0].to_dict()
            except Exception as exc:  # subscription may not include every table
                logger.warning("Skipping %s — query failed: %s", table, exc)
                continue
            out[table] = {
                k: (pd.Timestamp(v).strftime("%Y-%m-%d") if v is not None else None)
                for k, v in row.items()
            }
    finally:
        db.close()
    return out


def _compute_legacy_real_data_end(coverage: dict[str, dict[str, str]]) -> str | None:
    msf = coverage.get("crsp.msf", {}).get("max_date")
    msenames = coverage.get("crsp.msenames", {}).get("max_nameendt")
    if msf is None and msenames is None:
        return None
    if msf is None:
        return msenames
    if msenames is None:
        return msf
    return min(msf, msenames)


def _compute_ciz_real_data_end(coverage: dict[str, dict[str, str]]) -> str | None:
    """Pick the furthest available CIZ monthly endpoint, preferring the
    quarterly-refreshed ``crsp_q_stock`` schema. Returns None if no CIZ
    table is present in ``coverage``."""
    candidates: list[str] = []
    for table, key in _CIZ_PREFERENCE:
        val = coverage.get(table, {}).get(key)
        if val:
            candidates.append(val)
    if not candidates:
        return None
    # Pick the latest (max) date among present CIZ endpoints — the
    # preference order above only matters as documentation; once we have
    # any CIZ value, we take the most extended one.
    return max(candidates)


def _next_month_end(date_str: str) -> str:
    import pandas as pd
    ts = pd.Timestamp(date_str)
    # Step forward into the next month, then snap to its month-end.
    nxt = (ts + pd.offsets.MonthBegin(1)) + pd.offsets.MonthEnd(0)
    return nxt.strftime("%Y-%m-%d")


def build_manifest(coverage: dict[str, Any]) -> dict[str, Any]:
    legacy_end = _compute_legacy_real_data_end(coverage)
    ciz_end = _compute_ciz_real_data_end(coverage)
    real_end = ciz_end if ciz_end is not None else legacy_end
    if real_end is None:
        raise ValueError(
            "Neither legacy nor CIZ CRSP coverage is present — cannot "
            "compute real_data_end."
        )
    synthetic_start = _next_month_end(real_end)
    return {
        "generated_at_utc":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tables":               coverage,
        "legacy_real_data_end": legacy_end,
        "ciz_real_data_end":    ciz_end,
        "real_data_end":        real_end,
        "synthetic_start":      synthetic_start,
        "notes": (
            "real_data_end prefers the furthest CIZ/v2 monthly endpoint "
            "(crsp_q_stock.stkmthsecuritydata / crsp_q_stock.msf_v2 / "
            "crsp.stkmthsecuritydata / crsp.msf_v2) and falls back to "
            "min(crsp.msf.max_date, crsp.msenames.max_nameendt) when no CIZ "
            "table is available. Dates strictly after real_data_end must "
            "be treated as synthetic. synthetic_start is the next "
            "month-end after real_data_end."
        ),
    }


def write_manifest(manifest: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = manifest["generated_at_utc"].replace(":", "").replace("-", "")
    target = out_dir / f"coverage_{stamp}.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out_dir / "coverage_latest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrds-username", default=None,
                        help="WRDS account name (falls back to $WRDS_USERNAME)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip WRDS, use last-known verified coverage")
    parser.add_argument("--out-dir", default="outputs/data_coverage",
                        help="Directory to write the coverage manifest")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dry_run:
        logger.info("Dry-run: using last-verified KNOWN_COVERAGE (no WRDS call).")
        coverage = KNOWN_COVERAGE
    else:
        logger.info("Querying WRDS for max-date coverage (read-only).")
        coverage = _query_live(args.wrds_username)

    manifest = build_manifest(coverage)
    target = write_manifest(manifest, Path(args.out_dir))
    logger.info(
        "Wrote %s (real_data_end=%s, legacy=%s, ciz=%s)",
        target,
        manifest["real_data_end"],
        manifest["legacy_real_data_end"],
        manifest["ciz_real_data_end"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
