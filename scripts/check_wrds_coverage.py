"""
scripts/check_wrds_coverage.py
------------------------------
Query WRDS for the latest available date in each table we depend on
and persist a coverage manifest under ``outputs/data_coverage/``.

Tables checked:
  - crsp.msf                (monthly stock file: max date)
  - crsp.msenames           (name history: max namedt, max nameendt)
  - comp.funda              (annual fundamentals: max datadate)
  - comp.fundq              (quarterly fundamentals: max datadate)
  - crsp.ccmxpf_linktable   (CCM link: max linkdt, max linkenddt)

The script is read-only on WRDS, never logs the username, and writes:
  - outputs/data_coverage/coverage_<UTC-timestamp>.json
  - outputs/data_coverage/coverage_latest.json   (symlink-style copy)

It also computes ``real_data_end = min(crsp.msf max_date, crsp.msenames
max_nameendt)`` which is the cutoff downstream variants should treat as
"real data". Anything strictly after ``real_data_end`` is synthetic.

Usage
-----
    python scripts/check_wrds_coverage.py --wrds-username <user>
    python scripts/check_wrds_coverage.py --dry-run            # no WRDS call

The ``--dry-run`` flag short-circuits the WRDS connection and writes a
manifest from cached/known values (the ones produced 2026-05-10). It is
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
KNOWN_COVERAGE = {
    "crsp.msf":              {"max_date":      "2024-12-31"},
    "crsp.msenames":         {"max_namedt":    "2024-12-31",
                              "max_nameendt":  "2024-12-31"},
    "comp.funda":            {"max_datadate":  "2026-04-30"},
    "comp.fundq":            {"max_datadate":  "2026-04-30"},
    "crsp.ccmxpf_linktable": {"max_linkdt":    "2025-12-31",
                              "max_linkenddt": "2026-01-30"},
}

QUERIES = {
    "crsp.msf":              "SELECT MAX(date) AS max_date FROM crsp.msf",
    "crsp.msenames":         (
        "SELECT MAX(namedt) AS max_namedt, "
        "MAX(nameendt) AS max_nameendt FROM crsp.msenames"
    ),
    "comp.funda":            "SELECT MAX(datadate) AS max_datadate FROM comp.funda",
    "comp.fundq":            "SELECT MAX(datadate) AS max_datadate FROM comp.fundq",
    "crsp.ccmxpf_linktable": (
        "SELECT MAX(linkdt) AS max_linkdt, "
        "MAX(linkenddt) AS max_linkenddt FROM crsp.ccmxpf_linktable"
    ),
}


def _query_live(username: str | None) -> dict[str, dict[str, str]]:
    import wrds  # imported lazily so --dry-run works without the package
    import pandas as pd

    db = wrds.Connection(wrds_username=username or os.environ.get("WRDS_USERNAME"))
    out: dict[str, dict[str, str]] = {}
    try:
        for table, sql in QUERIES.items():
            row = db.raw_sql(sql).iloc[0].to_dict()
            out[table] = {
                k: (pd.Timestamp(v).strftime("%Y-%m-%d") if v is not None else None)
                for k, v in row.items()
            }
    finally:
        db.close()
    return out


def _compute_real_data_end(coverage: dict[str, dict[str, str]]) -> str:
    msf_end       = coverage["crsp.msf"]["max_date"]
    msenames_end  = coverage["crsp.msenames"]["max_nameendt"]
    return min(msf_end, msenames_end)


def build_manifest(coverage: dict[str, Any]) -> dict[str, Any]:
    real_end = _compute_real_data_end(coverage)
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tables":           coverage,
        "real_data_end":    real_end,
        "synthetic_start":  "2025-01-31",  # next month-end after 2024-12-31
        "notes": (
            "real_data_end = min(crsp.msf.max_date, crsp.msenames.max_nameendt). "
            "Dates strictly after real_data_end must be treated as synthetic."
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
    logger.info("Wrote %s (real_data_end=%s)", target, manifest["real_data_end"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
