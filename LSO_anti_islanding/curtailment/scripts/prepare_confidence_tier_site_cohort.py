"""
Prepare the site cohort used by the local SAPN2022 curtailment workflow.

The default cohort is the SAPN2022 confidence-tier assessed site list. Keeping
this as a one-column CSV makes build_structured_local.py operate on a fixed,
explicit set of sites instead of rediscovering the final analysis cohort.
"""

import argparse
from pathlib import Path

import polars as pl
from path_config import require_local_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# This tracked script keeps the SAPN root in `local_paths.py` so the repo does
# not commit machine-specific external data locations.
DEFAULT_SAPN_ROOT = require_local_path(
    "SAPN_ROOT",
    "root folder containing `All Results/` for the confidence-tier cohort export.",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "confidence_tier_site_ids.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write the confidence-tier assessed site_id cohort CSV."
    )
    parser.add_argument("--sapn-root", type=Path, default=DEFAULT_SAPN_ROOT)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="CSV with a site_id column. Defaults to SAPN All Results assessed_sites_overall.csv.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def cohort_source_path(sapn_root, source):
    if source is not None:
        return source
    return sapn_root / "All Results" / "site_compliance" / "assessed_sites_overall.csv"


def read_site_ids(path):
    sites = (
        pl.read_csv(path)
        .select(pl.col("site_id").cast(pl.Int64))
        .drop_nulls()
        .unique()
        .sort("site_id")
    )
    if sites.is_empty():
        raise ValueError(f"No site_id values found in {path}")
    return sites


def main():
    args = parse_args()
    source = cohort_source_path(args.sapn_root, args.source)
    sites = read_site_ids(source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sites.write_csv(args.output)

    print(f"Source: {source}")
    print(f"Sites: {sites.height}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
