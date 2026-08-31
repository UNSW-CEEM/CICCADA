"""Audit duplicates, build canonical phase parquet, and validate it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dnsp_analysis.canonical import build_canonical_phase  # noqa: E402
from dnsp_analysis.config import load_config  # noqa: E402
from dnsp_analysis.duplicates import run_duplicate_audit  # noqa: E402
from dnsp_analysis.metadata import metadata_output_path  # noqa: E402
from dnsp_analysis.validation import validate_canonical_phase  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--month")
    parser.add_argument("--site-bucket", type=int)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, check_inputs=True)
    scope = config.scope(args.month, args.site_bucket)
    if scope.is_full and not args.full:
        raise ValueError("Add --full explicitly for an unscoped run")
    if args.full and not scope.is_full:
        raise ValueError("--full cannot be combined with scope filters")
    if not metadata_output_path(config).is_file():
        raise FileNotFoundError(
            "Canonical metadata is missing. Run 01_prepare_metadata.py first."
        )

    run_duplicate_audit(config, scope, overwrite=args.overwrite)
    build_canonical_phase(config, scope, overwrite=args.overwrite)
    validate_canonical_phase(config, scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

