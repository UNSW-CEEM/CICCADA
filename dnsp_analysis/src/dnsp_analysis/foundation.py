"""Command-line orchestrator for Delivery 1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import build_canonical_phase
from .config import SourceScope, load_config
from .db import (
    canonical_output_path,
    duplicate_audit_path,
    scope_root,
)
from .duplicates import run_duplicate_audit
from .inventory import run_inventory
from .logging_utils import get_logger, write_json
from .metadata import (
    metadata_output_path,
    prepare_metadata,
    reconciliation_output_path,
)
from .validation import validate_canonical_phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate the Ausgrid Delivery 1 foundation layer."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--month", help="Optional YYYY-MM source month")
    parser.add_argument("--site-bucket", type=int)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Explicitly authorise an unscoped full-dataset run",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-inventory", action="store_true")
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--skip-duplicates", action="store_true")
    parser.add_argument("--skip-canonical", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser


def _run_manifest_path(config, scope: SourceScope) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return scope_root(config, scope) / "_manifests" / f"foundation_run_{stamp}.json"


def run(args: argparse.Namespace) -> dict[str, Any]:
    logger = get_logger()
    config = load_config(args.config, check_inputs=True)
    scope = config.scope(args.month, args.site_bucket)

    if args.full and not scope.is_full:
        raise ValueError("--full cannot be combined with --month or --site-bucket")
    if scope.is_full and not args.full:
        raise ValueError(
            "Refusing an unscoped run. Use --month/--site-bucket for a sample "
            "or add --full explicitly."
        )

    started = datetime.now(timezone.utc)
    stages: dict[str, Any] = {}

    if not args.skip_inventory:
        stages["inventory"] = run_inventory(config, scope)

    metadata_path = metadata_output_path(config)
    if not args.skip_metadata:
        reconciliation_path = reconciliation_output_path(config, scope)
        if (
            metadata_path.exists()
            and reconciliation_path.exists()
            and not args.overwrite
        ):
            logger.info(
                "Reusing metadata and scoped reconciliation: %s",
                reconciliation_path,
            )
            stages["metadata"] = {
                "status": "reused",
                "metadata_path": str(metadata_path),
                "reconciliation_path": str(reconciliation_path),
            }
        else:
            stages["metadata"] = prepare_metadata(
                config,
                scope,
                overwrite=args.overwrite,
            )

    audit_path = duplicate_audit_path(config, scope)
    if not args.skip_duplicates:
        if audit_path.exists() and not args.overwrite:
            logger.info("Reusing existing duplicate audit: %s", audit_path)
            stages["duplicates"] = {"status": "reused", "path": str(audit_path)}
        else:
            stages["duplicates"] = run_duplicate_audit(
                config,
                scope,
                overwrite=args.overwrite,
            )

    canonical_path = canonical_output_path(config, scope)
    if not args.skip_canonical:
        if canonical_path.exists() and not args.overwrite:
            logger.info("Reusing existing canonical output: %s", canonical_path)
            stages["canonical"] = {"status": "reused", "path": str(canonical_path)}
        else:
            stages["canonical"] = build_canonical_phase(
                config,
                scope,
                overwrite=args.overwrite,
            )

    if not args.skip_validation:
        stages["validation"] = validate_canonical_phase(config, scope)

    completed = datetime.now(timezone.utc)
    payload = {
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "config_file": str(args.config),
        "config": config.to_dict(),
        "scope": {
            "label": scope.label,
            "month": scope.month,
            "site_bucket": scope.site_bucket,
            "bucket_count": scope.bucket_count,
        },
        "overwrite": bool(args.overwrite),
        "stages": stages,
    }
    manifest_path = _run_manifest_path(config, scope)
    write_json(manifest_path, payload)
    logger.info("Foundation run manifest written to %s", manifest_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
