"""Command-line orchestration for Delivery 2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .db import scope_root
from .delivery2_profiles import build_site_profiles
from .delivery2_structured import build_structured_phase, build_structured_site
from .delivery2_validation import validate_delivery2
from .logging_utils import write_json


def run_delivery2(
    config,
    scope,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    profiles = build_site_profiles(config, scope, overwrite=overwrite)
    phase = build_structured_phase(config, scope, overwrite=overwrite)
    site = build_structured_site(config, scope, overwrite=overwrite)
    validation = validate_delivery2(config, scope)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope.label,
        "profiles": profiles,
        "structured_phase": phase,
        "structured_site": site,
        "validation_status": validation["status"],
    }
    manifest = scope_root(config, scope) / "_manifests" / "delivery2_run.json"
    write_json(manifest, payload)
    if validation["status"] != "pass":
        raise RuntimeError(f"Delivery 2 validation failed: {validation['failures']}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--month")
    parser.add_argument("--site-bucket", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, check_inputs=False)
    scope = config.scope(args.month, args.site_bucket)
    run_delivery2(config, scope, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
