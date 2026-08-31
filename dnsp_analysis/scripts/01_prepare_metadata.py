"""Normalise the workbook and reconcile IDs against telemetry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dnsp_analysis.config import load_config  # noqa: E402
from dnsp_analysis.metadata import prepare_metadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, check_inputs=True)
    prepare_metadata(config, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

