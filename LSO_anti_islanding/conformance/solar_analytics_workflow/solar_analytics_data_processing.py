"""Compatibility entry point for Solar Analytics preprocessing."""

import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parent
CONFORMANCE_DIR = WORKFLOW_DIR.parent
if str(CONFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_DIR))

from solar_analytics_workflow.preprocessing import build_cleaned_site_data


def main():
    cleaned_path = build_cleaned_site_data()
    print(f"Saved cleaned Solar Analytics data to {cleaned_path}")


if __name__ == "__main__":
    main()
