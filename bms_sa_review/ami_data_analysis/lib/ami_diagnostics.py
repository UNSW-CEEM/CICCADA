"""
Check tables and the environment report.
========================================

Phase 0. Ported from `solar_edge/lib/se_diagnostics.py` so that a check in this
package looks and reads exactly like a check in that one.

A "check" is a row: what was expected, what was observed, and whether that is a
pass. A notebook displays the frame, calls `summarise()`, and asserts. Nothing
here touches AWS beyond an optional credential probe.
"""

from __future__ import annotations

import importlib

import pandas as pd

from bms_sa_review.ami_data_analysis.config import ami_config as C

__all__ = ["check", "summarise", "environment_report", "aws_report"]

#: Packages this package genuinely cannot run without, by phase.
REQUIRED_PACKAGES = ["pandas", "numpy"]
#: Needed for the AWS-touching notebooks (00-04) only.
AWS_PACKAGES = ["boto3", "botocore", "awswrangler"]
#: Needed for the local notebooks (04-06) and the tests.
LOCAL_PACKAGES = ["duckdb", "pyarrow", "matplotlib", "nbformat", "pytest"]


def check(name: str, expected, observed, passed: bool, note: str = "",
          group: str = "") -> dict:
    """One check row."""
    return {
        "group": group,
        "check": name,
        "expected": expected,
        "observed": observed,
        "pass": bool(passed),
        "note": note,
    }


def summarise(checks: pd.DataFrame, label: str = "checks") -> bool:
    """Print a one-line verdict and return True if everything passed."""
    n_pass = int(checks["pass"].sum())
    n_total = len(checks)
    ok = n_pass == n_total
    print(f"{label}: {'PASS' if ok else 'FAIL'}  ({n_pass}/{n_total} checks)")
    if not ok:
        for row in checks[~checks["pass"]].itertuples():
            group = f"[{row.group}] " if getattr(row, "group", "") else ""
            print(f"  FAILED  {group}{row.check}")
            print(f"          expected {row.expected}, observed {row.observed}")
            if row.note:
                print(f"          {row.note}")
    return ok


def environment_report() -> pd.DataFrame:
    """
    Package versions and path resolution, as one pass/fail table.

    Run this before anything else. A missing package should fail here, in two
    seconds, not thirty minutes into an extract.

    `nbformat` is listed as local rather than required: it is only needed by
    `tests/test_notebook_names.py`, which skips without it. If it is absent that
    test is silently not running, which is worth seeing.
    """
    rows: list[dict] = []
    groups = [
        ("required", REQUIRED_PACKAGES, True),
        ("aws", AWS_PACKAGES, True),
        ("local", LOCAL_PACKAGES, False),
    ]
    for group, names, required in groups:
        for name in names:
            try:
                module = importlib.import_module(name)
                rows.append({
                    "group": group,
                    "item": f"package: {name}",
                    "status": "ok",
                    "detail": getattr(module, "__version__", "unknown"),
                    "required": required,
                })
            except ImportError:
                rows.append({
                    "group": group,
                    "item": f"package: {name}",
                    "status": "MISSING" if required else "absent",
                    "detail": "",
                    "required": required,
                })

    for label, path, exists in C.describe_paths():
        # STORE_DIR and DUCKDB_TEMP_DIR do not exist until Phase 4 creates them.
        rows.append({
            "group": "path",
            "item": f"path: {label}",
            "status": "ok" if exists else "not yet created",
            "detail": path,
            "required": label in {"REPO_ROOT", "ARTEFACT_DIR"},
        })

    return pd.DataFrame(rows)


def aws_report(status: dict) -> pd.DataFrame:
    """Render `ami_athena.credential_status()` as check rows."""
    return pd.DataFrame([
        check("SSO session valid", "valid", status["reason"], status["ok"],
              "" if status["ok"] else status["remedy"], group="aws"),
        check("profile", "ciccada", status["profile"],
              status["profile"] == "ciccada",
              "AWS_PROFILE overrides the default; not an error if deliberate",
              group="aws"),
        check("region", "ap-southeast-2", status["region"],
              status["region"] == "ap-southeast-2", group="aws"),
    ])
