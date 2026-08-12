"""
Diagnostics for the SolarEdge extension.
========================================

Deliverable D1: raw inventory and schema contract.

Everything here reads Parquet FOOTERS, not data. A full inventory of all 12
files touches no row groups and returns in well under a second, which is the
point: you can assert that the delivery is what you think it is before any
expensive step runs.

The one exception is `alias_mapping_summary()`, which reads a 43 KB CSV.

Typical use
-----------
    from solar_edge.lib import se_store, se_diagnostics as diag

    con = se_store.connect()
    inv = diag.raw_inventory(con)
    diag.run_d1_checks(con, inventory=inv)
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import duckdb
import pandas as pd

from solar_edge.config import se_config as C

__all__ = [
    "environment_report",
    "raw_inventory",
    "raw_schema",
    "schema_fingerprint",
    "alias_mapping_summary",
    "check_schema_contract",
    "check_row_counts",
    "check_alias_mapping",
    "run_d1_checks",
]

#: se_config.RAW_COLUMNS uses short type names; DuckDB reports these.
_TYPE_ALIASES = {
    "string": {"VARCHAR"},
    "float": {"FLOAT"},
    "double": {"DOUBLE"},
    "int": {"INTEGER", "BIGINT"},
}


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════

def environment_report() -> pd.DataFrame:
    """
    Package versions and path resolution, as a single pass/fail table.

    Run this before anything else. A missing package or an unresolved data root
    should fail here, in two seconds, not thirty minutes into an ingest.
    """
    rows: list[dict] = []

    required = ["duckdb", "pandas", "numpy"]
    optional = ["pyarrow", "matplotlib", "seaborn", "geopandas", "shapely", "pytz"]

    for name in required + optional:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            rows.append(
                {
                    "item": f"package: {name}",
                    "status": "ok",
                    "detail": version,
                    "required": name in required,
                }
            )
        except ImportError:
            rows.append(
                {
                    "item": f"package: {name}",
                    "status": "MISSING" if name in required else "absent",
                    "detail": "",
                    "required": name in required,
                }
            )

    for label, path, exists in C.describe_paths():
        rows.append(
            {
                "item": f"path: {label}",
                "status": "ok" if exists else "MISSING",
                "detail": path,
                "required": label not in {"STORE_DIR", "ARTEFACT_DIR"},
            }
        )

    frame = pd.DataFrame(rows)
    frame["pass"] = ~(frame["required"] & frame["status"].str.isupper())
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# RAW INVENTORY  (footer metadata only)
# ═══════════════════════════════════════════════════════════════════════════

def _sql_path(path: Path) -> str:
    return Path(path).as_posix().replace("'", "''")


def raw_schema(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    """Column name and DuckDB type for one Parquet file. Reads the footer only."""
    return con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(path)}')"
    ).df()[["column_name", "column_type"]]


def schema_fingerprint(schema: pd.DataFrame) -> str:
    """
    Stable 12-character hash of an ordered (name, type) schema.

    Two files with the same fingerprint are guaranteed to be readable by the same
    query. This is what makes "one schema or several?" an assertion rather than
    an assumption.
    """
    payload = "|".join(
        f"{row.column_name}:{row.column_type}" for row in schema.itertuples()
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def raw_inventory(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    One row per delivered file: month, rows, row groups, columns, size, schema
    fingerprint and writer.

    Footer reads only -- no row group is decompressed.

    The `n_row_groups` column is the one to look at. The delivered files have a
    single row group each, so there are no row-group statistics to prune on and
    every predicate forces a full column scan. That is the justification for the
    re-partitioning step in D3.
    """
    rows = []
    for path in C.raw_files():
        meta = con.execute(
            f"SELECT * FROM parquet_file_metadata('{_sql_path(path)}')"
        ).df()
        schema = raw_schema(con, path)
        rows.append(
            {
                "month": C.raw_month_of(path),
                "file": path.name,
                "n_rows": int(meta["num_rows"].iloc[0]),
                "n_row_groups": int(meta["num_row_groups"].iloc[0]),
                "n_columns": len(schema),
                "size_mb": round(path.stat().st_size / 1024**2, 1),
                "schema_fingerprint": schema_fingerprint(schema),
                "created_by": str(meta["created_by"].iloc[0]),
                "path": str(path),
            }
        )
    frame = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    frame["rows_per_row_group"] = (frame.n_rows / frame.n_row_groups).round(0).astype(int)
    return frame


def alias_mapping_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Site counts by state from the alias mapping. This is the entire site metadata."""
    return con.execute(
        """
        SELECT state,
               count(*)                    AS n_sites,
               count(DISTINCT zip_code)    AS n_postcodes,
               min(alias)                  AS first_alias,
               max(alias)                  AS last_alias
        FROM se_alias
        GROUP BY state
        ORDER BY n_sites DESC
        """
    ).df()


# ═══════════════════════════════════════════════════════════════════════════
# CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def _check(name: str, expected, observed, passed: bool, note: str = "") -> dict:
    return {
        "check": name,
        "expected": expected,
        "observed": observed,
        "pass": bool(passed),
        "note": note,
    }


def check_schema_contract(
    con: duckdb.DuckDBPyConnection, inventory: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Assert that all 12 files share one schema, and that it is the schema
    `se_config.RAW_COLUMNS` declares.

    Two separate claims, checked separately:
      1. internal consistency -- every file has the same fingerprint;
      2. contract conformance -- that fingerprint matches the declared columns
         and types, in order.
    """
    inventory = raw_inventory(con) if inventory is None else inventory
    checks: list[dict] = []

    fingerprints = sorted(inventory.schema_fingerprint.unique())
    checks.append(
        _check(
            "all files share one schema",
            "1 distinct fingerprint",
            f"{len(fingerprints)} ({', '.join(fingerprints)})",
            len(fingerprints) == 1,
        )
    )

    schema = raw_schema(con, Path(inventory.path.iloc[0]))
    observed_cols = list(schema.column_name)
    expected_cols = list(C.RAW_COLUMNS)

    checks.append(
        _check(
            "column names and order",
            f"{len(expected_cols)} columns",
            f"{len(observed_cols)} columns",
            observed_cols == expected_cols,
            "" if observed_cols == expected_cols
            else f"missing={set(expected_cols) - set(observed_cols)}, "
                 f"unexpected={set(observed_cols) - set(expected_cols)}",
        )
    )

    type_mismatches = []
    types = dict(zip(schema.column_name, schema.column_type))
    for column, declared in C.RAW_COLUMNS.items():
        actual = types.get(column)
        allowed = _TYPE_ALIASES.get(declared, {declared.upper()})
        if actual is None or actual.upper() not in allowed:
            type_mismatches.append(f"{column}: declared {declared}, found {actual}")
    checks.append(
        _check(
            "column types match contract",
            "0 mismatches",
            f"{len(type_mismatches)} mismatches",
            not type_mismatches,
            "; ".join(type_mismatches),
        )
    )

    return pd.DataFrame(checks)


def check_row_counts(
    con: duckdb.DuckDBPyConnection, inventory: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Assert the delivery is complete and unchanged: 12 months, no gaps, and per-file
    row counts matching `se_config.EXPECTED_RAW_ROWS`.

    If this fails you have been given a different extract, and every measured
    figure in the architecture proposal needs re-deriving.
    """
    inventory = raw_inventory(con) if inventory is None else inventory
    checks: list[dict] = []

    observed_months = list(inventory.month)
    expected_months = list(C.STUDY_MONTHS)
    checks.append(
        _check(
            "months present",
            f"{len(expected_months)} months, {expected_months[0]}..{expected_months[-1]}",
            f"{len(observed_months)} months",
            observed_months == expected_months,
            "" if observed_months == expected_months
            else f"missing={sorted(set(expected_months) - set(observed_months))}",
        )
    )

    observed_rows = dict(zip(inventory.month, inventory.n_rows))
    mismatches = [
        f"{month}: expected {expected:,}, found {observed_rows.get(month, 0):,}"
        for month, expected in C.EXPECTED_RAW_ROWS.items()
        if observed_rows.get(month) != expected
    ]
    checks.append(
        _check(
            "per-file row counts",
            "12 exact matches",
            f"{12 - len(mismatches)} matches",
            not mismatches,
            "; ".join(mismatches),
        )
    )

    total = int(inventory.n_rows.sum())
    checks.append(
        _check(
            "total rows",
            f"{C.EXPECTED_RAW_TOTAL_ROWS:,}",
            f"{total:,}",
            total == C.EXPECTED_RAW_TOTAL_ROWS,
        )
    )

    single_group = int((inventory.n_row_groups == 1).sum())
    checks.append(
        _check(
            "row-group structure",
            "informational",
            f"{single_group}/{len(inventory)} files have a single row group",
            True,
            "Single row groups mean no statistics to prune on: every predicate "
            "scans the whole column. This is what D3 re-partitioning fixes.",
        )
    )

    return pd.DataFrame(checks)


def check_alias_mapping(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Assert the site mapping is complete, unique, and covers only known states."""
    checks: list[dict] = []

    summary = con.execute(
        """
        SELECT count(*)                 AS n_rows,
               count(DISTINCT alias)    AS n_aliases,
               count(DISTINCT state)    AS n_states,
               count(*) FILTER (WHERE zip_code IS NULL) AS n_null_postcode
        FROM se_alias
        """
    ).df().iloc[0]

    checks.append(
        _check("alias mapping rows", f"{C.EXPECTED_N_SITES:,}",
               f"{int(summary.n_rows):,}", int(summary.n_rows) == C.EXPECTED_N_SITES)
    )
    checks.append(
        _check("aliases are unique", f"{int(summary.n_rows):,} distinct",
               f"{int(summary.n_aliases):,} distinct",
               int(summary.n_aliases) == int(summary.n_rows))
    )
    checks.append(
        _check("postcodes present", "0 nulls",
               f"{int(summary.n_null_postcode)} nulls", int(summary.n_null_postcode) == 0)
    )

    states = set(con.execute("SELECT DISTINCT state FROM se_alias").df().state)
    unknown = states - set(C.STATE_TIMEZONE)
    checks.append(
        _check(
            "every state has a timezone",
            f"states in {sorted(C.STATE_TIMEZONE)}",
            f"{len(states)} states",
            not unknown,
            "" if not unknown else f"no timezone mapped for: {sorted(unknown)}",
        )
    )

    return pd.DataFrame(checks)


def run_d1_checks(
    con: duckdb.DuckDBPyConnection, inventory: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Run every D1 check and return one tidy table.

    D1 is complete when every row has pass = True.
    """
    inventory = raw_inventory(con) if inventory is None else inventory
    frames = [
        check_schema_contract(con, inventory).assign(group="schema contract"),
        check_row_counts(con, inventory).assign(group="inventory"),
        check_alias_mapping(con).assign(group="alias mapping"),
    ]
    result = pd.concat(frames, ignore_index=True)
    return result[["group", "check", "expected", "observed", "pass", "note"]]


def summarise(checks: pd.DataFrame, label: str = "D1") -> bool:
    """Print a one-line verdict and return True if everything passed."""
    n_pass = int(checks["pass"].sum())
    n_total = len(checks)
    ok = n_pass == n_total
    verdict = "PASS" if ok else "FAIL"
    print(f"{label}: {verdict}  ({n_pass}/{n_total} checks)")
    if not ok:
        for row in checks[~checks["pass"]].itertuples():
            print(f"  FAILED  [{row.group}] {row.check}")
            print(f"          expected {row.expected}, observed {row.observed}")
            if row.note:
                print(f"          {row.note}")
    return ok
