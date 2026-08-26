"""
Data lake inventory: what is actually in the catalogue.
=======================================================

Almost everything here reads METADATA:

* **Glue** (`glue_inventory`, `column_inventory_glue`):
  It gives every database, every table, its storage location, its declared columns and its partition keys, without Athena being involved at all.
  
* **Iceberg metadata tables** (`partition_frame`) -- `SELECT * FROM "ts$partitions"`
  -- read the table's own manifests. Athena reports these as a near-zero scan.
  For an Iceberg fact table this is how you get exact per-partition ROW COUNTS
  and BYTE SIZES without touching a single data file. It is the single most
  useful query in this notebook.
* **`information_schema.columns`** gives the schema Athena will actually resolve,
  which is not always the schema Glue declares (`aws_config.describe` notes that
  the Glue column list can be empty for Iceberg).

Only `dimension_counts` and `sample` read data, and both are confined to
dimension tables or partition-filtered slices.

None of this is free in the strict sense: Athena bills a 10 MB minimum per
query, so a notebook of thirty metadata queries costs about a third of a cent.
`ami_athena.scan_report()` prints the real figure.
"""

from __future__ import annotations

import re

import pandas as pd

from bms_sa_review.ami_data_analysis.config import ami_config as C
from bms_sa_review.ami_data_analysis.lib import ami_athena as A

__all__ = [
    "glue_inventory", "column_inventory", "column_inventory_glue",
    "partition_frame", "should_probe_partitions", "probe_partitions",
    "normalise_partitions", "partition_totals",
    "database_summary", "summary_table",
    "dimension_counts", "sample", "guess_candidates",
]


# ═══════════════════════════════════════════════════════════════════════════
# GLUE CATALOGUE  (free)
# ═══════════════════════════════════════════════════════════════════════════

def glue_inventory(databases: list[str] | None = None) -> pd.DataFrame:
    """
    Every table in every Glue database, with the facts that decide how to query it.

    Columns:
      database, table, table_type, is_iceberg, n_columns, partition_keys,
      location, updated

    `is_iceberg` is the one that matters. An Iceberg table exposes `$partitions`,
    `$files`, `$snapshots` and `$history` metadata tables, which is how Phase 1
    gets row counts for free. A Hive table exposes only `$partitions`, and only
    its partition VALUES -- no counts.

    Free: Glue API calls are not billed by Athena.
    """
    session = A.get_session()
    glue = session.client("glue")

    if databases is None:
        databases = []
        for page in glue.get_paginator("get_databases").paginate():
            databases.extend(entry["Name"] for entry in page["DatabaseList"])

    rows = []
    for database in databases:
        for page in glue.get_paginator("get_tables").paginate(DatabaseName=database):
            for table in page["TableList"]:
                params = table.get("Parameters") or {}
                storage = table.get("StorageDescriptor") or {}
                table_type = params.get("table_type") or table.get("TableType") or ""
                rows.append({
                    "database": database,
                    "table": table["Name"],
                    "table_type": table.get("TableType", ""),
                    "format": table_type,
                    "is_iceberg": str(table_type).upper() == "ICEBERG",
                    "n_columns": len(storage.get("Columns") or []),
                    "partition_keys": ", ".join(
                        key["Name"] for key in (table.get("PartitionKeys") or [])
                    ),
                    "location": storage.get("Location", ""),
                    "updated": table.get("UpdateTime"),
                })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(["database", "table"]).reset_index(drop=True)
    return frame


def column_inventory_glue(database: str, table: str) -> pd.DataFrame:
    """
    Declared columns from Glue, including partition keys. Free.

    Can be empty for Iceberg tables -- that is a known Glue behaviour, not a
    failure. Use `column_inventory` (Athena `information_schema`) when it is.
    """
    glue = A.get_session().client("glue")
    detail = glue.get_table(DatabaseName=database, Name=table)["Table"]
    storage = detail.get("StorageDescriptor") or {}
    rows = [
        {"column_name": c["Name"], "data_type": c["Type"],
         "is_partition_key": False, "comment": c.get("Comment", "")}
        for c in (storage.get("Columns") or [])
    ]
    rows += [
        {"column_name": c["Name"], "data_type": c["Type"],
         "is_partition_key": True, "comment": c.get("Comment", "")}
        for c in (detail.get("PartitionKeys") or [])
    ]
    return pd.DataFrame(rows)


def column_inventory(database: str) -> pd.DataFrame:
    """
    Every column of every table in a database, as Athena resolves it.

    One query for a whole database, rather than one DESCRIBE per table. This is
    the REAL schema -- what Athena will actually let you select -- as distinct
    from what the pipeline code implies the table contains.
    """
    return A.aq(
        f"""
        SELECT table_name, ordinal_position, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{database}'
        ORDER BY table_name, ordinal_position
        """,
        database=database,
        label=f"information_schema.columns [{database}]",
    )


# ═══════════════════════════════════════════════════════════════════════════
# PARTITION METADATA
# ═══════════════════════════════════════════════════════════════════════════

def partition_frame(table: str, database: str | None = None) -> pd.DataFrame:
    """
    The `$partitions` metadata table, raw and unmodified.

    For Iceberg this carries per-partition record counts, file counts and byte
    sizes, and often per-column min/max statistics. For Hive it carries the
    partition VALUES only.

    Returned raw on purpose: the exact column set differs between Athena engine
    versions and between Iceberg and Hive, and the notebook should show you what
    came back before anything tries to interpret it. `normalise_partitions()`
    does the interpreting, tolerantly.

    Reads manifests, not data.
    """
    return A.aq(
        f'SELECT * FROM "{table}$partitions"',
        database=database or C.SAI,
        label=f"{table}$partitions",
    )


#: Candidate column names, in preference order, for each normalised field.
_RECORD_COUNT_NAMES = ("record_count", "row_count", "rowcount", "records")
_FILE_COUNT_NAMES = ("file_count", "filecount", "files", "data_file_count")
_SIZE_NAMES = ("total_size", "file_size_in_bytes", "size_in_bytes",
               "total_data_file_size_in_bytes", "size")


def _first_present(columns, candidates) -> str | None:
    """First candidate appearing in `columns`, case-insensitively. Pure."""
    lowered = {str(c).lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def _expand_struct_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Expand a struct-valued column into flat columns. Pure.

    Athena returns an Iceberg `partition` struct to pandas as either a dict or
    its string rendering `{year=2025, month=1, is_pv=true}`. Both are handled;
    anything else is left alone.
    """
    values = frame[column]
    if values.empty:
        return pd.DataFrame(index=frame.index)

    first = values.dropna()
    if first.empty:
        return pd.DataFrame(index=frame.index)
    first = first.iloc[0]

    if isinstance(first, dict):
        return pd.DataFrame(list(values.map(lambda v: v if isinstance(v, dict) else {})),
                            index=frame.index)

    if isinstance(first, str) and "=" in first:
        def parse(text):
            if not isinstance(text, str):
                return {}
            body = text.strip().strip("{}")
            out = {}
            for part in body.split(","):
                if "=" in part:
                    key, _, value = part.partition("=")
                    out[key.strip()] = value.strip()
            return out
        return pd.DataFrame(list(values.map(parse)), index=frame.index)

    return pd.DataFrame(index=frame.index)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion that leaves non-numeric values as NaN. Pure."""
    return pd.to_numeric(series, errors="coerce")


def normalise_partitions(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a raw `$partitions` frame into a tidy one. Pure -- no AWS, no I/O.

    Output columns, when the input supports them:
        the partition columns themselves (year, month, is_pv, ...),
        n_rows, n_files, size_bytes

    Tolerant by design. `$partitions` has three shapes in the wild -- a nested
    `partition` struct, a stringified struct, or flat partition columns -- and
    the count columns are named differently between engine versions. Anything
    that cannot be identified is simply not in the output, and the caller sees
    fewer columns rather than an exception.
    """
    if frame is None or not len(frame):
        return pd.DataFrame()

    parts = []

    if "partition" in frame.columns:
        expanded = _expand_struct_column(frame, "partition")
        if len(expanded.columns):
            parts.append(expanded)

    flat = [
        column for column in frame.columns
        if str(column).lower() in {"year", "month", "day", "is_pv", "dt_month"}
    ]
    if flat:
        parts.append(frame[flat])

    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)
    out = out.loc[:, ~out.columns.duplicated()]

    for target, candidates in (
        ("n_rows", _RECORD_COUNT_NAMES),
        ("n_files", _FILE_COUNT_NAMES),
        ("size_bytes", _SIZE_NAMES),
    ):
        source = _first_present(frame.columns, candidates)
        if source is not None:
            out[target] = _coerce_numeric(frame[source])

    for column in ("year", "month"):
        if column in out.columns:
            out[column] = _coerce_numeric(out[column]).astype("Int64")

    sort_by = [c for c in ("year", "month", "is_pv") if c in out.columns]
    if sort_by:
        out = out.sort_values(sort_by).reset_index(drop=True)
    return out


#: Column names `partition_totals` will report as the table's actual partition
#: columns, in preference order. Deliberately the same set `normalise_partitions`
#: knows how to recover, so the two never disagree about what "actual" means.
_PARTITION_COLUMN_NAMES = ("year", "month", "day", "is_pv")


def partition_totals(tidy: pd.DataFrame) -> dict:
    """
    Roll a tidy partition frame up to one dict. Pure.

    Keys: n_partitions, n_rows, n_files, size_bytes, year_min, year_max,
    first_partition, last_partition, is_pv_values, partition_columns. Missing
    inputs give None rather than a KeyError -- a Hive `$partitions` frame has
    values and nothing else, and that is a legitimate answer.

    `partition_columns` is the ACTUAL partition columns found in the data, which
    is the thing to trust over Glue's declared `PartitionKeys` -- see
    `should_probe_partitions` for why those two can disagree.
    """
    if tidy is None or not len(tidy):
        return {"n_partitions": 0, "partition_columns": []}

    out: dict = {"n_partitions": int(len(tidy))}
    out["partition_columns"] = [c for c in _PARTITION_COLUMN_NAMES if c in tidy.columns]
    for column in ("n_rows", "n_files", "size_bytes"):
        out[column] = float(tidy[column].sum()) if column in tidy.columns else None

    if "year" in tidy.columns and tidy["year"].notna().any():
        out["year_min"] = int(tidy["year"].min())
        out["year_max"] = int(tidy["year"].max())
        if "month" in tidy.columns and tidy["month"].notna().any():
            ordered = tidy.dropna(subset=["year", "month"]).sort_values(["year", "month"])
            first, last = ordered.iloc[0], ordered.iloc[-1]
            out["first_partition"] = f"{int(first.year)}-{int(first.month):02d}"
            out["last_partition"] = f"{int(last.year)}-{int(last.month):02d}"
    else:
        out["year_min"] = out["year_max"] = None

    if "is_pv" in tidy.columns:
        out["is_pv_values"] = sorted({str(v) for v in tidy["is_pv"].dropna().unique()})
    return out


def summary_table(catalog: pd.DataFrame, totals: dict[str, dict]) -> pd.DataFrame:
    """
    The one-screen inventory. Pure -- takes frames, returns a frame.

    `catalog` is `glue_inventory()`. `totals` maps "database.table" to the dict
    `partition_totals()` returned, for whichever tables were probed. Tables that
    were not probed appear with blanks rather than being dropped, so the screen
    shows the whole catalogue and not just the interesting part.
    """
    if catalog is None or not len(catalog):
        return pd.DataFrame()

    rows = []
    for entry in catalog.itertuples(index=False):
        key = f"{entry.database}.{entry.table}"
        stats = totals.get(key, {})
        size = stats.get("size_bytes")
        declared = entry.partition_keys
        actual = ", ".join(stats.get("partition_columns", []) or [])
        partitioned_by = declared or actual
        if not declared and actual:
            partitioned_by += "  [not declared in Glue]"

        rows.append({
            "database": entry.database,
            "table": entry.table,
            "fmt": "iceberg" if entry.is_iceberg else (entry.table_type or "hive").lower(),
            "cols": entry.n_columns,
            "partitioned_by": partitioned_by,
            "n_partitions": stats.get("n_partitions"),
            "n_rows": stats.get("n_rows"),
            "size": A.fmt_bytes(size) if size else "",
            "coverage": (
                f"{stats['first_partition']} .. {stats['last_partition']}"
                if stats.get("first_partition") else ""
            ),
            "is_pv": ", ".join(stats.get("is_pv_values", [])) or "",
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["n_rows", "database", "table"], ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def database_summary(catalog: pd.DataFrame) -> pd.DataFrame:
    """One row per Glue database: how many tables, how many of them Iceberg. Pure."""
    if catalog is None or not len(catalog):
        return pd.DataFrame()
    grouped = catalog.groupby("database", as_index=False).agg(
        n_tables=("table", "count"),
        n_iceberg=("is_iceberg", "sum"),
        n_partitioned=("partition_keys", lambda s: int((s.astype(str) != "").sum())),
    )
    grouped["n_iceberg"] = grouped["n_iceberg"].astype(int)
    return grouped.sort_values("n_tables", ascending=False).reset_index(drop=True)


def should_probe_partitions(is_iceberg: bool, declared_partition_keys: str) -> bool:
    """
    Whether a table's `$partitions` metadata is worth reading. Pure.

    Iceberg tables are ALWAYS probed: Glue's declared `PartitionKeys` reflects
    Hive-style partitioning and routinely comes back empty for an Iceberg table
    even when it is genuinely partitioned, because the spec lives in Iceberg's own
    metadata rather than in Glue. `$partitions` is authoritative for Iceberg
    regardless of what Glue says, and costs almost nothing to read even when the
    table turns out to be unpartitioned (one row of real totals comes back).

    A Hive table with no declared partition keys is skipped: it genuinely has no
    `$partitions` to serve, and asking would be a wasted, billed query that just
    errors.
    """
    if is_iceberg:
        return True
    return bool(str(declared_partition_keys or "").strip())


def probe_partitions(
    catalog: pd.DataFrame,
    only: list[str] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, dict], dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Read `$partitions` for every partitioned table in `catalog`.

    Returns `(totals, raws, log)`:
      totals -- "database.table" -> the dict `partition_totals()` produced
      raws   -- "database.table" -> the RAW `$partitions` frame, so the notebook
                can show what actually came back before anything interpreted it
      log    -- one row per attempt, with `ok` and the error message

    Never raises. A table whose `$partitions` is unavailable -- a view, a Hive
    table on an engine that will not serve it, a permissions gap -- is logged and
    skipped, because one awkward table should not stop the inventory.

    WHICH TABLES ARE PROBED, AND WHY THIS IS NOT JUST "has declared partition keys"
    --------------------------------------------------------------------------
    Iceberg hides its partition spec from Glue: `Table.PartitionKeys` reflects
    Hive-style declarative partitioning, and an Iceberg table registered in Glue
    routinely reports an EMPTY partition key list there even when it is genuinely
    partitioned -- the spec lives in the Iceberg metadata, not in Glue. `ts` in
    this catalogue is exactly this case: Glue says "(none)", but it is partitioned
    on (year, month, is_pv) per `aws_config.py`, and `$partitions` is the only
    place that is visible from the catalogue.

    So every Iceberg table is probed regardless of what Glue declares -- `$partitions`
    is authoritative for Iceberg either way, and on an unpartitioned Iceberg table it
    still returns one row of real totals. Only a Hive table with no declared keys is
    skipped, because a Hive table genuinely has no `$partitions` to read in that case,
    and asking would be a wasted, billed query that just errors.

    `partition_columns` in the returned totals (see `partition_totals`) is set from
    what `$partitions` actually contained, so a mismatch against the DECLARED keys
    is visible in `log` as `glue_hides_partitioning` rather than silently absorbed.
    """
    totals: dict[str, dict] = {}
    raws: dict[str, pd.DataFrame] = {}
    rows = []

    if catalog is None or not len(catalog):
        return totals, raws, pd.DataFrame()

    for entry in catalog.itertuples(index=False):
        key = f"{entry.database}.{entry.table}"
        if only is not None and key not in only and entry.table not in only:
            continue
        declared = str(getattr(entry, "partition_keys", "") or "").strip()
        is_iceberg = bool(getattr(entry, "is_iceberg", False))
        if not should_probe_partitions(is_iceberg, declared):
            continue

        try:
            raw = partition_frame(entry.table, database=entry.database)
            tidy = normalise_partitions(raw)
            raws[key] = raw
            stats = partition_totals(tidy)
            totals[key] = stats
            actual_cols = stats.get("partition_columns", [])
            hidden = is_iceberg and not declared and bool(actual_cols)
            rows.append({
                "table": key, "ok": True,
                "n_partitions": stats.get("n_partitions"),
                "declared_partition_keys": declared or "(none declared in Glue)",
                "actual_partition_columns": ", ".join(actual_cols),
                "glue_hides_partitioning": hidden,
                "error": "",
            })
            if verbose:
                n_rows = stats.get("n_rows")
                note = "  [partitioning hidden from Glue]" if hidden else ""
                print(f"  {key:<50} {stats.get('n_partitions', 0):>5} partitions"
                      + (f", {n_rows:>15,.0f} rows" if n_rows else "") + note)
        except Exception as exc:
            rows.append({"table": key, "ok": False, "n_partitions": None,
                         "declared_partition_keys": declared or "(none declared in Glue)",
                         "actual_partition_columns": "", "glue_hides_partitioning": False,
                         "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
            if verbose:
                print(f"  {key:<50} SKIPPED -- {type(exc).__name__}")

    log = pd.DataFrame(rows)
    if verbose and len(log) and "glue_hides_partitioning" in log.columns:
        hidden_rows = log[log.glue_hides_partitioning]
        if len(hidden_rows):
            print(f"\nNote: {len(hidden_rows)} Iceberg table(s) are partitioned but "
                  f"Glue declares no PartitionKeys for them -- a known Iceberg-on-Glue "
                  f"quirk, not a data problem. Real partition columns, from $partitions:")
            for row in hidden_rows.itertuples():
                print(f"  - {row.table}: {row.actual_partition_columns}")

    return totals, raws, log


# ═══════════════════════════════════════════════════════════════════════════
# DIMENSION TABLES  (small, cheap to read)
# ═══════════════════════════════════════════════════════════════════════════

def dimension_counts(table: str = "meta_up23c", database: str | None = None,
                     site_column: str = "site_id",
                     circuit_column: str = "circuit_id") -> pd.DataFrame:
    """
    Site and circuit counts from a dimension table.

    `meta_up23c` is one row per circuit, so this gives the fleet's shape in one
    query: how many circuits, how many sites, and the mean circuits per site --
    which is the first hint at how much sub-metering there is to work with.
    """
    return A.aq(
        f"""
        SELECT count(*)                            AS n_rows,
               count(DISTINCT {circuit_column})    AS n_circuits,
               count(DISTINCT {site_column})       AS n_sites,
               round(count(DISTINCT {circuit_column})
                     * 1.0 / count(DISTINCT {site_column}), 2) AS circuits_per_site
        FROM {table}
        """,
        database=database or C.SAI,
        label=f"dimension counts [{table}]",
    )


def sample(table: str, database: str | None = None, where: str | None = None,
           columns: str = "*", n: int = 5) -> pd.DataFrame:
    """
    A few real rows. Guarded: a big table needs a `where` with a partition
    predicate, or `ami_athena` refuses.
    """
    sql = f"SELECT {columns} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    sql += f" LIMIT {int(n)}"
    return A.aq(sql, database=database or C.SAI, label=f"sample [{table}]")


# ═══════════════════════════════════════════════════════════════════════════
# CANDIDATE SHORTLIST
# ═══════════════════════════════════════════════════════════════════════════

#: Substrings that suggest a table is relevant to building a synthetic meter.
#: Deliberately generous -- this SHORTLISTS for human reading, it does not decide.
_CANDIDATE_PATTERNS = (
    "ts", "meta", "circuit", "site", "structured", "uncurtailed",
    "battery", "bess", "load", "net", "partition_lookup", "inverter",
)


def guess_candidates(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Flag tables whose NAME suggests relevance. Pure.

    A naming heuristic and nothing more. It is here so the per-table detail
    section has an ordering, not so that anything gets excluded: a table this
    misses is still in `summary_table()` and still visible.
    """
    if catalog is None or not len(catalog):
        return pd.DataFrame()
    pattern = re.compile("|".join(re.escape(p) for p in _CANDIDATE_PATTERNS), re.I)
    out = catalog.copy()
    out["shortlisted"] = out["table"].astype(str).str.contains(pattern)
    return out
