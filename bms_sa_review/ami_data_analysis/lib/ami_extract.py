"""
Phase 5, step 1 -- bulk-landing a full year of `ts` data for the circuits
that survived Phase 4's per-site resolution.
==========================================================================

Nothing is derived here. This module only moves rows from Athena to local
Hive-partitioned Parquet, one (year, month, is_pv, circuit-id chunk) at a
time, so that no single query or in-memory DataFrame ever has to hold more
than one chunk -- full-fleet/full-year is on the order of a few billion
rows, which cannot be one `Athena.aq()` call into one pandas DataFrame the
way Phase 4's 50-site/1-day validation batch was.

Deliberately resolve-then-extract, not extract-then-resolve: Phase 4's
per-site resolution (duplicates, inactive circuits, storage/sign issues)
runs on a single representative day and decides which circuits survive
*before* this module spends any Athena scan or local disk on them. Extract
only ever sees the circuit_ids Phase 4 already decided are worth a full
year of data.

Every function here is pure given the `aq_fn` callable it's handed -- no
`ami_athena` import, no real Athena call happens inside this module. A
notebook wires in `ami_athena.aq` for a real run; tests wire in a fake that
returns fixture frames, so the whole chunking/write orchestration is
unit-testable without touching Athena or its cost.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

__all__ = [
    "chunk_circuit_ids",
    "months_in_range",
    "build_extract_sql",
    "extract_chunk",
    "write_extract_chunk",
    "run_extraction",
]


TS_COLUMNS = (
    "circuit_id", "t_stamp", "power", "energy", "energy_reactive",
    "energy_import", "energy_export", "energy_reactive_import",
    "energy_reactive_export", "power_factor", "voltage", "current",
)


def chunk_circuit_ids(circuit_ids, *, chunk_size: int = 800) -> list[list[int]]:
    """
    Split a flat collection of circuit_ids into fixed-size, sorted chunks.
    Pure.

    Sorted (not just batched in whatever order the caller handed them) so a
    rerun with the same `circuit_ids` produces the same chunks in the same
    order -- `run_extraction`'s per-chunk file naming depends on that for
    idempotency (see its docstring).
    """
    ids = sorted({int(c) for c in circuit_ids})
    if not ids:
        return []
    return [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]


def months_in_range(
    start_year: int, start_month: int, end_year: int, end_month: int,
) -> list[tuple[int, int]]:
    """
    Inclusive (year, month) pairs from (start_year, start_month) to
    (end_year, end_month). Pure.

    Raises ValueError if the range is empty or inverted -- an accidentally
    swapped start/end should fail loudly here, not silently extract zero
    months.
    """
    start_index = start_year * 12 + (start_month - 1)
    end_index = end_year * 12 + (end_month - 1)
    if end_index < start_index:
        raise ValueError(
            f"end ({end_year}-{end_month:02d}) is before start "
            f"({start_year}-{start_month:02d})."
        )
    pairs = []
    for index in range(start_index, end_index + 1):
        pairs.append((index // 12, index % 12 + 1))
    return pairs


def build_extract_sql(
    circuit_ids_chunk, year: int, month: int, is_pv: bool, *,
    table: str = "ts",
    columns=TS_COLUMNS,
) -> str:
    """
    The SELECT for one (year, month, is_pv, circuit-id chunk) extract
    query. Pure string-building, no execution.

    Mirrors notebook 04's real validation-batch pull exactly (same column
    list, same `year = .. AND month = .. AND is_pv = ..` partition
    predicate, same `circuit_id IN (...)` list) -- this is a generalisation
    of that pattern to an arbitrary month and chunk, not a new query shape.
    A whole year of one circuit-id chunk is still 12 separate calls (one
    per month), one per `is_pv` value, because `ts` partitions on exactly
    those three columns -- a query can't span partitions in a single WHERE
    the way it can span circuit_ids in one IN-list.
    """
    if not circuit_ids_chunk:
        raise ValueError("circuit_ids_chunk is empty -- nothing to query.")
    id_list_sql = ", ".join(str(int(c)) for c in circuit_ids_chunk)
    is_pv_sql = "true" if is_pv else "false"
    column_list = ", ".join(columns)
    return (
        f"SELECT {column_list}\n"
        f"FROM {table}\n"
        f"WHERE year = {year} AND month = {month} AND is_pv = {is_pv_sql}\n"
        f"  AND circuit_id IN ({id_list_sql})"
    )


def extract_chunk(
    aq_fn, circuit_ids_chunk, year: int, month: int, is_pv: bool, *,
    database=None, label: str | None = None,
) -> pd.DataFrame:
    """
    Run one (year, month, is_pv, chunk) extract query via `aq_fn` and
    return whatever it returns.

    `aq_fn` is called as `aq_fn(sql, database=database, label=label)` --
    this is exactly `ami_athena.aq`'s signature, so a real run passes that
    function directly. Nothing else in this module knows about Athena.
    """
    sql = build_extract_sql(circuit_ids_chunk, year, month, is_pv)
    resolved_label = label or (
        f"extract {year}-{month:02d} is_pv={is_pv} "
        f"({len(circuit_ids_chunk)} circuits)"
    )
    return aq_fn(sql, database=database, label=resolved_label)


def write_extract_chunk(
    frame: pd.DataFrame, store_dir, year: int, month: int, *,
    table_name: str = "ami_extract",
    partition_key: str = "dt_month",
    chunk_index: int = 0,
    is_pv: bool | None = None,
    compression: str = "zstd",
) -> Path | None:
    """
    Write one already-pulled chunk to the Hive-partitioned landing store:
    `<store_dir>/<table_name>/<partition_key>=YYYY-MM/part-<is_pv>-<chunk_index>.parquet`

    Returns the path written, or None if `frame` is empty (nothing is
    written for an empty chunk -- an empty file would otherwise silently
    inflate the manifest's row/file counts for no reason).

    The filename is DETERMINISTIC in (year, month, is_pv, chunk_index) --
    not a random uuid -- so re-running the same extraction overwrites the
    same file rather than accumulating duplicate part files. `chunk_index`
    must therefore be stable across reruns; `run_extraction` derives it
    from `chunk_circuit_ids`'s sorted, deterministic chunking, not from
    call order.
    """
    if frame is None or not len(frame):
        return None
    partition_dir = Path(store_dir) / table_name / f"{partition_key}={year}-{month:02d}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    is_pv_tag = "na" if is_pv is None else ("pv" if is_pv else "load")
    path = partition_dir / f"part-{is_pv_tag}-{chunk_index:04d}.parquet"
    frame.to_parquet(path, compression=compression, index=False)
    return path


def run_extraction(
    aq_fn, circuits: pd.DataFrame, year_month_pairs, store_dir, *,
    circuit_column: str = "circuit_id",
    is_pv_column: str = "is_pv",
    chunk_size: int = 800,
    table_name: str = "ami_extract",
    partition_key: str = "dt_month",
    database=None,
) -> pd.DataFrame:
    """
    Orchestrate the full chunked extraction: every (year, month) in
    `year_month_pairs`, crossed with both `is_pv` values present in
    `circuits`, crossed with every circuit-id chunk for that `is_pv` value.

    `circuits` is a small frame -- one row per SURVIVING circuit_id (from
    Phase 4's resolution, already filtered to `kept=True`), with at least
    `circuit_column` and `is_pv_column`. This function never queries
    `meta_up23c` itself; that decision already happened upstream.

    Never holds more than one chunk's pulled DataFrame in memory at a time
    -- each chunk is pulled via `extract_chunk`, written via
    `write_extract_chunk`, and dropped before the next chunk is pulled.

    Returns a provenance DataFrame, one row per (year, month, is_pv,
    chunk_index): `year, month, is_pv, chunk_index, n_circuits, n_rows,
    path` (`path` is None for a chunk that returned zero rows -- see
    `write_extract_chunk`). This is the audit trail for what got landed
    and from which query, mirroring `ami_config.STORE_TABLES`'s
    `ami_provenance` table.
    """
    if circuits is None or not len(circuits):
        return pd.DataFrame(columns=[
            "year", "month", "is_pv", "chunk_index", "n_circuits", "n_rows", "path",
        ])

    provenance_rows = []
    for is_pv_value in sorted(circuits[is_pv_column].dropna().unique().tolist()):
        side_ids = circuits.loc[
            circuits[is_pv_column] == is_pv_value, circuit_column
        ].tolist()
        chunks = chunk_circuit_ids(side_ids, chunk_size=chunk_size)
        for year, month in year_month_pairs:
            for chunk_index, id_chunk in enumerate(chunks):
                frame = extract_chunk(
                    aq_fn, id_chunk, year, month, bool(is_pv_value), database=database,
                )
                path = write_extract_chunk(
                    frame, store_dir, year, month,
                    table_name=table_name, partition_key=partition_key,
                    chunk_index=chunk_index, is_pv=bool(is_pv_value),
                )
                provenance_rows.append({
                    "year": year, "month": month, "is_pv": bool(is_pv_value),
                    "chunk_index": chunk_index, "n_circuits": len(id_chunk),
                    "n_rows": 0 if frame is None else len(frame), "path": path,
                })
                del frame

    return pd.DataFrame(provenance_rows)
