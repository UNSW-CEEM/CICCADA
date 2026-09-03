"""
DuckDB access layer for the SolarEdge store.
============================================

This is the SolarEdge equivalent of `bms_sa_review/shared/aws_config.py`: the one
module that knows how to open a connection and what the queryable relations are
called. Every other module writes plain SQL against logical names
(`se_raw`, `se_interval`, `se_site`, ...) and never sees a filesystem path.

Why DuckDB over local Parquet
-----------------------------
The dataset is 1.57 GB compressed (about 12 GB as float64 in memory), so it is
never loaded whole. DuckDB streams it out-of-core, and the SQL fragments already
emitted by `bms_sa_review.shared.as4777_curves` (`vvar_required_q_sql`,
`vw_max_p_sql`, `q_cap_absorbing_sql`, `q_impact_nearest_edge_sql`) run
essentially unchanged -- `CASE`, `power`, `sqrt`, `sign`, `count_if`, `bool_or`
and `quantile_cont` all exist in both engines. That keeps the SolarEdge queries
line-comparable with the Athena originals.

Typical use
-----------
    from oem_analysis.lib import se_store

    con = se_store.connect()
    se_store.store_status(con)                    # what exists so far
    df = se_store.q(con, "SELECT count(*) FROM se_raw")
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

from oem_analysis.config import se_config as C

__all__ = [
    "connect",
    "q",
    "preview_sql",
    "explain",
    "store_status",
    "relations",
]


# ═══════════════════════════════════════════════════════════════════════════
# CONNECTION
# ═══════════════════════════════════════════════════════════════════════════

def connect(
    memory_limit: str | None = None,
    threads: int | None = None,
    register: bool = True,
    verbose: bool = False,
) -> duckdb.DuckDBPyConnection:
    """
    Open an in-memory DuckDB connection with the SolarEdge relations registered.

    The connection itself is in-memory; all data stays in Parquet on disk. Nothing
    is copied into a DuckDB database file, which keeps the store readable by
    pandas / polars / Arrow and regenerable from the raw delivery.

    Parameters
    ----------
    memory_limit : e.g. "6GB". Defaults to se_config.DUCKDB_MEMORY_LIMIT.
    threads      : defaults to se_config.DUCKDB_THREADS.
    register     : register views over the raw files and any built store tables.
    verbose      : print what was registered.
    """
    con = duckdb.connect(database=":memory:")

    # Only constrain DuckDB if asked. Left alone it sizes itself from the machine,
    # which is more robust than any default we could guess here.
    limit = memory_limit or C.DUCKDB_MEMORY_LIMIT
    if limit:
        con.execute(f"SET memory_limit='{limit}'")
    n_threads = threads or C.DUCKDB_THREADS
    if n_threads:
        con.execute(f"SET threads={int(n_threads)}")

    # Progress bars emit carriage returns that corrupt notebook output and logs.
    con.execute("PRAGMA disable_progress_bar")
    # ICU provides IANA timezone support, which the DST resolution in se_ingest needs.
    try:
        con.execute("LOAD icu")
    except duckdb.Error:  # pragma: no cover - icu is bundled in supported versions
        pass

    temp_dir = Path(C.DUCKDB_TEMP_DIR)
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{temp_dir.as_posix()}'")
    except OSError:
        # Spilling to the default location is fine; not worth failing a connection over.
        pass

    if register:
        registered = register_all(con)
        if verbose:
            for name, detail in registered:
                print(f"  registered  {name:<20} {detail}")
    return con


# ═══════════════════════════════════════════════════════════════════════════
# VIEW REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════

def register_all(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """Register the raw views plus every store table that has been built. Returns a log."""
    return register_raw_views(con) + register_store_views(con)


def register_raw_views(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """
    Register views over the delivered files.

    se_raw    -- all 12 monthly files as one relation, plus a `source_file` column.
                 The column is lazy: it costs nothing unless selected.
    se_alias  -- the alias -> postcode/state mapping CSV.
    """
    log: list[tuple[str, str]] = []

    files = C.raw_files()
    con.execute(
        f"""
        CREATE OR REPLACE VIEW se_raw AS
        SELECT * FROM read_parquet({C.duckdb_path_list(files)}, filename = true)
        """
    )
    log.append(("se_raw", f"{len(files)} monthly Parquet files"))

    if Path(C.ALIAS_MAPPING_CSV).exists():
        csv_path = Path(C.ALIAS_MAPPING_CSV).as_posix().replace("'", "''")
        con.execute(
            f"""
            CREATE OR REPLACE VIEW se_alias AS
            SELECT * FROM read_csv('{csv_path}', header = true, AUTO_DETECT = true)
            """
        )
        log.append(("se_alias", Path(C.ALIAS_MAPPING_CSV).name))

    return log


def register_store_views(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """
    Register a view for each store table that exists.

    Missing tables are skipped silently: the store is built up deliverable by
    deliverable, so early notebooks legitimately run against a partial store.
    `store_status()` reports what is and is not there.
    """
    log: list[tuple[str, str]] = []

    for logical in C.STORE_TABLES:
        path = C.store_path(logical)
        if not path.exists():
            continue

        if logical in C.PARTITIONED_TABLES:
            # Existence of the directory is not enough: an interrupted build can
            # leave an empty shell behind, and registering a view over it would
            # fail loudly at connect time for no good reason.
            if not any(path.rglob("*.parquet")):
                continue
            glob = (path / "**" / "*.parquet").as_posix().replace("'", "''")
            source = f"read_parquet('{glob}', hive_partitioning = true, union_by_name = true)"
            detail = f"partitioned, {len(list(path.rglob('*.parquet')))} files"
        else:
            source = f"read_parquet('{path.as_posix()}')"
            detail = "single file"

        projection = C.STORE_VIEW_PROJECTION.get(logical, "*")
        con.execute(f"CREATE OR REPLACE VIEW {logical} AS SELECT {projection} FROM {source}")
        if projection != "*":
            detail += ", + derived columns"
        log.append((logical, detail))

    return log


# ═══════════════════════════════════════════════════════════════════════════
# QUERY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def q(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    timeit: bool = False,
    show_sql: bool = False,
) -> pd.DataFrame:
    """
    Run SQL and return a DataFrame.

    Parameters
    ----------
    timeit   : print the elapsed wall time. Useful when judging whether a query
               belongs in a notebook or in a materialisation step.
    show_sql : print the SQL before running it.
    """
    if show_sql:
        preview_sql(sql)
    start = time.perf_counter()
    frame = con.execute(sql).df()
    if timeit:
        print(f"[{time.perf_counter() - start:.2f}s, {len(frame):,} rows]")
    return frame


def preview_sql(sql: str) -> None:
    """
    Print SQL with line numbers, without running it.

    Mirrors the `preview_sql` habit from the Athena work: read the query first,
    then decide whether it is worth executing.
    """
    lines = [line for line in sql.strip().splitlines()]
    width = len(str(len(lines)))
    for number, line in enumerate(lines, start=1):
        print(f"{number:>{width}} | {line}")


def explain(con: duckdb.DuckDBPyConnection, sql: str, analyze: bool = False) -> None:
    """Print the DuckDB query plan. `analyze=True` runs the query and reports actual cost."""
    keyword = "EXPLAIN ANALYZE" if analyze else "EXPLAIN"
    for row in con.execute(f"{keyword} {sql}").fetchall():
        print(row[-1])


# ═══════════════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════════════

def _dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024**2
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


def store_status(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """
    One row per logical store table: whether it has been built, its size, and its
    row count if a connection is supplied.

    This is the SolarEdge analogue of `conformance_queries.table_provenance()` --
    it makes it impossible to run an analysis against a store you thought was
    complete but is not.
    """
    rows = []
    for logical in C.STORE_TABLES:
        path = C.store_path(logical)
        exists = path.exists()
        n_rows = None
        if exists and con is not None:
            try:
                n_rows = int(con.execute(f"SELECT count(*) FROM {logical}").fetchone()[0])
            except duckdb.Error:
                n_rows = None
        rows.append(
            {
                "logical_name": logical,
                "exists": exists,
                "kind": "partitioned" if logical in C.PARTITIONED_TABLES else "file",
                "size_mb": round(_dir_size_mb(path), 1) if exists else None,
                "n_rows": n_rows,
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def relations(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every view currently registered on the connection."""
    return con.execute(
        """
        SELECT table_name AS relation, table_type
        FROM information_schema.tables
        ORDER BY table_name
        """
    ).df()
