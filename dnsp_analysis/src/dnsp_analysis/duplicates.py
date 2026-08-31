"""Duplicate-key audit for phase telemetry."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import FoundationConfig, SourceScope
from .db import (
    connect,
    duplicate_audit_path,
    duplicate_summary_path,
    prepare_output_directory,
    prepare_output_file,
    scope_root,
)
from .logging_utils import get_logger, write_json
from .schemas import (
    physical_payload_hash_sql,
    scoped_source_sql,
    sql_string,
)


def conflicting_rows_path(config: FoundationConfig, scope: SourceScope) -> Path:
    return scope_root(config, scope) / "audit" / "conflicting_duplicate_rows.parquet"


def duplicate_chunk_directory(
    config: FoundationConfig,
    scope: SourceScope,
    run_id: str,
) -> Path:
    return config.paths.temp_directory / f"duplicate_chunks_{scope.label}_{run_id}"


def _month_windows(
    first_timestamp: datetime,
    last_timestamp: datetime,
) -> list[tuple[str, datetime, datetime]]:
    cursor = first_timestamp.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    final_month = last_timestamp.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    windows: list[tuple[str, datetime, datetime]] = []
    while cursor <= final_month:
        if cursor.month == 12:
            next_month = cursor.replace(
                year=cursor.year + 1,
                month=1,
            )
        else:
            next_month = cursor.replace(month=cursor.month + 1)
        windows.append((cursor.strftime("%Y-%m"), cursor, next_month))
        cursor = next_month
    return windows


def _timestamp_sql(value: datetime) -> str:
    return f"TIMESTAMP '{value:%Y-%m-%d %H:%M:%S}'"


def duplicate_audit_query(source_sql: str, tolerance: float) -> str:
    payload_hash = physical_payload_hash_sql(tolerance, "s")
    return f"""
        WITH source AS (
            {source_sql}
        )
        SELECT
            s.serial,
            s.measure_time,
            s.phase,
            count(*) AS row_count,
            count(DISTINCT {payload_hash}) AS payload_variant_count,
            count(DISTINCT s.source_file) AS source_file_count,
            CASE
                WHEN count(DISTINCT {payload_hash}) = 1
                    THEN 'identical_duplicate'
                ELSE 'conflicting_duplicate'
            END AS duplicate_class
        FROM source s
        GROUP BY s.serial, s.measure_time, s.phase
        HAVING count(*) > 1
    """.strip()


def run_duplicate_audit(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Classify every repeated source key and retain conflicting source rows."""

    logger = get_logger()
    logger.info("Auditing duplicate keys for scope: %s", scope.label)
    audit_path = prepare_output_file(
        config,
        duplicate_audit_path(config, scope),
        overwrite=overwrite,
    )
    conflict_path = prepare_output_file(
        config,
        conflicting_rows_path(config, scope),
        overwrite=overwrite,
    )
    source_sql = scoped_source_sql(config, scope)
    audit_sql = duplicate_audit_query(
        source_sql,
        config.quality.duplicate_float_tolerance,
    )

    chunk_dir: Path | None = None
    connection = connect(config)
    try:
        if scope.is_full:
            # A whole-year GROUP BY has nearly one group per source row and can
            # exhaust memory. Timestamp-month windows are disjoint for the
            # canonical key, so auditing them separately cannot split a
            # duplicate key. Fewer threads also reduce aggregate state.
            audit_threads = min(config.processing.threads, 2)
            connection.execute(f"SET threads = {audit_threads}")
            bounds = connection.execute(
                f"""
                SELECT min(measure_time), max(measure_time)
                FROM ({source_sql}) AS source
                """
            ).fetchone()
            if bounds[0] is None or bounds[1] is None:
                connection.execute(
                    f"""
                    COPY ({audit_sql})
                    TO {sql_string(audit_path)}
                    (
                        FORMAT PARQUET,
                        COMPRESSION {config.processing.parquet_compression}
                    )
                    """
                )
            else:
                chunk_dir = prepare_output_directory(
                    config,
                    duplicate_chunk_directory(
                        config,
                        scope,
                        uuid4().hex,
                    ),
                    overwrite=False,
                )
                chunk_dir.mkdir(parents=True, exist_ok=True)
                for label, start, end in _month_windows(
                    bounds[0],
                    bounds[1],
                ):
                    logger.info("Auditing duplicate keys for %s", label)
                    monthly_source_sql = f"""
                        SELECT *
                        FROM ({source_sql}) AS source
                        WHERE measure_time >= {_timestamp_sql(start)}
                          AND measure_time < {_timestamp_sql(end)}
                    """.strip()
                    monthly_audit_sql = duplicate_audit_query(
                        monthly_source_sql,
                        config.quality.duplicate_float_tolerance,
                    )
                    chunk_path = chunk_dir / f"duplicate_keys_{label}.parquet"
                    connection.execute(
                        f"""
                        COPY ({monthly_audit_sql})
                        TO {sql_string(chunk_path)}
                        (
                            FORMAT PARQUET,
                            COMPRESSION {config.processing.parquet_compression}
                        )
                        """
                    )

                chunk_glob = str(chunk_dir / "*.parquet")
                connection.execute(
                    f"""
                    COPY (
                        SELECT *
                        FROM read_parquet({sql_string(chunk_glob)})
                    )
                    TO {sql_string(audit_path)}
                    (
                        FORMAT PARQUET,
                        COMPRESSION {config.processing.parquet_compression}
                    )
                    """
                )
        else:
            connection.execute(
                f"""
                COPY ({audit_sql})
                TO {sql_string(audit_path)}
                (
                    FORMAT PARQUET,
                    COMPRESSION {config.processing.parquet_compression}
                )
                """
            )
        connection.execute(
            f"""
            COPY (
                WITH source AS ({source_sql}),
                conflicts AS (
                    SELECT serial, measure_time, phase
                    FROM read_parquet({sql_string(audit_path)})
                    WHERE duplicate_class = 'conflicting_duplicate'
                )
                SELECT s.*
                FROM source s
                INNER JOIN conflicts c USING (serial, measure_time, phase)
                ORDER BY s.serial, s.measure_time, s.phase, s.source_file
            )
            TO {sql_string(conflict_path)}
            (
                FORMAT PARQUET,
                COMPRESSION {config.processing.parquet_compression}
            )
            """
        )

        row = connection.execute(
            f"""
            SELECT
                count(*) AS duplicate_keys,
                coalesce(sum(row_count), 0) AS rows_in_duplicate_keys,
                coalesce(sum(row_count - 1), 0) AS repeated_rows_above_one_per_key,
                coalesce(count_if(
                    duplicate_class = 'identical_duplicate'
                ), 0)
                    AS identical_duplicate_keys,
                coalesce(count_if(
                    duplicate_class = 'conflicting_duplicate'
                ), 0)
                    AS conflicting_duplicate_keys,
                coalesce(sum(
                    CASE WHEN duplicate_class = 'identical_duplicate'
                        THEN row_count - 1 ELSE 0 END
                ), 0) AS identical_rows_to_collapse,
                coalesce(sum(
                    CASE WHEN duplicate_class = 'conflicting_duplicate'
                        THEN row_count ELSE 0 END
                ), 0) AS conflicting_rows_to_quarantine,
                coalesce(count_if(source_file_count > 1), 0)
                    AS cross_file_duplicate_keys
            FROM read_parquet({sql_string(audit_path)})
            """
        ).fetchone()
        names = [item[0] for item in connection.description]
        summary = {
            key: int(value)
            for key, value in zip(names, row, strict=True)
        }
    finally:
        connection.close()
        if chunk_dir is not None and chunk_dir.exists():
            try:
                shutil.rmtree(chunk_dir)
            except OSError as exc:
                # Windows/OneDrive can briefly retain a directory handle after
                # DuckDB closes its parquet readers. The next run uses a new
                # unique directory, so a delayed cleanup must not invalidate a
                # completed audit.
                logger.warning(
                    "Could not remove temporary duplicate chunks at %s: %s",
                    chunk_dir,
                    exc,
                )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "label": scope.label,
            "month": scope.month,
            "site_bucket": scope.site_bucket,
            "bucket_count": scope.bucket_count,
        },
        "float_tolerance": config.quality.duplicate_float_tolerance,
        **summary,
        "duplicate_audit_parquet": str(audit_path),
        "conflicting_rows_parquet": str(conflict_path),
    }
    write_json(duplicate_summary_path(config, scope), payload)
    logger.info("Duplicate audit written to %s", audit_path)
    logger.info(
        "Duplicate keys=%s; identical=%s; conflicting=%s",
        summary["duplicate_keys"],
        summary["identical_duplicate_keys"],
        summary["conflicting_duplicate_keys"],
    )
    return payload
