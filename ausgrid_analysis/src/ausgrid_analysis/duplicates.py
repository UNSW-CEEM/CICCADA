"""Duplicate-key audit for phase telemetry."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import FoundationConfig, SourceScope
from .db import (
    connect,
    duplicate_audit_path,
    duplicate_summary_path,
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

    connection = connect(config)
    try:
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
