"""Build canonical, deduplicated phase telemetry."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import FoundationConfig, SourceScope
from .db import (
    canonical_output_path,
    connect,
    duplicate_audit_path,
    prepare_output_directory,
    scope_root,
)
from .logging_utils import get_logger, write_json
from .metadata import metadata_output_path
from .schemas import scoped_source_sql, sql_string


def canonical_build_summary_path(
    config: FoundationConfig,
    scope: SourceScope,
) -> Path:
    return scope_root(config, scope) / "audit" / "canonical_build.json"


def canonical_query(
    config: FoundationConfig,
    scope: SourceScope,
    audit_path: Path,
    metadata_path: Path,
) -> str:
    """Return the exact canonical phase-table query."""

    active_sign = config.assumptions.active_export_sign
    absorbing_sign = config.assumptions.reactive_absorbing_sign
    source_timezone = config.assumptions.source_timezone.replace("'", "''")
    local_timezone = config.assumptions.local_timezone.replace("'", "''")
    v_min = config.quality.voltage_min_v
    v_max = config.quality.voltage_max_v
    bucket_count = config.processing.site_bucket_count
    source_sql = scoped_source_sql(config, scope)

    return f"""
        WITH source AS (
            {source_sql}
        ),
        audit AS (
            SELECT *
            FROM read_parquet({sql_string(audit_path)})
        ),
        eligible AS (
            SELECT
                s.*,
                coalesce(a.row_count, 1) AS duplicate_count,
                coalesce(a.source_file_count, 1) AS duplicate_source_file_count,
                coalesce(a.duplicate_class, 'unique') AS duplicate_status
            FROM source s
            LEFT JOIN audit a USING (serial, measure_time, phase)
            WHERE a.duplicate_class IS NULL
               OR a.duplicate_class = 'identical_duplicate'
            QUALIFY row_number() OVER (
                PARTITION BY s.serial, s.measure_time, s.phase
                ORDER BY
                    s.source_file,
                    s.source_month,
                    s.voltage_v,
                    s.current_a,
                    s.reactive_power_raw_var,
                    s.active_power_raw_w
            ) = 1
        )
        SELECT
            e.serial,
            timezone('{source_timezone}', e.measure_time) AS timestamp_utc,
            timezone(
                '{local_timezone}',
                timezone('{source_timezone}', e.measure_time)
            ) AS timestamp_local,
            e.phase,
            e.voltage_v,
            e.current_a,
            e.active_power_raw_w,
            e.reactive_power_raw_var,
            e.active_power_raw_w * {active_sign} AS p_export_w,
            e.reactive_power_raw_var * {absorbing_sign} AS q_absorbing_var,
            -e.reactive_power_raw_var * {absorbing_sign} AS q_generator_var,
            e.source_month,
            e.source_file,
            e.duplicate_count,
            e.duplicate_source_file_count,
            e.duplicate_status,
            (
                e.voltage_v IS NULL
                OR e.current_a IS NULL
                OR e.active_power_raw_w IS NULL
                OR e.reactive_power_raw_var IS NULL
            ) AS row_has_null_measurement,
            CASE
                WHEN e.voltage_v IS NULL THEN False
                ELSE e.voltage_v > {v_min} AND e.voltage_v <= {v_max}
            END AS voltage_physical_ok,
            m.serial IS NOT NULL AS metadata_available,
            m.analysis_cohort,
            m.install_phase_count,
            year(timezone(
                'UTC',
                timezone('{source_timezone}', e.measure_time)
            )) AS year_utc,
            month(timezone(
                'UTC',
                timezone('{source_timezone}', e.measure_time)
            )) AS month_utc,
            mod(hash(e.serial), {bucket_count}) AS site_bucket
        FROM eligible e
        LEFT JOIN read_parquet({sql_string(metadata_path)}) m USING (serial)
    """.strip()


def build_canonical_phase(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write partitioned canonical phase parquet for one scope."""

    logger = get_logger()
    audit_path = duplicate_audit_path(config, scope)
    metadata_path = metadata_output_path(config)
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Duplicate audit is required before canonicalisation: {audit_path}"
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Canonical metadata is required before canonicalisation: {metadata_path}"
        )

    output = prepare_output_directory(
        config,
        canonical_output_path(config, scope),
        overwrite=overwrite,
    )
    query = canonical_query(config, scope, audit_path, metadata_path)
    logger.info("Building canonical phase parquet for scope: %s", scope.label)

    connection = connect(config)
    try:
        connection.execute(
            f"""
            COPY ({query})
            TO {sql_string(output)}
            (
                FORMAT PARQUET,
                COMPRESSION {config.processing.parquet_compression},
                PARTITION_BY (year_utc, month_utc, site_bucket)
            )
            """
        )
        output_glob = str(output / "**" / "*.parquet")
        row = connection.execute(
            f"""
            SELECT
                count(*) AS n_rows,
                count(DISTINCT serial) AS n_serials,
                min(timestamp_utc) AS first_timestamp_utc,
                max(timestamp_utc) AS last_timestamp_utc,
                count_if(duplicate_status = 'identical_duplicate')
                    AS canonical_rows_from_duplicate_keys,
                count_if(NOT metadata_available) AS rows_without_metadata
            FROM read_parquet({sql_string(output_glob)}, hive_partitioning = true)
            """
        ).fetchone()
        names = [item[0] for item in connection.description]
        summary = dict(zip(names, row, strict=True))
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
        "output_directory": str(output),
        "active_export_sign": config.assumptions.active_export_sign,
        "reactive_absorbing_sign": config.assumptions.reactive_absorbing_sign,
        "local_timezone": config.assumptions.local_timezone,
        **summary,
    }
    write_json(canonical_build_summary_path(config, scope), payload)
    logger.info("Canonical phase parquet written to %s", output)
    logger.info("Canonical rows=%s; serials=%s", summary["n_rows"], summary["n_serials"])
    return payload
