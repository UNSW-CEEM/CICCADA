"""Source inventory and schema validation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import FoundationConfig, SourceScope
from .db import connect, scope_root
from .logging_utils import get_logger, write_json
from .schemas import scoped_source_sql, source_relation_sql


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def inventory_output_path(config: FoundationConfig, scope: SourceScope) -> Path:
    return scope_root(config, scope) / "_manifests" / "source_inventory.json"


def validate_source_schema(
    connection,
    config: FoundationConfig,
) -> list[dict[str, Any]]:
    schema = connection.execute(
        f"DESCRIBE SELECT * FROM {source_relation_sql(config)}"
    ).fetchdf()
    available = set(schema["column_name"].astype(str))
    missing = sorted(set(config.telemetry.required_columns) - available)
    if missing:
        raise ValueError(f"Telemetry parquet is missing required columns: {missing}")
    return schema.to_dict(orient="records")


def run_inventory(
    config: FoundationConfig,
    scope: SourceScope,
) -> dict[str, Any]:
    """Inspect source schema, cardinality, time coverage and source partitions."""

    logger = get_logger()
    logger.info("Inventorying source scope: %s", scope.label)
    connection = connect(config)
    try:
        schema = validate_source_schema(connection, config)
        source_sql = scoped_source_sql(config, scope)
        stats = connection.execute(
            f"""
            SELECT
                count(*) AS n_rows,
                count(DISTINCT serial) AS n_serials,
                min(measure_time) AS first_measure_time,
                max(measure_time) AS last_measure_time,
                count(DISTINCT phase) AS n_phases,
                count(DISTINCT source_month) AS n_source_months,
                count(DISTINCT source_file) AS n_source_files,
                count_if(measure_time IS NULL) AS null_timestamps,
                count_if(serial IS NULL OR trim(serial) = '') AS null_or_blank_serials
            FROM ({source_sql}) AS s
            """
        ).fetchone()
        names = [item[0] for item in connection.description]
        overall = dict(zip(names, stats, strict=True))

        by_month = connection.execute(
            f"""
            SELECT
                source_month,
                count(*) AS n_rows,
                count(DISTINCT serial) AS n_serials,
                min(measure_time) AS first_measure_time,
                max(measure_time) AS last_measure_time
            FROM ({source_sql}) AS s
            GROUP BY source_month
            ORDER BY source_month
            """
        ).fetchdf()

        by_phase = connection.execute(
            f"""
            SELECT phase, count(*) AS n_rows, count(DISTINCT serial) AS n_serials
            FROM ({source_sql}) AS s
            GROUP BY phase
            ORDER BY phase
            """
        ).fetchdf()
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
        "telemetry_file": _file_info(config.paths.telemetry_parquet),
        "metadata_file": _file_info(config.paths.metadata_workbook),
        "telemetry_schema": schema,
        "overall": overall,
        "by_source_month": by_month.to_dict(orient="records"),
        "by_phase": by_phase.to_dict(orient="records"),
        "assumptions": {
            "active_export_sign": config.assumptions.active_export_sign,
            "reactive_absorbing_sign": config.assumptions.reactive_absorbing_sign,
            "source_timezone": config.assumptions.source_timezone,
            "local_timezone": config.assumptions.local_timezone,
            "power_sample_type": config.assumptions.power_sample_type,
            "measurement_location": config.assumptions.measurement_location,
        },
    }
    output = inventory_output_path(config, scope)
    write_json(output, payload)
    logger.info("Source inventory written to %s", output)
    return payload

