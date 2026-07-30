"""Build analysis-ready interval tables without claiming inverter conformance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import FoundationConfig, SourceScope
from .db import (
    canonical_output_path,
    connect,
    prepare_output_directory,
    site_profile_path,
    structured_phase_output_path,
    structured_site_output_path,
)
from .logging_utils import get_logger
from .schemas import sql_string


def _glob(path) -> str:
    return str(path / "**" / "*.parquet")


def build_structured_phase(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Enrich each canonical phase row with interpretation and mapping fields."""

    canonical = canonical_output_path(config, scope)
    profiles = site_profile_path(config, scope)
    if not canonical.is_dir() or not profiles.is_file():
        raise FileNotFoundError("Canonical data and Delivery 2 profiles are required")
    output = prepare_output_directory(
        config, structured_phase_output_path(config, scope), overwrite=overwrite
    )
    connection = connect(config)
    try:
        canonical_glob = _glob(canonical)
        buckets = [
            int(row[0])
            for row in connection.execute(
                f"""SELECT DISTINCT site_bucket
                FROM read_parquet(
                    {sql_string(canonical_glob)}, hive_partitioning=true
                ) ORDER BY site_bucket"""
            ).fetchall()
        ]
        output.mkdir(parents=True, exist_ok=True)
        for bucket in buckets:
            bucket_output = output / f"site_bucket={bucket}"
            query = f"""
            SELECT
                c.*,
                (c.voltage_v IS NOT NULL AND c.voltage_physical_ok)
                    AS voltage_valid_for_analysis,
                (c.p_export_w IS NOT NULL AND c.q_absorbing_var IS NOT NULL)
                    AS power_measurement_available,
                c.current_a IS NOT NULL AS current_measurement_available,
                list_contains(
                    string_split(nullif(s.inferred_der_phases, ''), '|'),
                    c.phase
                ) AS is_inferred_der_phase,
                cast(c.timestamp_local AS DATE) AS local_date,
                hour(c.timestamp_local) AS local_hour,
                minute(c.timestamp_local) AS local_minute,
                cast(
                    date_diff(
                        'minute',
                        cast(c.timestamp_utc AS TIMESTAMP),
                        c.timestamp_local
                    ) AS INTEGER
                ) AS utc_offset_minutes,
                s.has_battery,
                s.phase_mapping_method,
                s.phase_mapping_confidence,
                s.phase_mapping_assessable,
                s.delivery2_primary_cohort,
                'net_meter' AS measurement_basis,
                'revenue_meter' AS voltage_measurement_location,
                false AS formal_inverter_conformance_assessable
            FROM read_parquet(
                {sql_string(canonical_glob)}, hive_partitioning = true
            ) c
            JOIN read_parquet({sql_string(profiles)}) s USING (serial)
            WHERE c.site_bucket = {bucket}
            """
            connection.execute(
                f"""COPY (
                    SELECT * EXCLUDE (site_bucket) FROM ({query})
                ) TO {sql_string(bucket_output)}
                (FORMAT PARQUET,
                 COMPRESSION {config.processing.parquet_compression},
                 PARTITION_BY (year_utc, month_utc))"""
            )
        row = connection.execute(
            f"""SELECT count(*), count(DISTINCT serial),
                count_if(is_inferred_der_phase)
                FROM read_parquet({sql_string(_glob(output))}, hive_partitioning=true)"""
        ).fetchone()
    finally:
        connection.close()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope.label,
        "rows": int(row[0]),
        "sites": int(row[1]),
        "rows_on_inferred_der_phases": int(row[2]),
        "output": str(output),
    }


def build_structured_site(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Aggregate phase telemetry while preserving incomplete-power states as null."""

    phase = structured_phase_output_path(config, scope)
    if not phase.is_dir():
        raise FileNotFoundError(f"Structured phase dataset is missing: {phase}")
    output = prepare_output_directory(
        config, structured_site_output_path(config, scope), overwrite=overwrite
    )
    connection = connect(config)
    try:
        phase_glob = _glob(phase)
        buckets = [
            int(row[0])
            for row in connection.execute(
                f"""SELECT DISTINCT site_bucket
                FROM read_parquet(
                    {sql_string(phase_glob)}, hive_partitioning=true
                ) ORDER BY site_bucket"""
            ).fetchall()
        ]
        output.mkdir(parents=True, exist_ok=True)
        for bucket in buckets:
            bucket_output = output / f"site_bucket={bucket}"
            query = f"""
            WITH x AS (
                SELECT *
                FROM read_parquet(
                    {sql_string(phase_glob)}, hive_partitioning = true
                )
                WHERE site_bucket = {bucket}
            ),
            g AS (
                SELECT
                    serial,
                    timestamp_utc,
                    timestamp_local,
                    local_date,
                    local_hour,
                    local_minute,
                    utc_offset_minutes,
                    any_value(metadata_available) AS metadata_available,
                    any_value(analysis_cohort) AS analysis_cohort,
                    any_value(has_battery) AS has_battery,
                    any_value(phase_mapping_method) AS phase_mapping_method,
                    any_value(phase_mapping_confidence) AS phase_mapping_confidence,
                    any_value(phase_mapping_assessable) AS phase_mapping_assessable,
                    any_value(delivery2_primary_cohort) AS delivery2_primary_cohort,
                    count(*) AS observed_phase_rows,
                    count_if(power_measurement_available) AS measured_power_phase_rows,
                    count_if(is_inferred_der_phase) AS expected_der_phase_rows,
                    count_if(is_inferred_der_phase AND power_measurement_available)
                        AS measured_der_phase_rows,
                    min(voltage_v) FILTER (WHERE voltage_valid_for_analysis)
                        AS voltage_min_valid_v,
                    avg(voltage_v) FILTER (WHERE voltage_valid_for_analysis)
                        AS voltage_mean_valid_v,
                    max(voltage_v) FILTER (WHERE voltage_valid_for_analysis)
                        AS voltage_max_valid_v,
                    min(voltage_v) FILTER (
                        WHERE voltage_valid_for_analysis AND is_inferred_der_phase
                    ) AS der_voltage_min_valid_v,
                    avg(voltage_v) FILTER (
                        WHERE voltage_valid_for_analysis AND is_inferred_der_phase
                    ) AS der_voltage_mean_valid_v,
                    max(voltage_v) FILTER (
                        WHERE voltage_valid_for_analysis AND is_inferred_der_phase
                    ) AS der_voltage_max_valid_v,
                    sum(p_export_w) FILTER (WHERE power_measurement_available)
                        AS p_export_net_observed_w,
                    sum(q_absorbing_var) FILTER (WHERE power_measurement_available)
                        AS q_absorbing_net_observed_var,
                    sum(p_export_w) FILTER (
                        WHERE is_inferred_der_phase AND power_measurement_available
                    ) AS p_export_der_phase_net_observed_w,
                    sum(q_absorbing_var) FILTER (
                        WHERE is_inferred_der_phase AND power_measurement_available
                    ) AS q_absorbing_der_phase_net_observed_var,
                    max(voltage_v) FILTER (WHERE phase='A') AS voltage_a_v,
                    max(voltage_v) FILTER (WHERE phase='B') AS voltage_b_v,
                    max(voltage_v) FILTER (WHERE phase='C') AS voltage_c_v,
                    sum(p_export_w) FILTER (WHERE phase='A') AS p_export_a_w,
                    sum(p_export_w) FILTER (WHERE phase='B') AS p_export_b_w,
                    sum(p_export_w) FILTER (WHERE phase='C') AS p_export_c_w,
                    year(timestamp_utc) AS year_utc,
                    month(timestamp_utc) AS month_utc,
                    any_value(site_bucket) AS site_bucket
                FROM x
                GROUP BY
                    serial, timestamp_utc, timestamp_local, local_date,
                    local_hour, local_minute, utc_offset_minutes
            )
            SELECT
                *,
                expected_der_phase_rows > 0
                    AND measured_der_phase_rows = expected_der_phase_rows
                    AS der_phase_power_complete,
                CASE WHEN expected_der_phase_rows > 0
                          AND measured_der_phase_rows = expected_der_phase_rows
                    THEN p_export_der_phase_net_observed_w END
                    AS p_export_der_phase_net_complete_w,
                CASE WHEN expected_der_phase_rows > 0
                          AND measured_der_phase_rows = expected_der_phase_rows
                    THEN q_absorbing_der_phase_net_observed_var END
                    AS q_absorbing_der_phase_net_complete_var,
                'net_meter' AS measurement_basis,
                'revenue_meter' AS voltage_measurement_location,
                false AS formal_inverter_conformance_assessable
            FROM g
            """
            connection.execute(
                f"""COPY (
                    SELECT * EXCLUDE (site_bucket) FROM ({query})
                ) TO {sql_string(bucket_output)}
                (FORMAT PARQUET,
                 COMPRESSION {config.processing.parquet_compression},
                 PARTITION_BY (year_utc, month_utc))"""
            )
        row = connection.execute(
            f"""SELECT count(*), count(DISTINCT serial),
                count_if(NOT der_phase_power_complete)
                FROM read_parquet({sql_string(_glob(output))}, hive_partitioning=true)"""
        ).fetchone()
    finally:
        connection.close()
    logger = get_logger()
    logger.info("Structured site intervals written: %s rows", row[0])
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope.label,
        "rows": int(row[0]),
        "sites": int(row[1]),
        "incomplete_der_power_rows": int(row[2]),
        "output": str(output),
    }
