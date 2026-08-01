"""Validation for mechanism result denominators, keys and provenance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import FoundationConfig, SourceScope
from .db import (
    connect,
    structured_phase_output_path,
    structured_site_output_path,
)
from .logging_utils import write_json
from .mechanism_paths import (
    mechanism_validation_path,
    response_observability_path,
    voltvar_results_path,
    voltwatt_results_path,
)
from .schemas import sql_string


def _glob(path) -> str:
    return str(path / "**" / "*.parquet")


def validate_mechanism_results(
    config: FoundationConfig,
    scope: SourceScope,
) -> dict[str, Any]:
    """Validate all three Delivery 4 tables without blending their questions."""

    site = structured_site_output_path(config, scope)
    phase = structured_phase_output_path(config, scope)
    vv = voltvar_results_path(config, scope)
    vw = voltwatt_results_path(config, scope)
    response = response_observability_path(config, scope)
    for path in (site, phase):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (vv, vw, response):
        if not path.is_file():
            raise FileNotFoundError(path)

    connection = connect(config)
    try:
        site_relation = (
            f"read_parquet({sql_string(_glob(site))}, hive_partitioning=true)"
        )
        phase_relation = (
            f"read_parquet({sql_string(_glob(phase))}, hive_partitioning=true)"
        )
        source_site_rows = int(
            connection.execute(f"SELECT count(*) FROM {site_relation}").fetchone()[0]
        )
        source_phase_rows = int(
            connection.execute(f"SELECT count(*) FROM {phase_relation}").fetchone()[0]
        )

        vv_row = connection.execute(
            f"""
            SELECT
                count(*) AS result_rows,
                coalesce(sum(n_source_intervals), 0) AS source_intervals,
                coalesce(sum(
                    n_source_intervals
                    - n_ineligible_site
                    - n_missing_input
                    - n_not_activated
                    - n_sign_unverified
                    - n_capacity_unavailable
                    - n_below_minimum_active_power
                    - n_assessable
                ), 0) AS denominator_difference,
                coalesce(sum(
                    n_assessable
                    - n_proxy_within_curve_band
                    - n_proxy_q_adverse
                    - n_proxy_q_inactive
                    - n_proxy_q_significant_shortfall
                    - n_proxy_q_near_conformant
                    - n_proxy_q_major_surplus
                ), 0) AS classification_difference,
                (SELECT coalesce(sum(n - 1), 0) FROM (
                    SELECT count(*) n
                    FROM read_parquet({sql_string(vv)})
                    GROUP BY serial, year_utc, month_utc, phase_scope,
                             voltage_bin_lower_v
                    HAVING n > 1
                )) AS duplicate_keys,
                count_if(measurement_basis <> 'net_meter_proxy'
                    OR voltage_measurement_location <> 'revenue_meter'
                    OR formal_inverter_conformance_assessable)
                    AS provenance_errors,
                coalesce(sum(n_assessable), 0) AS assessable_intervals,
                coalesce(sum(n_capacity_unavailable), 0)
                    AS capacity_unavailable_intervals
            FROM read_parquet({sql_string(vv)})
            """
        ).fetchone()

        vw_row = connection.execute(
            f"""
            SELECT
                count(*) AS result_rows,
                coalesce(sum(n_source_intervals), 0) AS source_intervals,
                coalesce(sum(
                    n_source_intervals
                    - n_ineligible_site
                    - n_missing_input
                    - n_not_activated
                    - n_sign_unverified
                    - n_not_exporting
                    - n_capacity_unavailable
                    - n_assessable
                ), 0) AS denominator_difference,
                coalesce(sum(
                    n_assessable
                    - n_proxy_exceeds_curve_ceiling
                    - n_proxy_does_not_exceed_curve_ceiling
                ), 0) AS classification_difference,
                (SELECT coalesce(sum(n - 1), 0) FROM (
                    SELECT count(*) n
                    FROM read_parquet({sql_string(vw)})
                    GROUP BY serial, year_utc, month_utc, phase_scope,
                             voltage_bin_lower_v
                    HAVING n > 1
                )) AS duplicate_keys,
                count_if(measurement_basis <> 'net_meter_proxy'
                    OR voltage_measurement_location <> 'revenue_meter'
                    OR formal_inverter_conformance_assessable)
                    AS provenance_errors,
                coalesce(sum(n_assessable), 0) AS assessable_intervals,
                coalesce(sum(n_capacity_unavailable), 0)
                    AS capacity_unavailable_intervals
            FROM read_parquet({sql_string(vw)})
            """
        ).fetchone()

        response_row = connection.execute(
            f"""
            SELECT
                count(*) AS result_rows,
                coalesce(sum(n_source_intervals), 0) AS source_intervals,
                (SELECT coalesce(sum(n - 1), 0) FROM (
                    SELECT count(*) n
                    FROM read_parquet({sql_string(response)})
                    GROUP BY serial, year_utc, month_utc, phase
                    HAVING n > 1
                )) AS duplicate_keys,
                count_if(measurement_basis <> 'net_meter_proxy'
                    OR voltage_measurement_location <> 'revenue_meter'
                    OR NOT observability_only
                    OR formal_inverter_conformance_assessable)
                    AS provenance_errors
            FROM read_parquet({sql_string(response)})
            """
        ).fetchone()

        dst_collision_groups = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                    SELECT serial, timestamp_local
                    FROM {site_relation}
                    GROUP BY serial, timestamp_local
                    HAVING count(DISTINCT timestamp_utc) > 1
                )
                """
            ).fetchone()[0]
        )
        monthly = connection.execute(
            f"""
            SELECT
                year_utc,
                month_utc,
                sum(n_source_intervals) AS n_source_intervals,
                sum(n_assessable) AS n_voltvar_assessable
            FROM read_parquet({sql_string(vv)})
            GROUP BY year_utc, month_utc
            ORDER BY year_utc, month_utc
            """
        ).fetchdf().to_dict(orient="records")
    finally:
        connection.close()

    failures: list[str] = []
    if int(vv_row[1]) != source_site_rows:
        failures.append("Volt-VAr source denominator differs from structured site rows")
    if int(vw_row[1]) != source_site_rows:
        failures.append("Volt-Watt source denominator differs from structured site rows")
    if int(response_row[1]) != source_phase_rows:
        failures.append(
            "response-observability denominator differs from structured phase rows"
        )
    if int(vv_row[2]) or int(vv_row[3]):
        failures.append("Volt-VAr denominator/classification accounting failed")
    if int(vw_row[2]) or int(vw_row[3]):
        failures.append("Volt-Watt denominator/classification accounting failed")
    if int(vv_row[4]) or int(vw_row[4]) or int(response_row[2]):
        failures.append("one or more result keys are not unique")
    if int(vv_row[5]) or int(vw_row[5]) or int(response_row[3]):
        failures.append("one or more result rows violate provenance guardrails")

    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope.label,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "structured_site_rows": source_site_rows,
        "structured_phase_rows": source_phase_rows,
        "voltvar_result_rows": int(vv_row[0]),
        "voltvar_source_intervals": int(vv_row[1]),
        "voltvar_denominator_difference": int(vv_row[2]),
        "voltvar_classification_difference": int(vv_row[3]),
        "voltvar_duplicate_keys": int(vv_row[4]),
        "voltvar_provenance_errors": int(vv_row[5]),
        "voltvar_assessable_intervals": int(vv_row[6]),
        "voltvar_capacity_unavailable_intervals": int(vv_row[7]),
        "voltwatt_result_rows": int(vw_row[0]),
        "voltwatt_source_intervals": int(vw_row[1]),
        "voltwatt_denominator_difference": int(vw_row[2]),
        "voltwatt_classification_difference": int(vw_row[3]),
        "voltwatt_duplicate_keys": int(vw_row[4]),
        "voltwatt_provenance_errors": int(vw_row[5]),
        "voltwatt_assessable_intervals": int(vw_row[6]),
        "voltwatt_capacity_unavailable_intervals": int(vw_row[7]),
        "response_result_rows": int(response_row[0]),
        "response_source_intervals": int(response_row[1]),
        "response_duplicate_keys": int(response_row[2]),
        "response_provenance_errors": int(response_row[3]),
        "source_dst_local_collision_groups": dst_collision_groups,
        "result_key_time_basis": "UTC-derived year/month; never timestamp_local",
        "counterfactual_supported_curtailment": "not_built_gate_7_unmet",
        "monthly_coverage": monthly,
    }
    write_json(mechanism_validation_path(config, scope), payload)
    return payload
