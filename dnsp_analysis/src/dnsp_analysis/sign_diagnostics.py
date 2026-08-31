"""Deterministic empirical diagnostics for the working P/Q sign hypotheses."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .analysis_cohort import site_eligibility_path
from .config import FoundationConfig
from .db import (
    connect,
    prepare_output_file,
    structured_phase_output_path,
    structured_site_output_path,
)
from .mechanism_config import MechanismAnalysisConfig
from .mechanism_paths import (
    sign_candidate_days_path,
    sign_phase_intervals_path,
    sign_site_intervals_path,
)
from .power_conventions import q_generator_from_absorbing_sql
from .schemas import sql_string


def _glob(path) -> str:
    return str(path / "**" / "*.parquet")


def _core_site_gate_sql(alias: str = "e") -> str:
    return " AND ".join(
        (
            f"coalesce({alias}.gate_solar_only, false)",
            f"coalesce({alias}.gate_no_battery, false)",
            f"coalesce({alias}.gate_no_controlled_load, false)",
            f"coalesce({alias}.gate_mapping, false)",
            f"coalesce({alias}.gate_power_coverage, false)",
        )
    )


def build_sign_diagnostics(
    config: FoundationConfig,
    mechanism: MechanismAnalysisConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write three bounded files for manual review of 2–3 candidate site-days.

    Candidate selection uses no desired-response sign.  It ranks solar-only,
    no-battery, no-controlled-load site-days only by high-voltage exposure and
    positive derived net export, avoiding selection on the conclusion.
    """

    mechanism.validate()
    full_scope = config.scope(None, None)
    site_root = structured_site_output_path(config, full_scope)
    phase_root = structured_phase_output_path(config, full_scope)
    eligibility = site_eligibility_path(config)
    for path in (site_root, phase_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not eligibility.is_file():
        raise FileNotFoundError(eligibility)

    candidate_path = prepare_output_file(
        config, sign_candidate_days_path(config, mechanism), overwrite=overwrite
    )
    site_path = prepare_output_file(
        config, sign_site_intervals_path(config, mechanism), overwrite=overwrite
    )
    phase_path = prepare_output_file(
        config, sign_phase_intervals_path(config, mechanism), overwrite=overwrite
    )

    comparison_voltage = f"s.{mechanism.comparison_voltage_column}"
    p_export_column = mechanism.comparison_p_column
    q_absorbing_column = mechanism.comparison_q_absorbing_column
    q_generator = q_generator_from_absorbing_sql(f"s.{q_absorbing_column}")
    site_relation = (
        f"read_parquet({sql_string(_glob(site_root))}, hive_partitioning=true)"
    )
    phase_relation = (
        f"read_parquet({sql_string(_glob(phase_root))}, hive_partitioning=true)"
    )
    eligibility_relation = f"read_parquet({sql_string(eligibility)})"
    year, month = (int(part) for part in mechanism.sign_audit_month.split("-"))

    connection = connect(config)
    try:
        candidate_sql = f"""
        WITH daily AS (
            SELECT
                s.serial,
                s.local_date,
                count(*) AS n_day_intervals,
                min({comparison_voltage}) AS minimum_voltage_v,
                max({comparison_voltage}) AS maximum_voltage_v,
                count_if(
                    {comparison_voltage} BETWEEN 253.0 AND 258.0
                    AND s.{p_export_column} > 0
                ) AS n_high_voltage_export_intervals,
                max(s.{p_export_column}) AS maximum_export_w
            FROM {site_relation} s
            JOIN {eligibility_relation} e USING (serial)
            WHERE s.year_utc = {year}
              AND s.month_utc = {month}
              AND {_core_site_gate_sql("e")}
              AND {mechanism.power_scope_complete_sql}
              AND {comparison_voltage} IS NOT NULL
            GROUP BY s.serial, s.local_date
            HAVING maximum_voltage_v >= 253.0
               AND minimum_voltage_v < 253.0
               AND n_high_voltage_export_intervals >= 3
        ),
        best_per_site AS (
            SELECT *,
                row_number() OVER (
                    PARTITION BY serial
                    ORDER BY n_high_voltage_export_intervals DESC,
                             maximum_voltage_v DESC,
                             local_date
                ) AS site_day_rank
            FROM daily
        )
        SELECT * EXCLUDE (site_day_rank)
        FROM best_per_site
        WHERE site_day_rank = 1
        ORDER BY n_high_voltage_export_intervals DESC,
                 maximum_voltage_v DESC,
                 serial
        LIMIT {mechanism.sign_audit_site_count}
        """
        connection.execute(
            f"""COPY ({candidate_sql}) TO {sql_string(candidate_path)}
            (FORMAT PARQUET, COMPRESSION {config.processing.parquet_compression})"""
        )
        n_candidates = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({sql_string(candidate_path)})"
            ).fetchone()[0]
        )
        if n_candidates != mechanism.sign_audit_site_count:
            raise RuntimeError(
                f"Found {n_candidates} sign candidates; "
                f"expected {mechanism.sign_audit_site_count}"
            )

        site_sql = f"""
        SELECT
            s.serial,
            s.timestamp_utc,
            s.timestamp_local,
            s.local_date,
            s.local_hour,
            {comparison_voltage} AS comparison_voltage_v,
            s.der_voltage_min_valid_v,
            s.der_voltage_mean_valid_v,
            s.der_voltage_max_valid_v,
            s.voltage_a_v,
            s.voltage_b_v,
            s.voltage_c_v,
            s.{p_export_column} AS p_export_net_proxy_w,
            s.{q_absorbing_column} AS q_absorbing_net_proxy_var,
            {q_generator} AS q_generator_net_proxy_var,
            {mechanism.power_scope_complete_sql} AS power_scope_complete,
            {sql_string(mechanism.voltage_basis_label)} AS voltage_basis,
            'net_meter_proxy' AS measurement_basis,
            'revenue_meter' AS voltage_measurement_location
        FROM {site_relation} s
        JOIN read_parquet({sql_string(candidate_path)}) c
          ON s.serial = c.serial AND s.local_date = c.local_date
        ORDER BY s.serial, s.timestamp_utc
        """
        connection.execute(
            f"""COPY ({site_sql}) TO {sql_string(site_path)}
            (FORMAT PARQUET, COMPRESSION {config.processing.parquet_compression})"""
        )

        # Structured phase already persists q_generator_var under the canonical
        # schemas.normalize_reactive_power contract.  No second sign flip occurs.
        phase_sql = f"""
        SELECT
            p.serial,
            p.timestamp_utc,
            p.timestamp_local,
            p.local_date,
            p.phase,
            p.voltage_v,
            p.active_power_raw_w,
            p.p_export_w,
            p.reactive_power_raw_var,
            p.q_absorbing_var,
            p.q_generator_var,
            p.is_inferred_der_phase,
            p.power_measurement_available,
            p.source_file,
            'net_meter_proxy' AS measurement_basis,
            'revenue_meter' AS voltage_measurement_location
        FROM {phase_relation} p
        JOIN read_parquet({sql_string(candidate_path)}) c
          ON p.serial = c.serial AND p.local_date = c.local_date
        ORDER BY p.serial, p.timestamp_utc, p.phase
        """
        connection.execute(
            f"""COPY ({phase_sql}) TO {sql_string(phase_path)}
            (FORMAT PARQUET, COMPRESSION {config.processing.parquet_compression})"""
        )

        contract = connection.execute(
            f"""
            SELECT
                count(*) AS n_phase_rows,
                count_if(
                    p_export_w IS DISTINCT FROM
                    active_power_raw_w * {config.assumptions.active_export_sign}
                ) AS active_contract_errors,
                count_if(
                    q_absorbing_var IS DISTINCT FROM
                    reactive_power_raw_var
                    * {config.assumptions.reactive_absorbing_sign}
                ) AS absorbing_contract_errors,
                count_if(
                    q_generator_var IS DISTINCT FROM -q_absorbing_var
                ) AS generator_contract_errors
            FROM read_parquet({sql_string(phase_path)})
            """
        ).fetchone()
        n_site_rows = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({sql_string(site_path)})"
            ).fetchone()[0]
        )
    finally:
        connection.close()

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_month": mechanism.sign_audit_month,
        "candidate_sites": n_candidates,
        "site_interval_rows": n_site_rows,
        "phase_interval_rows": int(contract[0]),
        "active_contract_errors": int(contract[1]),
        "absorbing_contract_errors": int(contract[2]),
        "generator_contract_errors": int(contract[3]),
        "active_sign_review_state": mechanism.active_sign_review_state,
        "reactive_sign_review_state": mechanism.reactive_sign_review_state,
        "candidate_days": str(candidate_path),
        "site_intervals": str(site_path),
        "phase_intervals": str(phase_path),
    }
