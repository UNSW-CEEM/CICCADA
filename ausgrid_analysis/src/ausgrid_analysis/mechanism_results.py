"""Build compact, auditable mechanism-result tables.

The outputs are net-meter proxies, never direct inverter measurements.  Volt-
VAr and Volt-Watt tables retain separate denominators and statuses.  Response
observability is a third, independent question.  There is intentionally no
curtailment builder while methodology gate 7 remains unmet.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis_cohort import site_eligibility_path
from .as4777_curves import (
    Q_CAPABILITY,
    Q_IMPACT_THRESHOLDS,
    VOLT_VAR,
    VOLT_WATT,
    q_conformance_floor_absorbing_sql,
    q_impact_nearest_edge_sql,
    vvar_required_q_sql,
    vw_max_p_sql,
)
from .config import FoundationConfig, SourceScope
from .db import (
    connect,
    prepare_output_file,
    structured_phase_output_path,
    structured_site_output_path,
)
from .mechanism_config import MechanismAnalysisConfig
from .mechanism_paths import (
    response_observability_path,
    voltvar_results_path,
    voltwatt_results_path,
)
from .power_conventions import q_generator_from_absorbing_sql
from .schemas import sql_string


def _glob(path: Path) -> str:
    return str(path / "**" / "*.parquet")


def _bool_sql(value: bool) -> str:
    return "true" if value else "false"


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


def _check_inputs(
    config: FoundationConfig,
    scope: SourceScope,
) -> tuple[Path, Path, Path]:
    site = structured_site_output_path(config, scope)
    phase = structured_phase_output_path(config, scope)
    eligibility = site_eligibility_path(config)
    for path in (site, phase):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not eligibility.is_file():
        raise FileNotFoundError(eligibility)
    return site, phase, eligibility


def _base_site_sql(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
) -> str:
    site, _, eligibility = _check_inputs(config, scope)
    voltage = f"s.{mechanism.comparison_voltage_column}"
    p_export = f"s.{mechanism.comparison_p_column}"
    q_absorbing = f"s.{mechanism.comparison_q_absorbing_column}"
    # This is the one site-level conversion from the persisted positive-
    # absorbing source.  It cross-references schemas.normalize_reactive_power.
    q_generator = q_generator_from_absorbing_sql(
        f"s.{mechanism.comparison_q_absorbing_column}"
    )
    width = mechanism.voltage_bin_width_v
    return f"""
        SELECT
            s.serial,
            s.timestamp_utc,
            s.timestamp_local,
            s.year_utc,
            s.month_utc,
            {mechanism.phase_scope_sql} AS phase_scope,
            {voltage} AS comparison_voltage_v,
            CASE WHEN {voltage} IS NOT NULL
                THEN floor({voltage} / {width}) * {width}
            END AS voltage_bin_lower_v,
            s.der_voltage_min_valid_v,
            s.der_voltage_mean_valid_v,
            s.der_voltage_max_valid_v,
            s.voltage_min_valid_v,
            s.voltage_mean_valid_v,
            s.voltage_max_valid_v,
            s.voltage_a_v,
            s.voltage_b_v,
            s.voltage_c_v,
            {p_export} AS p_export_net_proxy_w,
            {q_absorbing} AS q_absorbing_net_proxy_var,
            {q_generator} AS q_generator_net_proxy_var,
            {mechanism.power_scope_complete_sql} AS power_scope_complete,
            ({_core_site_gate_sql("e")}) AS site_eligible,
            e.analysis_cohort,
            coalesce(e.has_battery, false) AS has_battery,
            e.controlled_load_status,
            e.phase_mapping_confidence,
            e.s_rated_kva * 1000.0 AS capacity_reference_va,
            e.s_rated_source AS capacity_source
        FROM read_parquet(
            {sql_string(_glob(site))}, hive_partitioning=true
        ) s
        LEFT JOIN read_parquet({sql_string(eligibility)}) e USING (serial)
    """.strip()


def _voltvar_sql(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
) -> str:
    base = _base_site_sql(config, scope, mechanism)
    q_required = vvar_required_q_sql(
        "comparison_voltage_v", "capacity_reference_va"
    )
    q_floor = q_conformance_floor_absorbing_sql(
        "p_export_net_proxy_w", "capacity_reference_va"
    )
    tol = mechanism.tolerance_fraction
    sign_ready = _bool_sql(
        mechanism.active_sign_ready and mechanism.reactive_sign_ready
    )
    q_impact = q_impact_nearest_edge_sql(
        "q_generator_net_proxy_var",
        "q_min_final_var",
        "q_max_final_var",
        assessable_sql="denominator_status = 'assessable'",
    )
    threshold_1, threshold_2, threshold_3, threshold_4 = Q_IMPACT_THRESHOLDS
    return f"""
    WITH base AS ({base}),
    denominator AS (
        SELECT *,
            comparison_voltage_v < {VOLT_VAR.v2}
                OR comparison_voltage_v > {VOLT_VAR.v3} AS curve_active,
            CASE
                WHEN NOT site_eligible THEN 'ineligible_site'
                WHEN comparison_voltage_v IS NULL
                  OR NOT power_scope_complete
                  OR p_export_net_proxy_w IS NULL
                  OR q_generator_net_proxy_var IS NULL THEN 'missing_input'
                WHEN comparison_voltage_v BETWEEN {VOLT_VAR.v2} AND {VOLT_VAR.v3}
                    THEN 'not_activated'
                WHEN NOT {sign_ready} THEN 'sign_unverified'
                WHEN capacity_reference_va IS NULL OR capacity_reference_va <= 0
                    THEN 'capacity_unavailable'
                WHEN abs(p_export_net_proxy_w)
                    < {Q_CAPABILITY.p_min} * capacity_reference_va
                    THEN 'below_minimum_active_power'
                ELSE 'assessable'
            END AS denominator_status
        FROM base
    ),
    curves AS (
        SELECT *,
            {q_required} AS q_required_var,
            {q_floor} AS q_capability_absorbing_var,
            {tol} * capacity_reference_va AS tolerance_var
        FROM denominator
    ),
    unclamped AS (
        SELECT *,
            q_required_var - tolerance_var AS q_unclamped_min_var,
            q_required_var + tolerance_var AS q_unclamped_max_var
        FROM curves
    ),
    bands AS (
        SELECT *,
            CASE WHEN q_unclamped_max_var < 0
                THEN greatest(
                    q_unclamped_max_var,
                    q_capability_absorbing_var + tolerance_var
                )
                ELSE q_unclamped_max_var
            END AS q_max_final_var,
            CASE WHEN q_unclamped_min_var > 0
                THEN least(
                    q_unclamped_min_var,
                    -q_capability_absorbing_var - tolerance_var
                )
                ELSE q_unclamped_min_var
            END AS q_min_final_var
        FROM unclamped
    ),
    scored AS (
        SELECT *, {q_impact} AS q_impact
        FROM bands
    ),
    classified AS (
        SELECT *,
            CASE
                WHEN denominator_status <> 'assessable' THEN denominator_status
                WHEN q_generator_net_proxy_var BETWEEN q_min_final_var
                                                   AND q_max_final_var
                    THEN 'proxy_within_curve_band'
                WHEN q_impact < {threshold_1} THEN 'proxy_q_adverse'
                WHEN q_impact <= {threshold_2} THEN 'proxy_q_inactive'
                WHEN q_impact < {threshold_3} THEN 'proxy_q_significant_shortfall'
                WHEN q_impact <= {threshold_4} THEN 'proxy_q_near_conformant'
                ELSE 'proxy_q_major_surplus'
            END AS proxy_curve_status
        FROM scored
    )
    SELECT
        serial,
        year_utc,
        month_utc,
        phase_scope,
        voltage_bin_lower_v,
        any_value(analysis_cohort) AS analysis_cohort,
        any_value(phase_mapping_confidence) AS phase_mapping_confidence,
        min(comparison_voltage_v) AS minimum_comparison_voltage_v,
        max(comparison_voltage_v) AS maximum_comparison_voltage_v,
        count(*) AS n_source_intervals,
        count_if(denominator_status = 'ineligible_site') AS n_ineligible_site,
        count_if(denominator_status = 'missing_input') AS n_missing_input,
        count_if(denominator_status = 'not_activated') AS n_not_activated,
        count_if(denominator_status = 'sign_unverified') AS n_sign_unverified,
        count_if(denominator_status = 'capacity_unavailable')
            AS n_capacity_unavailable,
        count_if(denominator_status = 'below_minimum_active_power')
            AS n_below_minimum_active_power,
        count_if(denominator_status = 'assessable') AS n_assessable,
        count_if(proxy_curve_status = 'proxy_within_curve_band')
            AS n_proxy_within_curve_band,
        count_if(proxy_curve_status = 'proxy_q_adverse') AS n_proxy_q_adverse,
        count_if(proxy_curve_status = 'proxy_q_inactive') AS n_proxy_q_inactive,
        count_if(proxy_curve_status = 'proxy_q_significant_shortfall')
            AS n_proxy_q_significant_shortfall,
        count_if(proxy_curve_status = 'proxy_q_near_conformant')
            AS n_proxy_q_near_conformant,
        count_if(proxy_curve_status = 'proxy_q_major_surplus')
            AS n_proxy_q_major_surplus,
        avg(q_impact) FILTER (WHERE denominator_status = 'assessable')
            AS mean_q_impact,
        {sql_string(mechanism.methodology_id)} AS methodology_id,
        'net_meter_proxy' AS measurement_basis,
        'revenue_meter' AS voltage_measurement_location,
        {sql_string(mechanism.voltage_basis_label)} AS voltage_basis,
        {sql_string(mechanism.capacity_basis)} AS capacity_basis,
        {sql_string(mechanism.active_sign_review_state)}
            AS active_sign_review_state,
        {sql_string(mechanism.reactive_sign_review_state)}
            AS reactive_sign_review_state,
        false AS formal_inverter_conformance_assessable
    FROM classified
    GROUP BY serial, year_utc, month_utc, phase_scope, voltage_bin_lower_v
    ORDER BY serial, year_utc, month_utc, phase_scope, voltage_bin_lower_v
    """.strip()


def _voltwatt_sql(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
) -> str:
    base = _base_site_sql(config, scope, mechanism)
    p_ceiling = vw_max_p_sql("comparison_voltage_v", "capacity_reference_va")
    tol = mechanism.tolerance_fraction
    sign_ready = _bool_sql(mechanism.active_sign_ready)
    return f"""
    WITH base AS ({base}),
    denominator AS (
        SELECT *,
            CASE
                WHEN NOT site_eligible THEN 'ineligible_site'
                WHEN comparison_voltage_v IS NULL
                  OR NOT power_scope_complete
                  OR p_export_net_proxy_w IS NULL THEN 'missing_input'
                WHEN comparison_voltage_v <= {VOLT_WATT.v1} THEN 'not_activated'
                WHEN NOT {sign_ready} THEN 'sign_unverified'
                WHEN p_export_net_proxy_w <= 0 THEN 'not_exporting'
                WHEN capacity_reference_va IS NULL OR capacity_reference_va <= 0
                    THEN 'capacity_unavailable'
                ELSE 'assessable'
            END AS denominator_status
        FROM base
    ),
    scored AS (
        SELECT *,
            {p_ceiling} AS p_curve_ceiling_w,
            ({p_ceiling}) + {tol} * capacity_reference_va
                AS p_curve_ceiling_with_tolerance_w
        FROM denominator
    ),
    classified AS (
        SELECT *,
            CASE
                WHEN denominator_status <> 'assessable' THEN denominator_status
                WHEN p_export_net_proxy_w > p_curve_ceiling_with_tolerance_w
                    THEN 'proxy_exceeds_curve_ceiling'
                ELSE 'proxy_does_not_exceed_curve_ceiling'
            END AS proxy_curve_status
        FROM scored
    )
    SELECT
        serial,
        year_utc,
        month_utc,
        phase_scope,
        voltage_bin_lower_v,
        any_value(analysis_cohort) AS analysis_cohort,
        any_value(phase_mapping_confidence) AS phase_mapping_confidence,
        min(comparison_voltage_v) AS minimum_comparison_voltage_v,
        max(comparison_voltage_v) AS maximum_comparison_voltage_v,
        count(*) AS n_source_intervals,
        count_if(denominator_status = 'ineligible_site') AS n_ineligible_site,
        count_if(denominator_status = 'missing_input') AS n_missing_input,
        count_if(denominator_status = 'not_activated') AS n_not_activated,
        count_if(denominator_status = 'sign_unverified') AS n_sign_unverified,
        count_if(denominator_status = 'not_exporting') AS n_not_exporting,
        count_if(denominator_status = 'capacity_unavailable')
            AS n_capacity_unavailable,
        count_if(denominator_status = 'assessable') AS n_assessable,
        count_if(proxy_curve_status = 'proxy_exceeds_curve_ceiling')
            AS n_proxy_exceeds_curve_ceiling,
        count_if(proxy_curve_status = 'proxy_does_not_exceed_curve_ceiling')
            AS n_proxy_does_not_exceed_curve_ceiling,
        {sql_string(mechanism.methodology_id)} AS methodology_id,
        'net_meter_proxy' AS measurement_basis,
        'revenue_meter' AS voltage_measurement_location,
        {sql_string(mechanism.voltage_basis_label)} AS voltage_basis,
        {sql_string(mechanism.capacity_basis)} AS capacity_basis,
        {sql_string(mechanism.active_sign_review_state)}
            AS active_sign_review_state,
        false AS formal_inverter_conformance_assessable,
        'Below-ceiling net export is not proof of inverter conformance'
            AS interpretation_guardrail
    FROM classified
    GROUP BY serial, year_utc, month_utc, phase_scope, voltage_bin_lower_v
    ORDER BY serial, year_utc, month_utc, phase_scope, voltage_bin_lower_v
    """.strip()


def _response_observability_sql(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
) -> str:
    _, phase, eligibility = _check_inputs(config, scope)
    phase_relation = (
        f"read_parquet({sql_string(_glob(phase))}, hive_partitioning=true)"
    )
    eligibility_relation = f"read_parquet({sql_string(eligibility)})"
    active_ready = _bool_sql(mechanism.active_sign_ready)
    reactive_ready = _bool_sql(mechanism.reactive_sign_ready)
    n_min = mechanism.minimum_response_intervals
    span_min = mechanism.minimum_response_voltage_span_v
    return f"""
    WITH base AS (
        SELECT
            p.serial,
            p.year_utc,
            p.month_utc,
            p.phase,
            p.voltage_v,
            p.p_export_w,
            p.q_generator_var,
            p.voltage_valid_for_analysis,
            p.power_measurement_available,
            p.is_inferred_der_phase,
            ({_core_site_gate_sql("e")}) AS site_eligible,
            e.analysis_cohort,
            e.phase_mapping_confidence
        FROM {phase_relation} p
        LEFT JOIN {eligibility_relation} e USING (serial)
    ),
    aggregated AS (
        SELECT
            serial,
            year_utc,
            month_utc,
            phase,
            any_value(analysis_cohort) AS analysis_cohort,
            any_value(phase_mapping_confidence) AS phase_mapping_confidence,
            bool_or(site_eligible) AS site_eligible,
            bool_or(is_inferred_der_phase) AS inferred_der_phase,
            count(*) AS n_source_intervals,
            count_if(
                site_eligible AND is_inferred_der_phase
                AND voltage_valid_for_analysis AND power_measurement_available
            ) AS n_valid_power_voltage,
            count_if(
                site_eligible AND is_inferred_der_phase
                AND voltage_valid_for_analysis AND power_measurement_available
                AND voltage_v > {VOLT_VAR.v3}
            ) AS n_voltvar_excited_intervals,
            min(voltage_v) FILTER (
                WHERE site_eligible AND is_inferred_der_phase
                  AND voltage_valid_for_analysis AND power_measurement_available
                  AND voltage_v > {VOLT_VAR.v3}
            ) AS voltvar_minimum_voltage_v,
            max(voltage_v) FILTER (
                WHERE site_eligible AND is_inferred_der_phase
                  AND voltage_valid_for_analysis AND power_measurement_available
                  AND voltage_v > {VOLT_VAR.v3}
            ) AS voltvar_maximum_voltage_v,
            regr_slope(q_generator_var, voltage_v) FILTER (
                WHERE site_eligible AND is_inferred_der_phase
                  AND voltage_valid_for_analysis AND power_measurement_available
                  AND voltage_v > {VOLT_VAR.v3}
            ) AS q_generator_slope_var_per_v,
            corr(q_generator_var, voltage_v) FILTER (
                WHERE site_eligible AND is_inferred_der_phase
                  AND voltage_valid_for_analysis AND power_measurement_available
                  AND voltage_v > {VOLT_VAR.v3}
            ) AS q_generator_voltage_correlation,
            count_if(
                site_eligible AND is_inferred_der_phase
                AND voltage_valid_for_analysis AND power_measurement_available
                AND voltage_v > {VOLT_WATT.v1} AND p_export_w > 0
            ) AS n_voltwatt_excited_export_intervals,
            min(voltage_v) FILTER (
                WHERE site_eligible AND is_inferred_der_phase
                  AND voltage_valid_for_analysis AND power_measurement_available
                  AND voltage_v > {VOLT_WATT.v1} AND p_export_w > 0
            ) AS voltwatt_minimum_voltage_v,
            max(voltage_v) FILTER (
                WHERE site_eligible AND is_inferred_der_phase
                  AND voltage_valid_for_analysis AND power_measurement_available
                  AND voltage_v > {VOLT_WATT.v1} AND p_export_w > 0
            ) AS voltwatt_maximum_voltage_v,
            regr_slope(p_export_w, voltage_v) FILTER (
                WHERE site_eligible AND is_inferred_der_phase
                  AND voltage_valid_for_analysis AND power_measurement_available
                  AND voltage_v > {VOLT_WATT.v1} AND p_export_w > 0
            ) AS p_export_slope_w_per_v,
            corr(p_export_w, voltage_v) FILTER (
                WHERE site_eligible AND is_inferred_der_phase
                  AND voltage_valid_for_analysis AND power_measurement_available
                  AND voltage_v > {VOLT_WATT.v1} AND p_export_w > 0
            ) AS p_export_voltage_correlation
        FROM base
        GROUP BY serial, year_utc, month_utc, phase
    )
    SELECT *,
        (voltvar_maximum_voltage_v - voltvar_minimum_voltage_v)
            AS voltvar_voltage_span_v,
        (voltwatt_maximum_voltage_v - voltwatt_minimum_voltage_v)
            AS voltwatt_voltage_span_v,
        CASE
            WHEN NOT site_eligible THEN 'ineligible_site'
            WHEN NOT inferred_der_phase THEN 'not_inferred_der_phase'
            WHEN NOT {reactive_ready} THEN 'sign_unverified'
            WHEN n_voltvar_excited_intervals < {n_min}
              OR coalesce(
                    voltvar_maximum_voltage_v - voltvar_minimum_voltage_v,
                    0
                 ) < {span_min}
                THEN 'insufficient_excitation'
            WHEN q_generator_slope_var_per_v < 0
                THEN 'expected_direction_observed'
            ELSE 'opposite_or_flat_direction_observed'
        END AS voltvar_observability_status,
        CASE
            WHEN NOT site_eligible THEN 'ineligible_site'
            WHEN NOT inferred_der_phase THEN 'not_inferred_der_phase'
            WHEN NOT {active_ready} THEN 'sign_unverified'
            WHEN n_voltwatt_excited_export_intervals < {n_min}
              OR coalesce(
                    voltwatt_maximum_voltage_v - voltwatt_minimum_voltage_v,
                    0
                 ) < {span_min}
                THEN 'insufficient_excitation'
            WHEN p_export_slope_w_per_v < 0 THEN 'drop_direction_observed'
            ELSE 'opposite_or_flat_direction_observed'
        END AS voltwatt_observability_status,
        {sql_string(mechanism.methodology_id)} AS methodology_id,
        'net_meter_proxy' AS measurement_basis,
        'revenue_meter' AS voltage_measurement_location,
        {sql_string(mechanism.active_sign_review_state)}
            AS active_sign_review_state,
        {sql_string(mechanism.reactive_sign_review_state)}
            AS reactive_sign_review_state,
        true AS observability_only,
        false AS formal_inverter_conformance_assessable
    FROM aggregated
    ORDER BY serial, year_utc, month_utc, phase
    """.strip()


def _copy_query(
    config: FoundationConfig,
    query: str,
    output: Path,
    *,
    overwrite: bool,
) -> int:
    output = prepare_output_file(config, output, overwrite=overwrite)
    connection = connect(config)
    try:
        connection.execute(
            f"""COPY ({query}) TO {sql_string(output)}
            (FORMAT PARQUET, COMPRESSION {config.processing.parquet_compression})"""
        )
        return int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({sql_string(output)})"
            ).fetchone()[0]
        )
    finally:
        connection.close()


def build_voltvar_results(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build Volt-VAr proxy results; absent verified capacity stays unassessable."""

    mechanism.validate()
    output = voltvar_results_path(config, scope, mechanism)
    rows = _copy_query(
        config, _voltvar_sql(config, scope, mechanism), output, overwrite=overwrite
    )
    return _summary("voltvar", scope, rows, output, mechanism)


def build_voltwatt_results(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build Volt-Watt proxy results without claiming below-ceiling conformance."""

    mechanism.validate()
    output = voltwatt_results_path(config, scope, mechanism)
    rows = _copy_query(
        config, _voltwatt_sql(config, scope, mechanism), output, overwrite=overwrite
    )
    return _summary("voltwatt", scope, rows, output, mechanism)


def build_response_observability(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build independent phase-month response-observability diagnostics.

    Deliberately not namespaced by phase_scope_basis: this table already
    reports every actual telemetry phase (DER-inferred or not, tagged via
    is_inferred_der_phase in the status columns), so it is unaffected by the
    Volt-VAr/Volt-Watt phase-scope choice and stays a single shared table.
    """

    mechanism.validate()
    output = response_observability_path(config, scope)
    rows = _copy_query(
        config,
        _response_observability_sql(config, scope, mechanism),
        output,
        overwrite=overwrite,
    )
    return _summary("response_observability", scope, rows, output, mechanism)


def _summary(
    mechanism_name: str,
    scope: SourceScope,
    rows: int,
    output: Path,
    mechanism: MechanismAnalysisConfig,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mechanism": mechanism_name,
        "scope": scope.label,
        "rows": rows,
        "methodology_id": mechanism.methodology_id,
        "measurement_basis": "net_meter_proxy",
        "voltage_measurement_location": "revenue_meter",
        "capacity_basis": mechanism.capacity_basis,
        "formal_inverter_conformance_assessable": False,
        "output": str(output),
    }
