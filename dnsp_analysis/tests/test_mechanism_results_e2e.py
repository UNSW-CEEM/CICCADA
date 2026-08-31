from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pandas as pd

from dnsp_analysis.analysis_cohort import site_eligibility_path
from dnsp_analysis.config import (
    AssumptionConfig,
    FoundationConfig,
    MetadataConfig,
    PathConfig,
    ProcessingConfig,
    QualityConfig,
    TelemetryConfig,
)
from dnsp_analysis.db import (
    structured_phase_output_path,
    structured_site_output_path,
)
from dnsp_analysis.mechanism_config import MechanismAnalysisConfig
from dnsp_analysis.mechanism_paths import (
    response_observability_path,
    voltvar_results_path,
    voltwatt_results_path,
)
from dnsp_analysis.mechanism_results import (
    build_response_observability,
    build_voltvar_results,
    build_voltwatt_results,
)
from dnsp_analysis.mechanism_validation import validate_mechanism_results


def _config(tmp_path) -> FoundationConfig:
    return FoundationConfig(
        paths=PathConfig(
            tmp_path / "raw.parquet",
            tmp_path / "metadata.xlsx",
            tmp_path / "derived",
        ),
        metadata=MetadataConfig("Sheet1", "ID"),
        telemetry=TelemetryConfig(
            "serial", "time", "phase", "voltage", "current",
            "q", "p", "month", "file",
        ),
        assumptions=AssumptionConfig(
            -1, -1, "UTC", "Australia/Sydney",
            "instantaneous", "revenue_meter",
        ),
        quality=QualityConfig(0.0, 300.0, 1e-6),
        processing=ProcessingConfig(4, 1, "1GB", "zstd"),
    )


def _write_inputs(config: FoundationConfig) -> None:
    scope = config.scope(None, None)
    utc_a = datetime(2025, 4, 5, 15, 30, tzinfo=timezone.utc)
    utc_b = datetime(2025, 4, 5, 16, 30, tzinfo=timezone.utc)
    repeated_local = datetime(2025, 4, 6, 2, 30)
    site = pd.DataFrame(
        [
            ["rated", utc_a, repeated_local, 2025, 4, 254.0, 253.0, 253.5,
             254.0, 253.0, None, 4000.0, 1000.0, True,
             254.0, 253.0, 253.5, 4000.0, 1000.0, 1, 1],
            ["rated", utc_b, repeated_local, 2025, 4, 260.0, 258.0, 259.0,
             260.0, 258.0, 260.0, 2000.0, 3000.0, True,
             260.0, 258.0, 259.333333, 2000.0, 3000.0, 2, 2],
            ["unrated", utc_a, repeated_local, 2025, 4, 255.0, 254.0, 254.5,
             255.0, None, None, 3000.0, 1500.0, True,
             255.0, 254.0, 254.5, 3000.0, 1500.0, 1, 1],
        ],
        columns=[
            "serial", "timestamp_utc", "timestamp_local", "year_utc",
            "month_utc", "der_voltage_max_valid_v",
            "der_voltage_min_valid_v", "der_voltage_mean_valid_v",
            "voltage_a_v", "voltage_b_v", "voltage_c_v",
            "p_export_der_phase_net_complete_w",
            "q_absorbing_der_phase_net_complete_var",
            "der_phase_power_complete",
            # all_phases-basis equivalents (phase_scope_basis="all_phases"
            # is not exercised by this test, but the columns must exist --
            # production data always carries both column sets).
            "voltage_max_valid_v", "voltage_min_valid_v", "voltage_mean_valid_v",
            "p_export_net_observed_w", "q_absorbing_net_observed_var",
            "observed_phase_rows", "measured_power_phase_rows",
        ],
    )
    phase = pd.DataFrame(
        [
            ["rated", utc_a, repeated_local, 2025, 4, "A", 254.0,
             4000.0, -1000.0, True, True, True],
            ["rated", utc_b, repeated_local, 2025, 4, "A", 260.0,
             2000.0, -3000.0, True, True, True],
            ["unrated", utc_a, repeated_local, 2025, 4, "A", 255.0,
             3000.0, -1500.0, True, True, True],
        ],
        columns=[
            "serial", "timestamp_utc", "timestamp_local", "year_utc",
            "month_utc", "phase", "voltage_v", "p_export_w",
            "q_generator_var", "voltage_valid_for_analysis",
            "power_measurement_available", "is_inferred_der_phase",
        ],
    )
    eligibility = pd.DataFrame(
        [
            ["rated", "A", "solar_only", False, "no", "high",
             True, True, True, True, True, 5.0, "provider"],
            ["unrated", "A", "solar_only", False, "no", "high",
             True, True, True, True, True, None, "unavailable"],
        ],
        columns=[
            "serial", "inferred_der_phases", "analysis_cohort",
            "has_battery", "controlled_load_status",
            "phase_mapping_confidence", "gate_solar_only",
            "gate_no_battery", "gate_no_controlled_load", "gate_mapping",
            "gate_power_coverage", "s_rated_kva", "s_rated_source",
        ],
    )

    connection = duckdb.connect()
    for frame, output in (
        (site, structured_site_output_path(config, scope)),
        (phase, structured_phase_output_path(config, scope)),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        connection.register("_frame", frame)
        connection.execute(
            f"COPY _frame TO '{output}' "
            "(FORMAT PARQUET, PARTITION_BY (year_utc, month_utc))"
        )
        connection.unregister("_frame")
    eligibility_path = site_eligibility_path(config)
    eligibility_path.parent.mkdir(parents=True, exist_ok=True)
    connection.register("_eligibility", eligibility)
    connection.execute(
        f"COPY _eligibility TO '{eligibility_path}' (FORMAT PARQUET)"
    )
    connection.close()


def test_results_preserve_denominators_utc_and_unresolved_capacity(tmp_path) -> None:
    config = _config(tmp_path)
    _write_inputs(config)
    scope = config.scope(None, None)
    mechanism = MechanismAnalysisConfig(
        minimum_response_intervals=2,
        minimum_response_voltage_span_v=3.0,
        active_sign_review_state=(
            "empirically_supported_pending_provider_confirmation"
        ),
        reactive_sign_review_state=(
            "empirically_supported_pending_provider_confirmation"
        ),
    ).validate()

    build_voltvar_results(config, scope, mechanism)
    build_voltwatt_results(config, scope, mechanism)
    build_response_observability(config, scope, mechanism)
    validation = validate_mechanism_results(config, scope)

    assert validation["status"] == "pass"
    assert validation["structured_site_rows"] == 3
    assert validation["source_dst_local_collision_groups"] == 1
    assert validation["voltvar_denominator_difference"] == 0
    assert validation["voltwatt_denominator_difference"] == 0

    vv = duckdb.sql(
        f"SELECT * FROM read_parquet('{voltvar_results_path(config, scope)}')"
    ).fetchdf()
    vw = duckdb.sql(
        f"SELECT * FROM read_parquet('{voltwatt_results_path(config, scope)}')"
    ).fetchdf()
    obs = duckdb.sql(
        f"SELECT * FROM read_parquet('{response_observability_path(config, scope)}')"
    ).fetchdf()

    assert vv.loc[vv.serial.eq("unrated"), "n_capacity_unavailable"].sum() == 1
    assert vw.loc[vw.serial.eq("unrated"), "n_capacity_unavailable"].sum() == 1
    assert not vv.formal_inverter_conformance_assessable.any()
    assert not vw.formal_inverter_conformance_assessable.any()
    rated_obs = obs.loc[obs.serial.eq("rated")].iloc[0]
    assert rated_obs["voltvar_observability_status"] == "expected_direction_observed"
    assert rated_obs["voltwatt_observability_status"] == "drop_direction_observed"
