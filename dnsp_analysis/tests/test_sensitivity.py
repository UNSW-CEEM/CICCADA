from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pandas as pd
import pytest

from dnsp_analysis.analysis_cohort import site_eligibility_path
from dnsp_analysis.as4777_curves import Q_IMPACT_THRESHOLDS
from dnsp_analysis.config import (
    AssumptionConfig,
    FoundationConfig,
    MetadataConfig,
    PathConfig,
    ProcessingConfig,
    QualityConfig,
    StructuredTelemetryConfig,
    TelemetryConfig,
)
from dnsp_analysis.db import (
    prepare_output_file,
    site_phase_profile_path,
    structured_phase_output_path,
    structured_site_output_path,
)
from dnsp_analysis.mechanism_config import MechanismAnalysisConfig
from dnsp_analysis.mechanism_paths import voltvar_results_path
from dnsp_analysis.mechanism_results import (
    build_voltvar_results,
    voltvar_q_impact_histogram,
)
from dnsp_analysis.sensitivity import (
    capacity_percentile_sensitivity,
    phase_mapping_sensitivity,
    q_impact_bucket_sensitivity,
    tolerance_fraction_sensitivity,
)


def _config(tmp_path) -> FoundationConfig:
    return FoundationConfig(
        paths=PathConfig(
            tmp_path / "raw.parquet", tmp_path / "metadata.xlsx", tmp_path / "derived"
        ),
        metadata=MetadataConfig("Sheet1", "ID"),
        telemetry=TelemetryConfig(
            "serial", "time", "phase", "voltage", "current", "q", "p", "month", "file"
        ),
        assumptions=AssumptionConfig(
            -1, -1, "UTC", "Australia/Sydney", "instantaneous", "revenue_meter"
        ),
        quality=QualityConfig(0.0, 300.0, 1e-6),
        processing=ProcessingConfig(4, 1, "1GB", "zstd"),
    )


def _mechanism(**overrides) -> MechanismAnalysisConfig:
    defaults = dict(
        minimum_response_intervals=2,
        minimum_response_voltage_span_v=3.0,
        active_sign_review_state="empirically_supported_pending_provider_confirmation",
        reactive_sign_review_state="empirically_supported_pending_provider_confirmation",
    )
    defaults.update(overrides)
    return MechanismAnalysisConfig(**defaults).validate()


def _write_voltvar_inputs(config: FoundationConfig) -> None:
    """One 'rated' site with two assessable Volt-VAr intervals (one lands
    inside the tolerance-clamped band -> conformant, one outside -> some
    non-conformant bucket) and one 'unrated' site with no s_rated_kva
    (stays capacity_unavailable) -- same fixture shape as
    test_mechanism_results_e2e.py, reused here since it already exercises
    real Q_impact spread across buckets.
    """

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
    connection.execute(f"COPY _eligibility TO '{eligibility_path}' (FORMAT PARQUET)")
    connection.close()


def _production_totals(config, scope, mechanism) -> dict[str, int]:
    frame = duckdb.sql(
        f"SELECT * FROM read_parquet('{voltvar_results_path(config, scope, mechanism)}')"
    ).fetchdf()
    return {
        column: int(frame[column].sum())
        for column in (
            "n_assessable", "n_conformant", "n_adverse", "n_inactive",
            "n_major_deficit", "n_minor_deviation", "n_major_surplus",
        )
    }


# --- q_impact_bucket_sensitivity / tolerance_fraction_sensitivity / histogram ---


def test_q_impact_bucket_sensitivity_default_thresholds_reproduce_production(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_voltvar_inputs(config)
    mechanism = _mechanism()
    build_voltvar_results(config, scope, mechanism)
    expected = _production_totals(config, scope, mechanism)

    result = q_impact_bucket_sensitivity(config, scope, mechanism)
    row = result.loc[result["threshold_set"] == "band_x1"].iloc[0]

    assert (row["adverse_cutoff"], row["inactive_cutoff"],
            row["major_deficit_cutoff"], row["minor_deviation_cutoff"]) == Q_IMPACT_THRESHOLDS
    for column, value in expected.items():
        assert int(row[column]) == value, column


def test_q_impact_bucket_sensitivity_bucket_sum_matches_assessable(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_voltvar_inputs(config)
    mechanism = _mechanism()
    build_voltvar_results(config, scope, mechanism)

    result = q_impact_bucket_sensitivity(config, scope, mechanism)
    bucket_columns = [
        "n_conformant", "n_adverse", "n_inactive",
        "n_major_deficit", "n_minor_deviation", "n_major_surplus",
    ]
    for _, row in result.iterrows():
        assert row[bucket_columns].sum() == row["n_assessable"]


def test_q_impact_bucket_sensitivity_wider_bands_move_mass_toward_conformant_side(
    tmp_path,
) -> None:
    """A wider inactive/minor_deviation band can only ever reclassify rows
    from adverse/major_deficit into inactive/minor_deviation (never the
    reverse) -- the non-conformance count should be monotonically
    non-increasing as the band widens, holding the fixed 'conformant' count
    aside.
    """

    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_voltvar_inputs(config)
    mechanism = _mechanism()
    build_voltvar_results(config, scope, mechanism)

    result = q_impact_bucket_sensitivity(config, scope, mechanism, bin_width=0.005)
    ordered = result.set_index("threshold_set").loc[
        ["band_x0.5", "band_x0.75", "band_x1", "band_x1.25", "band_x1.5"]
    ]
    non_conformance = (
        ordered["n_adverse"] + ordered["n_inactive"] + ordered["n_major_deficit"]
    )
    assert non_conformance.is_monotonic_decreasing


def test_tolerance_fraction_sensitivity_at_production_value_matches_build(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_voltvar_inputs(config)
    mechanism = _mechanism()  # tolerance_fraction defaults to 0.04
    build_voltvar_results(config, scope, mechanism)
    expected = _production_totals(config, scope, mechanism)

    result = tolerance_fraction_sensitivity(
        config, scope, mechanism, tolerance_fractions=(0.04,)
    )
    row = result.iloc[0]
    for column, value in expected.items():
        assert int(row[column]) == value, column
    assert row["conformance_fraction"] + row["non_conformance_fraction"] == pytest.approx(1.0)


def test_tolerance_fraction_sensitivity_wider_tolerance_never_shrinks_conformance(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_voltvar_inputs(config)
    mechanism = _mechanism()
    build_voltvar_results(config, scope, mechanism)

    result = tolerance_fraction_sensitivity(
        config, scope, mechanism, tolerance_fractions=(0.01, 0.04, 0.10, 0.20)
    ).sort_values("tolerance_fraction")
    assert result["n_conformant"].is_monotonic_increasing


def test_voltvar_q_impact_histogram_sums_to_n_assessable(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_voltvar_inputs(config)
    mechanism = _mechanism()
    build_voltvar_results(config, scope, mechanism)
    expected = _production_totals(config, scope, mechanism)

    histogram = voltvar_q_impact_histogram(config, scope, mechanism)
    assert int(histogram["n"].sum()) == expected["n_assessable"]


# --- capacity_percentile_sensitivity ---


def _write_capacity_inputs(config: FoundationConfig, scope) -> None:
    rows = []
    for i in range(100):
        hour = i % 24
        voltage = 252.0 if 11 <= hour <= 15 else 235.0
        export_w = float(i * 10)
        rows.append(
            dict(
                serial="S1",
                timestamp_utc=datetime(2025, 4, 1, hour, 0, tzinfo=timezone.utc),
                timestamp_local=datetime(2025, 4, 1, hour, 0),
                year_utc=2025,
                month_utc=4,
                der_voltage_max_valid_v=voltage,
                der_voltage_min_valid_v=voltage,
                der_voltage_mean_valid_v=voltage,
                voltage_a_v=voltage,
                voltage_b_v=voltage,
                voltage_c_v=voltage,
                p_export_der_phase_net_complete_w=export_w,
                q_absorbing_der_phase_net_complete_var=-50.0,
                der_phase_power_complete=True,
                voltage_max_valid_v=voltage,
                voltage_min_valid_v=voltage,
                voltage_mean_valid_v=voltage,
                p_export_net_observed_w=export_w,
                q_absorbing_net_observed_var=-50.0,
                observed_phase_rows=1,
                measured_power_phase_rows=1,
            )
        )
    frame = pd.DataFrame(rows)
    output = structured_site_output_path(config, scope)
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.register("_frame", frame)
    connection.execute(
        f"COPY _frame TO '{output}' (FORMAT PARQUET, PARTITION_BY (year_utc, month_utc))"
    )
    connection.unregister("_frame")
    connection.close()

    eligibility = pd.DataFrame(
        [
            [
                "S1", "A", "solar_only", False, "no", "high",
                True, True, True, True, True, None, "unavailable", None,
            ],
        ],
        columns=[
            "serial", "inferred_der_phases", "analysis_cohort", "has_battery",
            "controlled_load_status", "phase_mapping_confidence",
            "gate_solar_only", "gate_no_battery", "gate_no_controlled_load",
            "gate_mapping", "gate_power_coverage", "s_rated_kva",
            "s_rated_source", "solar_capacity_kw",
        ],
    )
    eligibility_path = site_eligibility_path(config)
    eligibility_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.register("_eligibility", eligibility)
    connection.execute(f"COPY _eligibility TO '{eligibility_path}' (FORMAT PARQUET)")
    connection.unregister("_eligibility")
    connection.close()


def test_capacity_percentile_sensitivity_rejects_non_empirical_basis(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    mechanism = _mechanism(capacity_basis="solar_capacity_kw_proxy")
    with pytest.raises(ValueError, match="empirical"):
        capacity_percentile_sensitivity(config, scope, mechanism, percentiles=(0.5,))


def test_capacity_percentile_sensitivity_is_monotonic_in_percentile(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_capacity_inputs(config, scope)
    mechanism = _mechanism(capacity_basis="p99_net_export_proxy")

    result = capacity_percentile_sensitivity(
        config, scope, mechanism, percentiles=(0.5, 0.9, 0.99)
    ).sort_values("capacity_proxy_percentile")

    assert result["max_capacity_proxy_va"].is_monotonic_increasing
    assert result["output"].is_unique  # each percentile gets its own namespaced path
    assert (result["n_null_proxy"] == 0).all()


# --- phase_mapping_sensitivity ---


def _structured_telemetry_config() -> StructuredTelemetryConfig:
    return StructuredTelemetryConfig()


def _write_phase_profile(config: FoundationConfig, scope) -> None:
    frame = pd.DataFrame(
        [
            # Site 1: unambiguous single-phase mapping (matches
            # test_telemetry_profiles.py's baseline "high confidence" case).
            {"serial": "clean", "phase": "A", "power_measurement_available": True,
             "solar_signature_w": 1000.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 1},
            {"serial": "clean", "phase": "B", "power_measurement_available": True,
             "solar_signature_w": 50.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 1},
            {"serial": "clean", "phase": "C", "power_measurement_available": True,
             "solar_signature_w": 20.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 1},
            # Site 2: floor signature (120 W) sits above the production
            # min_signature_w (100) but below a +25% variant (125) -- this
            # site's confidence should flip from 'medium' to 'low' once that
            # one threshold is raised, and nowhere else.
            {"serial": "borderline", "phase": "A", "power_measurement_available": True,
             "solar_signature_w": 120.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 1},
            {"serial": "borderline", "phase": "B", "power_measurement_available": True,
             "solar_signature_w": 96.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 1},
            {"serial": "borderline", "phase": "C", "power_measurement_available": True,
             "solar_signature_w": 10.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 1},
        ]
    )
    path = site_phase_profile_path(config, scope)
    output = prepare_output_file(config, path, overwrite=False)
    connection = duckdb.connect()
    connection.register("_frame", frame)
    connection.execute(f"COPY _frame TO '{output}' (FORMAT PARQUET)")
    connection.unregister("_frame")
    connection.close()


def test_phase_mapping_sensitivity_requires_built_profile(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    with pytest.raises(FileNotFoundError):
        phase_mapping_sensitivity(config, scope)


def test_phase_mapping_sensitivity_production_baseline(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_phase_profile(config, scope)

    result = phase_mapping_sensitivity(config, scope)
    production = result.loc[result["variant"] == "production"].iloc[0]
    assert production["n_sites"] == 2
    assert production["n_high"] == 1  # 'clean'
    assert production["n_medium"] == 1  # 'borderline'
    assert production["n_low"] == 0


def test_phase_mapping_sensitivity_raising_min_signature_flips_borderline_site(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_phase_profile(config, scope)

    result = phase_mapping_sensitivity(config, scope)
    plus25 = result.loc[
        result["variant"] == "phase_mapping_min_signature_w__minus25pct"
    ].iloc[0]
    # -25% (75 W) is still below the borderline site's 120 W floor: unchanged.
    assert plus25["n_medium"] == 1
    assert plus25["n_low"] == 0

    flipped = result.loc[
        result["variant"] == "phase_mapping_min_signature_w__plus25pct"
    ].iloc[0]
    # +25% (125 W) exceeds the borderline site's 120 W floor: flips to low.
    assert flipped["n_medium"] == 0
    assert flipped["n_low"] == 1
    assert flipped["n_high"] == 1  # 'clean' site (1000 W floor) unaffected


def test_phase_mapping_sensitivity_accepts_explicit_variants(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_phase_profile(config, scope)

    custom = {"tight": StructuredTelemetryConfig(phase_mapping_min_signature_w=5000.0)}
    result = phase_mapping_sensitivity(config, scope, variants=custom)
    assert list(result["variant"]) == ["tight"]
    row = result.iloc[0]
    assert row["n_high"] == 0
    assert row["n_low"] == 2  # both sites' floors now fall below 5000 W
