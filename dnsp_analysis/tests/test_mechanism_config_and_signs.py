from __future__ import annotations

import duckdb
import pytest

from ausgrid_analysis.mechanism_config import MechanismAnalysisConfig
from ausgrid_analysis.power_conventions import (
    q_generator_from_absorbing,
    q_generator_from_absorbing_sql,
)
from ausgrid_analysis.schemas import normalize_reactive_power


def test_named_q_conversion_matches_canonical_sign_contract() -> None:
    q_absorbing, q_generator = normalize_reactive_power(100.0, -1)
    assert q_absorbing == -100.0
    assert q_generator == 100.0
    assert q_generator_from_absorbing(q_absorbing) == q_generator
    sql_value = duckdb.sql(
        "SELECT " + q_generator_from_absorbing_sql("-100.0")
    ).fetchone()[0]
    assert sql_value == q_generator
    assert q_generator_from_absorbing(None) is None


def test_voltage_and_capacity_choices_are_restricted() -> None:
    assert MechanismAnalysisConfig(voltage_aggregate="max").validate()
    assert MechanismAnalysisConfig(voltage_aggregate="avg").validate()
    with pytest.raises(ValueError, match="voltage_aggregate"):
        MechanismAnalysisConfig(voltage_aggregate="median").validate()
    with pytest.raises(ValueError, match="only verified"):
        MechanismAnalysisConfig(
            capacity_basis="solar_capacity_kw"
        ).validate()


def test_phase_scope_basis_is_restricted_and_defaults_der_inferred() -> None:
    default = MechanismAnalysisConfig().validate()
    assert default.phase_scope_basis == "der_inferred"
    assert default.comparison_p_column == "p_export_der_phase_net_complete_w"
    assert default.comparison_q_absorbing_column == (
        "q_absorbing_der_phase_net_complete_var"
    )
    assert default.power_scope_complete_sql == "s.der_phase_power_complete"
    assert default.phase_scope_sql == "coalesce(e.inferred_der_phases, 'unmapped')"
    # methodology_id must be unchanged from before phase_scope_basis existed,
    # so every already-built der_inferred output stays reusable/comparable.
    assert "__phase_" not in default.methodology_id

    all_phases = MechanismAnalysisConfig(phase_scope_basis="all_phases").validate()
    assert all_phases.comparison_p_column == "p_export_net_observed_w"
    assert all_phases.comparison_q_absorbing_column == "q_absorbing_net_observed_var"
    assert all_phases.power_scope_complete_sql == (
        "(s.measured_power_phase_rows = s.observed_phase_rows "
        "AND s.observed_phase_rows > 0)"
    )
    assert all_phases.phase_scope_sql == "'all_phases'"
    assert "__phase_all_phases__" in all_phases.methodology_id
    assert all_phases.methodology_id != default.methodology_id

    with pytest.raises(ValueError, match="phase_scope_basis"):
        MechanismAnalysisConfig(phase_scope_basis="single_phase_only").validate()


def test_phase_scope_basis_namespaces_only_the_non_default_path(tmp_path) -> None:
    from ausgrid_analysis.config import (
        AssumptionConfig, FoundationConfig, MetadataConfig, PathConfig,
        ProcessingConfig, QualityConfig, TelemetryConfig,
    )
    from ausgrid_analysis.mechanism_paths import voltvar_results_path

    config = FoundationConfig(
        paths=PathConfig(tmp_path / "raw.parquet", tmp_path / "meta.xlsx", tmp_path / "derived"),
        metadata=MetadataConfig("Sheet1", "ID"),
        telemetry=TelemetryConfig("serial", "time", "phase", "voltage", "current", "q", "p", "month", "file"),
        assumptions=AssumptionConfig(-1, -1, "UTC", "Australia/Sydney", "instantaneous", "revenue_meter"),
        quality=QualityConfig(0.0, 300.0, 1e-6),
        processing=ProcessingConfig(4, 1, "1GB", "zstd"),
    )
    scope = config.scope(None, None)
    default_path = voltvar_results_path(config, scope)
    explicit_default_path = voltvar_results_path(
        config, scope, MechanismAnalysisConfig()
    )
    all_phases_path = voltvar_results_path(
        config, scope, MechanismAnalysisConfig(phase_scope_basis="all_phases")
    )
    assert default_path == explicit_default_path
    assert all_phases_path != default_path
    assert "phase_scope_all_phases" in str(all_phases_path)
    assert "phase_scope_all_phases" not in str(default_path)


def test_capacity_basis_proxies_are_restricted_and_default_s_rated_kva() -> None:
    default = MechanismAnalysisConfig().validate()
    assert default.capacity_basis == "s_rated_kva"
    assert not default.capacity_is_empirical
    assert default.capacity_metadata_column == "s_rated_kva"
    assert "__capacity_" in default.methodology_id  # unchanged, was always there

    solar = MechanismAnalysisConfig(capacity_basis="solar_capacity_kw_proxy").validate()
    assert not solar.capacity_is_empirical
    assert solar.capacity_metadata_column == "solar_capacity_kw"
    assert "solar_capacity_kw_proxy" in solar.methodology_id
    assert solar.methodology_id != default.methodology_id

    p99 = MechanismAnalysisConfig(capacity_basis="p99_net_export_proxy").validate()
    assert p99.capacity_is_empirical
    assert p99.capacity_metadata_column is None
    assert "p99_net_export_proxy_p99" in p99.methodology_id

    p95 = MechanismAnalysisConfig(
        capacity_basis="p99_net_export_proxy", capacity_proxy_percentile=0.95
    ).validate()
    assert p95.methodology_id != p99.methodology_id  # different percentile, different id

    with pytest.raises(ValueError, match="capacity_basis"):
        MechanismAnalysisConfig(capacity_basis="approved_capacity_kw").validate()
    with pytest.raises(ValueError, match="capacity_proxy_percentile"):
        MechanismAnalysisConfig(capacity_proxy_percentile=0.0).validate()
    with pytest.raises(ValueError, match="capacity_proxy_percentile"):
        MechanismAnalysisConfig(capacity_proxy_percentile=1.0).validate()


def test_capacity_basis_namespaces_independently_of_phase_scope_basis(tmp_path) -> None:
    from ausgrid_analysis.config import (
        AssumptionConfig, FoundationConfig, MetadataConfig, PathConfig,
        ProcessingConfig, QualityConfig, TelemetryConfig,
    )
    from ausgrid_analysis.mechanism_paths import capacity_proxy_path, voltvar_results_path

    config = FoundationConfig(
        paths=PathConfig(tmp_path / "raw.parquet", tmp_path / "meta.xlsx", tmp_path / "derived"),
        metadata=MetadataConfig("Sheet1", "ID"),
        telemetry=TelemetryConfig("serial", "time", "phase", "voltage", "current", "q", "p", "month", "file"),
        assumptions=AssumptionConfig(-1, -1, "UTC", "Australia/Sydney", "instantaneous", "revenue_meter"),
        quality=QualityConfig(0.0, 300.0, 1e-6),
        processing=ProcessingConfig(4, 1, "1GB", "zstd"),
    )
    scope = config.scope(None, None)
    default_path = voltvar_results_path(config, scope)
    solar_path = voltvar_results_path(
        config, scope, MechanismAnalysisConfig(capacity_basis="solar_capacity_kw_proxy")
    )
    p99_der_path = voltvar_results_path(
        config, scope, MechanismAnalysisConfig(capacity_basis="p99_net_export_proxy")
    )
    p99_all_phases_path = voltvar_results_path(
        config,
        scope,
        MechanismAnalysisConfig(
            capacity_basis="p99_net_export_proxy", phase_scope_basis="all_phases"
        ),
    )
    all_paths = {default_path, solar_path, p99_der_path, p99_all_phases_path}
    assert len(all_paths) == 4, "every combination must resolve to a distinct path"
    assert "capacity_solar_capacity_kw_proxy" in str(solar_path)
    assert "capacity_p99_net_export_proxy_p99" in str(p99_der_path)
    assert "phase_scope_all_phases" in str(p99_all_phases_path)
    assert "capacity_p99_net_export_proxy_p99" in str(p99_all_phases_path)

    proxy_path = capacity_proxy_path(
        config, scope, MechanismAnalysisConfig(capacity_basis="p99_net_export_proxy")
    )
    assert proxy_path.parent.name == "analysis_cohort"
    with pytest.raises(ValueError, match="empirical"):
        capacity_proxy_path(
            config, scope, MechanismAnalysisConfig(capacity_basis="solar_capacity_kw_proxy")
        )


def test_signs_must_be_independently_ready() -> None:
    config = MechanismAnalysisConfig(
        active_sign_review_state=(
            "empirically_supported_pending_provider_confirmation"
        ),
        reactive_sign_review_state="contradicted",
    ).validate()
    assert config.active_sign_ready
    assert not config.reactive_sign_ready
