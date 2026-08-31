from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
import pytest

from ausgrid_analysis.analysis_cohort import site_eligibility_path
from ausgrid_analysis.capacity_proxy import build_capacity_proxy
from ausgrid_analysis.config import (
    AssumptionConfig,
    FoundationConfig,
    MetadataConfig,
    PathConfig,
    ProcessingConfig,
    QualityConfig,
    TelemetryConfig,
)
from ausgrid_analysis.db import structured_phase_output_path, structured_site_output_path
from ausgrid_analysis.mechanism_config import MechanismAnalysisConfig
from ausgrid_analysis.mechanism_paths import capacity_proxy_path, voltvar_results_path
from ausgrid_analysis.mechanism_results import build_voltvar_results


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


def _write_site_intervals(config: FoundationConfig, scope) -> None:
    """One site ('S1') with 100 half-hour rows across a day, high-voltage in
    the afternoon (activates Volt-VAr), varying net export so a percentile
    proxy has real spread. No s_rated_kva -- both proxies must still work.
    """

    rows = []
    for i in range(100):
        hour = i % 24
        # Afternoon hours get high voltage (activates the Volt-VAr curve);
        # export power ramps 0..990 W across the 100 rows.
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


def _write_phase_intervals(config: FoundationConfig, scope) -> None:
    """A minimal structured_phase_intervals partition -- build_voltvar_results
    requires this directory to exist (shared input check with
    build_response_observability), even though this module's own SQL never
    reads it.
    """

    rows = [
        dict(
            serial="S1",
            timestamp_utc=datetime(2025, 4, 1, 12, 0, tzinfo=timezone.utc),
            timestamp_local=datetime(2025, 4, 1, 12, 0),
            year_utc=2025,
            month_utc=4,
            phase="A",
            voltage_v=252.0,
            p_export_w=500.0,
            q_generator_var=-50.0,
            voltage_valid_for_analysis=True,
            power_measurement_available=True,
            is_inferred_der_phase=True,
        )
    ]
    frame = pd.DataFrame(rows)
    output = structured_phase_output_path(config, scope)
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.register("_frame", frame)
    connection.execute(
        f"COPY _frame TO '{output}' (FORMAT PARQUET, PARTITION_BY (year_utc, month_utc))"
    )
    connection.unregister("_frame")
    connection.close()


def _write_eligibility(config: FoundationConfig, *, solar_capacity_kw) -> None:
    eligibility = pd.DataFrame(
        [
            [
                "S1", "A", "solar_only", False, "no", "high",
                True, True, True, True, True, None, "unavailable", solar_capacity_kw,
            ],
        ],
        columns=[
            "serial", "inferred_der_phases", "analysis_cohort",
            "has_battery", "controlled_load_status", "phase_mapping_confidence",
            "gate_solar_only", "gate_no_battery", "gate_no_controlled_load",
            "gate_mapping", "gate_power_coverage", "s_rated_kva", "s_rated_source",
            "solar_capacity_kw",
        ],
    )
    path = site_eligibility_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.register("_eligibility", eligibility)
    connection.execute(f"COPY _eligibility TO '{path}' (FORMAT PARQUET)")
    connection.unregister("_eligibility")
    connection.close()


def _mechanism(**overrides) -> MechanismAnalysisConfig:
    defaults = dict(
        active_sign_review_state="empirically_supported_pending_provider_confirmation",
        reactive_sign_review_state="empirically_supported_pending_provider_confirmation",
    )
    defaults.update(overrides)
    return MechanismAnalysisConfig(**defaults).validate()


def test_build_capacity_proxy_rejects_non_empirical_basis(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    mechanism = _mechanism(capacity_basis="solar_capacity_kw_proxy")
    with pytest.raises(ValueError, match="empirical"):
        build_capacity_proxy(config, scope, mechanism)


def test_build_capacity_proxy_matches_numpy_percentile(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_site_intervals(config, scope)
    _write_eligibility(config, solar_capacity_kw=None)
    mechanism = _mechanism(capacity_basis="p99_net_export_proxy")

    result = build_capacity_proxy(config, scope, mechanism)
    assert result["rows"] == 1
    assert result["n_null_proxy"] == 0

    expected = float(np.percentile([i * 10 for i in range(100)], 99))
    connection = duckdb.connect()
    (actual,) = connection.execute(
        f"SELECT capacity_proxy_va FROM read_parquet('{capacity_proxy_path(config, scope, mechanism)}')"
    ).fetchone()
    connection.close()
    assert actual == pytest.approx(expected, rel=0.05)


def test_voltvar_results_missing_capacity_proxy_raises(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_site_intervals(config, scope)
    _write_phase_intervals(config, scope)
    _write_eligibility(config, solar_capacity_kw=None)
    mechanism = _mechanism(capacity_basis="p99_net_export_proxy")
    with pytest.raises(FileNotFoundError):
        build_voltvar_results(config, scope, mechanism)


def test_solar_capacity_kw_proxy_yields_real_assessable_rows(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_site_intervals(config, scope)
    _write_phase_intervals(config, scope)
    # 1 kW DC -> 1000 VA proxy: the fixture's export power tops out at 990 W,
    # so this keeps the 20%-of-capacity minimum-active-power floor (200 W)
    # achievable by the higher-power synthetic rows.
    _write_eligibility(config, solar_capacity_kw=1.0)
    mechanism = _mechanism(capacity_basis="solar_capacity_kw_proxy")

    build_voltvar_results(config, scope, mechanism)
    connection = duckdb.connect()
    frame = connection.execute(
        f"SELECT * FROM read_parquet('{voltvar_results_path(config, scope, mechanism)}')"
    ).fetchdf()
    connection.close()

    assert int(frame["n_assessable"].sum()) > 0
    assert frame["capacity_source"].eq("solar_capacity_kw_proxy").all()
    assert frame["capacity_reference_va"].eq(1000.0).all()
    assert not frame["formal_inverter_conformance_assessable"].any()


def test_p99_net_export_proxy_yields_real_assessable_rows(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_site_intervals(config, scope)
    _write_phase_intervals(config, scope)
    _write_eligibility(config, solar_capacity_kw=None)
    mechanism = _mechanism(capacity_basis="p99_net_export_proxy")

    build_capacity_proxy(config, scope, mechanism)
    build_voltvar_results(config, scope, mechanism)
    connection = duckdb.connect()
    frame = connection.execute(
        f"SELECT * FROM read_parquet('{voltvar_results_path(config, scope, mechanism)}')"
    ).fetchdf()
    connection.close()

    assert int(frame["n_assessable"].sum()) > 0
    assert frame["capacity_source"].eq("p99_net_export_proxy").all()
    assert (frame["capacity_reference_va"] > 0).all()


def test_default_s_rated_kva_basis_unaffected_by_new_proxies(tmp_path) -> None:
    """The original, unnamespaced basis must still behave exactly as before:
    null s_rated_kva stays capacity_unavailable, not silently filled in.
    """

    config = _config(tmp_path)
    scope = config.scope(None, None)
    _write_site_intervals(config, scope)
    _write_phase_intervals(config, scope)
    _write_eligibility(config, solar_capacity_kw=5.0)  # present, but must be ignored
    mechanism = _mechanism()  # capacity_basis defaults to 's_rated_kva'

    build_voltvar_results(config, scope, mechanism)
    connection = duckdb.connect()
    frame = connection.execute(
        f"SELECT * FROM read_parquet('{voltvar_results_path(config, scope, mechanism)}')"
    ).fetchdf()
    connection.close()

    assert int(frame["n_assessable"].sum()) == 0
    assert (frame["n_capacity_unavailable"] > 0).any()
    assert frame["capacity_source"].eq("unavailable").all()
