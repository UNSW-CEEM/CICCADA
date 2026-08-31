from __future__ import annotations

import inspect

import duckdb
import pandas as pd
import pytest

from ausgrid_analysis import result_plots as rp
from ausgrid_analysis import result_views as rv
from ausgrid_analysis.config import (
    AssumptionConfig,
    FoundationConfig,
    MetadataConfig,
    PathConfig,
    ProcessingConfig,
    QualityConfig,
    TelemetryConfig,
)
from ausgrid_analysis.mechanism_config import MechanismAnalysisConfig
from ausgrid_analysis.mechanism_paths import (
    response_observability_path,
    voltvar_results_path,
    voltwatt_results_path,
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


def _write_parquet(rows: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    connection = duckdb.connect()
    connection.register("_frame", frame)
    connection.execute(f"COPY _frame TO '{path}' (FORMAT PARQUET)")
    connection.unregister("_frame")
    connection.close()


_DEFAULT_METHODOLOGY_ID = "net_meter_proxy__voltage_avg__capacity_s_rated_kva__tol_0.04"


def _voltvar_row(**overrides) -> dict:
    row = dict(
        serial="S1",
        year_utc=2025,
        month_utc=4,
        phase_scope="A",
        voltage_bin_lower_v=250.0,
        analysis_cohort="solar_only",
        phase_mapping_confidence="high",
        minimum_comparison_voltage_v=250.0,
        maximum_comparison_voltage_v=251.0,
        n_source_intervals=10,
        n_ineligible_site=0,
        n_missing_input=0,
        n_not_activated=0,
        n_sign_unverified=0,
        n_capacity_unavailable=0,
        n_below_minimum_active_power=0,
        n_assessable=10,
        n_conformant=5,
        n_adverse=1,
        n_inactive=1,
        n_major_deficit=1,
        n_minor_deviation=1,
        n_major_surplus=1,
        n_responded=9,
        mean_q_impact=0.1,
        methodology_id=_DEFAULT_METHODOLOGY_ID,
        measurement_basis="net_meter_proxy",
        voltage_measurement_location="revenue_meter",
        voltage_basis="mean_inferred_der_phase_revenue_meter_voltage",
        capacity_basis="s_rated_kva",
        active_sign_review_state="unverified",
        reactive_sign_review_state="unverified",
        formal_inverter_conformance_assessable=False,
    )
    row.update(overrides)
    return row


def _voltwatt_row(**overrides) -> dict:
    row = dict(
        serial="S1",
        year_utc=2025,
        month_utc=4,
        phase_scope="A",
        voltage_bin_lower_v=254.0,
        analysis_cohort="solar_only",
        phase_mapping_confidence="high",
        minimum_comparison_voltage_v=254.0,
        maximum_comparison_voltage_v=255.0,
        n_source_intervals=8,
        n_ineligible_site=0,
        n_missing_input=0,
        n_not_activated=0,
        n_sign_unverified=0,
        n_not_exporting=0,
        n_capacity_unavailable=0,
        n_assessable=8,
        n_proxy_exceeds_curve_ceiling=3,
        n_proxy_does_not_exceed_curve_ceiling=5,
        methodology_id=_DEFAULT_METHODOLOGY_ID,
        measurement_basis="net_meter_proxy",
        voltage_measurement_location="revenue_meter",
        voltage_basis="mean_inferred_der_phase_revenue_meter_voltage",
        capacity_basis="s_rated_kva",
        active_sign_review_state="unverified",
        formal_inverter_conformance_assessable=False,
        interpretation_guardrail="Below-ceiling net export is not proof of inverter conformance",
    )
    row.update(overrides)
    return row


def _response_row(**overrides) -> dict:
    row = dict(
        serial="S1",
        year_utc=2025,
        month_utc=4,
        phase="A",
        analysis_cohort="solar_only",
        phase_mapping_confidence="high",
        site_eligible=True,
        inferred_der_phase=True,
        n_source_intervals=20,
        n_valid_power_voltage=20,
        n_voltvar_excited_intervals=12,
        voltvar_minimum_voltage_v=241.0,
        voltvar_maximum_voltage_v=250.0,
        q_generator_slope_var_per_v=-5.0,
        q_generator_voltage_correlation=-0.4,
        voltvar_voltage_span_v=9.0,
        n_voltwatt_excited_export_intervals=6,
        voltwatt_minimum_voltage_v=254.0,
        voltwatt_maximum_voltage_v=257.0,
        p_export_slope_w_per_v=-3.0,
        p_export_voltage_correlation=-0.2,
        voltwatt_voltage_span_v=3.0,
        voltvar_observability_status="expected_direction_observed",
        voltwatt_observability_status="drop_direction_observed",
        methodology_id=_DEFAULT_METHODOLOGY_ID,
        measurement_basis="net_meter_proxy",
        voltage_measurement_location="revenue_meter",
        active_sign_review_state="unverified",
        reactive_sign_review_state="unverified",
        observability_only=True,
        formal_inverter_conformance_assessable=False,
    )
    row.update(overrides)
    return row


@pytest.fixture()
def views(tmp_path):
    """A small, internally-consistent der_inferred fixture set."""

    config = _config(tmp_path)
    scope = config.scope(None, None)
    mechanism = MechanismAnalysisConfig().validate()

    voltvar_rows = [
        _voltvar_row(serial="S1", voltage_bin_lower_v=250.0),
        _voltvar_row(
            serial="S2",
            voltage_bin_lower_v=251.0,
            analysis_cohort="solar_battery",
            n_source_intervals=5,
            n_ineligible_site=5,
            n_assessable=0,
            n_conformant=0,
            n_adverse=0,
            n_inactive=0,
            n_major_deficit=0,
            n_minor_deviation=0,
            n_major_surplus=0,
            n_responded=0,
            mean_q_impact=None,
        ),
    ]
    voltwatt_rows = [
        _voltwatt_row(serial="S1", voltage_bin_lower_v=254.0),
        _voltwatt_row(
            serial="S2",
            voltage_bin_lower_v=255.0,
            analysis_cohort="solar_battery",
            n_source_intervals=4,
            n_not_exporting=4,
            n_assessable=0,
            n_proxy_exceeds_curve_ceiling=0,
            n_proxy_does_not_exceed_curve_ceiling=0,
        ),
    ]
    response_rows = [
        _response_row(serial="S1", phase="A"),
        _response_row(
            serial="S1",
            phase="B",
            inferred_der_phase=False,
            n_voltvar_excited_intervals=0,
            n_voltwatt_excited_export_intervals=0,
            q_generator_slope_var_per_v=None,
            q_generator_voltage_correlation=None,
            p_export_slope_w_per_v=None,
            p_export_voltage_correlation=None,
            voltvar_voltage_span_v=None,
            voltwatt_voltage_span_v=None,
            voltvar_observability_status="not_inferred_der_phase",
            voltwatt_observability_status="not_inferred_der_phase",
        ),
        # Same (year_utc, month_utc) as the rows above but a different site --
        # exercises UTC-month grouping merging multiple rows correctly.
        _response_row(serial="S3", phase="A", year_utc=2025, month_utc=4),
    ]

    _write_parquet(voltvar_rows, voltvar_results_path(config, scope, mechanism))
    _write_parquet(voltwatt_rows, voltwatt_results_path(config, scope, mechanism))
    _write_parquet(response_rows, response_observability_path(config, scope))

    return {
        "config": config,
        "scope": scope,
        "mechanism": mechanism,
        "voltvar_rows": voltvar_rows,
        "voltwatt_rows": voltwatt_rows,
        "response_rows": response_rows,
    }


# ---------------------------------------------------------------------------
# Denominator/classification reconciliation
# ---------------------------------------------------------------------------


def test_fleet_and_grouped_denominator_sums_reconcile_exactly(views) -> None:
    config, scope, mechanism = views["config"], views["scope"], views["mechanism"]

    fleet = rv.voltvar_denominator_view(config, scope, mechanism=mechanism)
    expected_total = sum(row["n_source_intervals"] for row in views["voltvar_rows"])
    assert int(fleet["n_source_intervals"].iloc[0]) == expected_total

    by_serial = rv.voltvar_denominator_view(
        config, scope, mechanism=mechanism, dimensions=("serial",)
    )
    assert by_serial.set_index("serial")["n_source_intervals"].to_dict() == {
        "S1": 10,
        "S2": 5,
    }
    # Grouped sums must reconcile back to the fleet total.
    assert int(by_serial["n_source_intervals"].sum()) == expected_total


def test_classification_counts_reconcile_to_n_assessable(views) -> None:
    config, scope, mechanism = views["config"], views["scope"], views["mechanism"]
    status = rv.voltvar_status_view(config, scope, mechanism=mechanism)
    classified = sum(int(status[c].iloc[0]) for c in rv.VOLTVAR_STATUS_COLUMNS)
    assert classified == int(status["n_assessable"].iloc[0])


def test_site_conformance_fractions_are_complementary(views) -> None:
    """S1: n_conformant=5, n_minor_deviation=1, n_major_surplus=1 -> n_conformance=7;
    n_adverse=1, n_inactive=1, n_major_deficit=1 -> n_non_conformance=3; both out
    of n_assessable=10. S2 has n_assessable=0 -> not_assessable, both fractions null.
    """

    config, scope, mechanism = views["config"], views["scope"], views["mechanism"]
    frame = rv.voltvar_site_conformance_view(config, scope, mechanism=mechanism).set_index("serial")

    s1 = frame.loc["S1"]
    assert s1["n_conformance"] == 7
    assert s1["n_non_conformance"] == 3
    assert s1["conformance_fraction"] == pytest.approx(0.7)
    assert s1["non_conformance_fraction"] == pytest.approx(0.3)
    assert s1["conformance_fraction"] + s1["non_conformance_fraction"] == pytest.approx(1.0)
    assert s1["site_status"] == "conformant"  # 0.7 >= default 0.5 threshold

    s2 = frame.loc["S2"]
    assert s2["site_status"] == "not_assessable"
    assert pd.isna(s2["conformance_fraction"])
    assert pd.isna(s2["non_conformance_fraction"])


def test_site_conformance_threshold_is_configurable(views) -> None:
    config, scope, mechanism = views["config"], views["scope"], views["mechanism"]
    strict = rv.voltvar_site_conformance_view(
        config, scope, mechanism=mechanism, conformance_threshold=0.9
    ).set_index("serial")
    # S1's conformance_fraction is 0.7 -- conformant at the 0.5 default, but
    # non_conformant once the bar is raised to the colleague's implied 90%.
    assert strict.loc["S1", "site_status"] == "non_conformant"


def test_validate_result_views_catches_a_broken_classification(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    mechanism = MechanismAnalysisConfig().validate()
    broken = _voltvar_row(n_assessable=10, n_adverse=3)  # 5+3+1+1+1+1=12 != 10
    _write_parquet([broken], voltvar_results_path(config, scope, mechanism))
    _write_parquet([_voltwatt_row()], voltwatt_results_path(config, scope, mechanism))
    _write_parquet([_response_row()], response_observability_path(config, scope))

    result = rv.validate_result_views(config, scope, mechanism=mechanism)
    assert result["status"] == "fail"
    assert any("classification" in failure for failure in result["failures"])


# ---------------------------------------------------------------------------
# Null-on-zero and low-denominator handling
# ---------------------------------------------------------------------------


def test_zero_denominators_yield_null_fractions_not_zero(views) -> None:
    config, scope, mechanism = views["config"], views["scope"], views["mechanism"]
    status = rv.voltvar_status_view(
        config, scope, mechanism=mechanism, dimensions=("serial",)
    )
    s2 = status.set_index("serial").loc["S2"]
    assert int(s2["n_assessable"]) == 0
    fraction_cols = [c for c in status.columns if c.endswith("_fraction_of_assessable")]
    for col in fraction_cols:
        assert pd.isna(s2[col]), f"{col} should be null (not 0) when n_assessable == 0"


def test_low_denominator_flags_are_retained(views) -> None:
    config, scope, mechanism = views["config"], views["scope"], views["mechanism"]
    status = rv.voltvar_status_view(
        config, scope, mechanism=mechanism, dimensions=("serial",), minimum_denominator=5
    )
    flags = status.set_index("serial")["low_denominator_warning"].to_dict()
    assert flags["S1"] is False  # n_assessable=10 >= 5
    assert flags["S2"] is True  # n_assessable=0 < 5


# ---------------------------------------------------------------------------
# Terminology preservation
# ---------------------------------------------------------------------------


def test_voltwatt_below_ceiling_label_is_not_renamed_conformance(views) -> None:
    config, scope, mechanism = views["config"], views["scope"], views["mechanism"]
    status = rv.voltwatt_status_view(config, scope, mechanism=mechanism)
    assert "n_proxy_does_not_exceed_curve_ceiling" in status.columns
    assert "proxy_does_not_exceed_curve_ceiling_fraction_of_assessable" in status.columns
    assert not any("conform" in c.lower() for c in status.columns)
    label = rp.STATUS_LABELS["n_proxy_does_not_exceed_curve_ceiling"]
    assert "conform" not in label.lower()
    assert label == "proxy_does_not_exceed_curve_ceiling"


# ---------------------------------------------------------------------------
# phase_scope (curve) vs phase (observability) cannot be confused
# ---------------------------------------------------------------------------


def test_curve_phase_scope_and_observability_phase_cannot_be_confused(views) -> None:
    assert "phase_scope" in rv.CURVE_ALLOWED_DIMENSIONS
    assert "phase" not in rv.CURVE_ALLOWED_DIMENSIONS
    assert "phase" in rv.OBSERVABILITY_ALLOWED_DIMENSIONS
    assert "phase_scope" not in rv.OBSERVABILITY_ALLOWED_DIMENSIONS

    config, scope, mechanism = views["config"], views["scope"], views["mechanism"]
    with pytest.raises(ValueError, match="unknown dimension"):
        rv.voltvar_denominator_view(config, scope, mechanism=mechanism, dimensions=("phase",))
    with pytest.raises(ValueError, match="unknown dimension"):
        rv.observability_status_view(
            config, scope, mechanism=mechanism, dimensions=("phase_scope",)
        )


# ---------------------------------------------------------------------------
# UTC month dimensions
# ---------------------------------------------------------------------------


def test_utc_month_dimension_groups_rows_from_different_sites_together(views) -> None:
    config, scope, mechanism = views["config"], views["scope"], views["mechanism"]
    # response_rows has three rows: S1/A, S1/B and S3/A, all year_utc=2025,
    # month_utc=4 -- grouping by (year_utc, month_utc) must merge all three
    # into a single row, not split them by any local-time key (there is no
    # timestamp_local column in this table at all).
    by_month = rv.observability_status_view(
        config, scope, mechanism=mechanism, dimensions=("year_utc", "month_utc")
    )
    assert len(by_month) == 1
    assert int(by_month["n_site_phase_months"].iloc[0]) == len(views["response_rows"])


def test_no_local_timestamp_dimension_is_ever_allowed() -> None:
    assert "timestamp_local" not in rv.CURVE_ALLOWED_DIMENSIONS
    assert "timestamp_local" not in rv.OBSERVABILITY_ALLOWED_DIMENSIONS


# ---------------------------------------------------------------------------
# Mixed methodology/provenance rejection
# ---------------------------------------------------------------------------


def test_mixed_methodology_within_one_table_is_rejected(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    mechanism = MechanismAnalysisConfig().validate()
    rows = [
        _voltvar_row(methodology_id="id_a"),
        _voltvar_row(serial="S2", methodology_id="id_b"),
    ]
    _write_parquet(rows, voltvar_results_path(config, scope, mechanism))
    _write_parquet([_voltwatt_row()], voltwatt_results_path(config, scope, mechanism))
    _write_parquet([_response_row()], response_observability_path(config, scope))

    with pytest.raises(ValueError, match="distinct values"):
        rv.result_context(config, scope, mechanism=mechanism)


def test_mixed_methodology_across_voltvar_and_voltwatt_is_rejected(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    mechanism = MechanismAnalysisConfig().validate()
    _write_parquet(
        [_voltvar_row(methodology_id="id_a")],
        voltvar_results_path(config, scope, mechanism),
    )
    _write_parquet(
        [_voltwatt_row(methodology_id="id_b")],
        voltwatt_results_path(config, scope, mechanism),
    )
    _write_parquet([_response_row()], response_observability_path(config, scope))

    with pytest.raises(ValueError, match="disagrees"):
        rv.result_context(config, scope, mechanism=mechanism)


def test_response_observability_methodology_mismatch_is_flagged_not_raised(views) -> None:
    """response_observability is deliberately never rebuilt per phase_scope_basis
    track, so its methodology_id legitimately differs from an all_phases
    curve-table run. That must be reported, not raised as an error.
    """

    config, scope = views["config"], views["scope"]
    all_phases = MechanismAnalysisConfig(phase_scope_basis="all_phases").validate()
    _write_parquet(
        [_voltvar_row(methodology_id="all_phases_id")],
        voltvar_results_path(config, scope, all_phases),
    )
    _write_parquet(
        [_voltwatt_row(methodology_id="all_phases_id")],
        voltwatt_results_path(config, scope, all_phases),
    )
    context = rv.result_context(config, scope, mechanism=all_phases)
    assert context["methodology_id"] == "all_phases_id"
    assert context["response_observability_methodology_id"] == _DEFAULT_METHODOLOGY_ID
    assert context["response_observability_methodology_matches_curve_tables"] is False


# ---------------------------------------------------------------------------
# No combined score; curtailment has no numeric estimate
# ---------------------------------------------------------------------------


def test_no_combined_score_or_view_exists() -> None:
    for module in (rv, rp):
        public_names = [name for name in dir(module) if not name.startswith("_")]
        offending = [name for name in public_names if "score" in name.lower()]
        assert not offending, f"{module.__name__} must not define a combined score: {offending}"


def test_unavailable_curtailment_has_no_numeric_estimate() -> None:
    context = rv.CURTAILMENT_UNAVAILABLE_CONTEXT
    assert context["status"] == "unavailable"
    assert "gate 7" in context["reason"]
    for value in context.values():
        assert isinstance(value, str), "curtailment context must contain no numeric fields"

    signature = inspect.signature(rp.plot_curtailment_unavailable)
    for parameter in signature.parameters.values():
        if parameter.name == "ax":
            continue
        assert parameter.default is not inspect.Parameter.empty or parameter.name == "curtailment_context", (
            "plot_curtailment_unavailable must not require a caller-supplied numeric argument"
        )

    ax = rp.plot_curtailment_unavailable()
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "unavailable" in texts.lower()
    assert "gate 7" in texts.lower()


# ---------------------------------------------------------------------------
# Dual-track path resolution (Delivery 5's extension beyond the original spec)
# ---------------------------------------------------------------------------


def test_der_inferred_and_all_phases_tracks_do_not_collide(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    der_inferred = MechanismAnalysisConfig().validate()
    all_phases = MechanismAnalysisConfig(phase_scope_basis="all_phases").validate()

    _write_parquet(
        [_voltvar_row(n_source_intervals=10)],
        voltvar_results_path(config, scope, der_inferred),
    )
    _write_parquet(
        [_voltvar_row(n_source_intervals=99)],
        voltvar_results_path(config, scope, all_phases),
    )

    der_view = rv.voltvar_denominator_view(config, scope, mechanism=der_inferred)
    all_view = rv.voltvar_denominator_view(config, scope, mechanism=all_phases)
    assert int(der_view["n_source_intervals"].iloc[0]) == 10
    assert int(all_view["n_source_intervals"].iloc[0]) == 99


def test_missing_file_raises_file_not_found(tmp_path) -> None:
    config = _config(tmp_path)
    scope = config.scope(None, None)
    mechanism = MechanismAnalysisConfig().validate()
    with pytest.raises(FileNotFoundError):
        rv.voltvar_denominator_view(config, scope, mechanism=mechanism)
