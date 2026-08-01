from __future__ import annotations

import duckdb
import pytest

from ausgrid_analysis.as4777_curves import (
    VOLT_VAR,
    add_tolerance,
    classify_voltvar_interval,
    q_conformance_floor_absorbing,
    q_conformance_floor_absorbing_sql,
    q_impact_nearest_edge,
    vvar_required_q,
    vvar_required_q_sql,
    vw_max_p,
    vw_max_p_sql,
)


def test_signed_breakpoints_and_tolerance() -> None:
    assert VOLT_VAR.q4 == -0.60
    assert vvar_required_q(207, 10) == pytest.approx(4.4)
    assert vvar_required_q(220, 10) == pytest.approx(0)
    assert vvar_required_q(240, 10) == pytest.approx(0)
    assert vvar_required_q(249, 10) == pytest.approx(-3.0)
    assert vvar_required_q(258, 10) == pytest.approx(-6.0)
    assert vw_max_p(253, 10) == pytest.approx(10)
    assert vw_max_p(256.5, 10) == pytest.approx(6)
    assert vw_max_p(260, 10) == pytest.approx(2)
    assert add_tolerance(6, 10, direction=1) == pytest.approx(6.4)


def test_figure_2_1_corrected_floor_does_not_relax_above_eighty_percent() -> None:
    assert q_conformance_floor_absorbing(1.9, 10) == pytest.approx(0)
    assert q_conformance_floor_absorbing(2.0, 10) == pytest.approx(-4.4)
    assert q_conformance_floor_absorbing(6.0, 10) == pytest.approx(-4.4)
    assert q_conformance_floor_absorbing(7.0, 10) == pytest.approx(-5.25)
    assert q_conformance_floor_absorbing(9.0, 10) == pytest.approx(-6.0)


@pytest.mark.parametrize("voltage", [200, 207, 213.5, 220, 240, 249, 258, 265])
def test_python_and_duckdb_curve_expressions_agree(voltage: float) -> None:
    capacity = 10.0
    row = duckdb.sql(
        "SELECT "
        + vvar_required_q_sql(str(voltage), str(capacity))
        + ", "
        + vw_max_p_sql(str(voltage), str(capacity))
        + ", "
        + q_conformance_floor_absorbing_sql("7.0", str(capacity))
    ).fetchone()
    assert row[0] == pytest.approx(vvar_required_q(voltage, capacity))
    assert row[1] == pytest.approx(vw_max_p(voltage, capacity))
    assert float(row[2]) == pytest.approx(
        q_conformance_floor_absorbing(7.0, capacity)
    )


def test_q_impact_uses_generator_convention_without_flipping_curve() -> None:
    assert q_impact_nearest_edge(-5.0, -6.0, -4.0) > 0
    assert q_impact_nearest_edge(+5.0, -6.0, -4.0) < 0
    assert q_impact_nearest_edge(-5.0, -6.0, -4.0, assessable=False) is None


def test_scalar_voltvar_classification_is_promoted_into_package() -> None:
    within = classify_voltvar_interval(258.0, 5.0, -6.0, 10.0)
    adverse = classify_voltvar_interval(258.0, 5.0, +6.0, 10.0)
    missing_rating = classify_voltvar_interval(258.0, 5.0, -6.0, None)
    assert within.status == "conforming"
    assert adverse.status == "Q_adverse"
    assert missing_rating.status == "not_assessable"
