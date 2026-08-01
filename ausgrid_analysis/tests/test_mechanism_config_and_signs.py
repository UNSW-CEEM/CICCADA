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


def test_signs_must_be_independently_ready() -> None:
    config = MechanismAnalysisConfig(
        active_sign_review_state=(
            "empirically_supported_pending_provider_confirmation"
        ),
        reactive_sign_review_state="contradicted",
    ).validate()
    assert config.active_sign_ready
    assert not config.reactive_sign_ready
