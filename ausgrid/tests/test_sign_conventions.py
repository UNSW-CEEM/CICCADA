from __future__ import annotations

from ausgrid_analysis.schemas import (
    normalize_active_power,
    normalize_reactive_power,
)


def test_negative_active_power_becomes_positive_export() -> None:
    assert normalize_active_power(-5000.0, export_sign=-1) == 5000.0
    assert normalize_active_power(800.0, export_sign=-1) == -800.0


def test_negative_reactive_power_means_absorbing() -> None:
    absorbing, generator_q = normalize_reactive_power(-600.0, absorbing_sign=-1)
    assert absorbing == 600.0
    assert generator_q == -600.0


def test_signs_can_be_flipped_without_changing_raw_values() -> None:
    absorbing, generator_q = normalize_reactive_power(600.0, absorbing_sign=1)
    assert absorbing == 600.0
    assert generator_q == -600.0

