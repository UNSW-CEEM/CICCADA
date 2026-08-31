from __future__ import annotations

from pathlib import Path

import pandas as pd

from ausgrid_analysis.config import (
    AssumptionConfig,
    StructuredTelemetryConfig,
    FoundationConfig,
    MetadataConfig,
    PathConfig,
    ProcessingConfig,
    QualityConfig,
    TelemetryConfig,
)
from ausgrid_analysis.telemetry_profiles import derive_site_profiles


def _config(tmp_path: Path) -> FoundationConfig:
    return FoundationConfig(
        paths=PathConfig(tmp_path / "source.parquet", tmp_path / "meta.xlsx", tmp_path),
        metadata=MetadataConfig("Sheet1", "ID"),
        telemetry=TelemetryConfig(
            "serial", "time", "phase", "voltage", "current", "q", "p", "month", "file"
        ),
        assumptions=AssumptionConfig(
            -1, -1, "UTC", "Australia/Sydney", "instantaneous", "revenue_meter"
        ),
        quality=QualityConfig(0.0, 300.0, 1e-6),
        processing=ProcessingConfig(32, 2, "1GB", "zstd"),
        structured_telemetry=StructuredTelemetryConfig(),
    )


def test_single_phase_mapping_uses_strong_day_night_signature(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"serial": "1", "phase": "A", "power_measurement_available": True,
             "solar_signature_w": 1000.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 1},
            {"serial": "1", "phase": "B", "power_measurement_available": True,
             "solar_signature_w": 50.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 1},
            {"serial": "1", "phase": "C", "power_measurement_available": True,
             "solar_signature_w": 20.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 1},
        ]
    )
    result = derive_site_profiles(frame, _config(tmp_path)).iloc[0]
    assert result["inferred_der_phases"] == "A"
    assert result["phase_mapping_confidence"] == "high"
    assert bool(result["solar_only_mapped_cohort"])


def test_battery_is_never_in_primary_cohort(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [{"serial": "2", "phase": "B", "power_measurement_available": True,
          "solar_signature_w": 500.0, "metadata_available": True,
          "analysis_cohort": "solar_battery", "has_battery": True,
          "install_phase_count": 1}]
    )
    result = derive_site_profiles(frame, _config(tmp_path)).iloc[0]
    assert result["phase_mapping_confidence"] == "high"
    assert not bool(result["solar_only_mapped_cohort"])
    assert not bool(result["formal_inverter_conformance_assessable"])


def test_equal_count_all_phases_pass_signature_floor_stays_high(tmp_path: Path) -> None:
    # install_phase_count matches power-measured phase count AND every phase
    # clears the signature floor -- unchanged, existing behaviour.
    frame = pd.DataFrame(
        [
            {"serial": "4", "phase": "A", "power_measurement_available": True,
             "solar_signature_w": 3000.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 2},
            {"serial": "4", "phase": "B", "power_measurement_available": True,
             "solar_signature_w": 2500.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 2},
        ]
    )
    result = derive_site_profiles(frame, _config(tmp_path)).iloc[0]
    assert result["phase_mapping_method"] == "all_power_available_phases"
    assert result["inferred_der_phases"] == "A|B"
    assert result["phase_mapping_confidence"] == "high"


def test_equal_count_one_weak_phase_is_filtered_not_trusted(tmp_path: Path) -> None:
    # This is the 810584444 pattern: install_phase_count == power_phase_count
    # (3 == 3), but only one phase actually looks like solar. The other two
    # must not be silently stamped as DER-connected just because the counts
    # matched.
    frame = pd.DataFrame(
        [
            {"serial": "5", "phase": "A", "power_measurement_available": True,
             "solar_signature_w": 2567.35, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 3},
            {"serial": "5", "phase": "B", "power_measurement_available": True,
             "solar_signature_w": -598.58, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 3},
            {"serial": "5", "phase": "C", "power_measurement_available": True,
             "solar_signature_w": 0.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 3},
        ]
    )
    result = derive_site_profiles(frame, _config(tmp_path)).iloc[0]
    assert result["phase_mapping_method"] == "signature_filtered_from_install_count"
    assert result["inferred_der_phases"] == "A"
    assert result["phase_mapping_confidence"] == "high"
    assert bool(result["phase_mapping_assessable"])
    assert bool(result["solar_only_mapped_cohort"])


def test_equal_count_no_phase_clears_floor_is_low_confidence(tmp_path: Path) -> None:
    # install_phase_count == power_phase_count, but nothing looks like solar
    # at all -- must not be stamped 'high' just because the counts matched.
    frame = pd.DataFrame(
        [
            {"serial": "6", "phase": "A", "power_measurement_available": True,
             "solar_signature_w": 10.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 2},
            {"serial": "6", "phase": "B", "power_measurement_available": True,
             "solar_signature_w": -25.0, "metadata_available": True,
             "analysis_cohort": "solar_only", "has_battery": False,
             "install_phase_count": 2},
        ]
    )
    result = derive_site_profiles(frame, _config(tmp_path)).iloc[0]
    assert result["phase_mapping_method"] == "signature_filtered_from_install_count"
    assert result["inferred_der_phases"] == ""
    assert result["phase_mapping_confidence"] == "low"
    assert not bool(result["phase_mapping_assessable"])
    assert not bool(result["solar_only_mapped_cohort"])


def test_missing_power_phases_is_insufficient_not_zero(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [{"serial": "3", "phase": "A", "power_measurement_available": False,
          "solar_signature_w": None, "metadata_available": True,
          "analysis_cohort": "solar_only", "has_battery": False,
          "install_phase_count": 1}]
    )
    result = derive_site_profiles(frame, _config(tmp_path)).iloc[0]
    assert result["phase_mapping_confidence"] == "insufficient"
    assert result["inferred_der_phases"] == ""
    assert not bool(result["solar_only_mapped_cohort"])
