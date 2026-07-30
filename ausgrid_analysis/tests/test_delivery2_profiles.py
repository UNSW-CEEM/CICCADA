from __future__ import annotations

from pathlib import Path

import pandas as pd

from ausgrid_analysis.config import (
    AssumptionConfig,
    Delivery2Config,
    FoundationConfig,
    MetadataConfig,
    PathConfig,
    ProcessingConfig,
    QualityConfig,
    TelemetryConfig,
)
from ausgrid_analysis.delivery2_profiles import derive_site_profiles


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
        delivery2=Delivery2Config(),
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
    assert bool(result["delivery2_primary_cohort"])


def test_battery_is_never_in_primary_cohort(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [{"serial": "2", "phase": "B", "power_measurement_available": True,
          "solar_signature_w": 500.0, "metadata_available": True,
          "analysis_cohort": "solar_battery", "has_battery": True,
          "install_phase_count": 1}]
    )
    result = derive_site_profiles(frame, _config(tmp_path)).iloc[0]
    assert result["phase_mapping_confidence"] == "high"
    assert not bool(result["delivery2_primary_cohort"])
    assert not bool(result["formal_inverter_conformance_assessable"])


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
    assert not bool(result["delivery2_primary_cohort"])
