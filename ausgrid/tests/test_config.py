from __future__ import annotations

from pathlib import Path

import pytest

from ausgrid_analysis.config import SourceScope, load_config


def _toml(tmp_path: Path, active_sign: int = -1) -> str:
    return f"""
[paths]
telemetry_parquet = '{(tmp_path / "telemetry.parquet").as_posix()}'
metadata_workbook = '{(tmp_path / "metadata.xlsx").as_posix()}'
derived_root = '{(tmp_path / "derived").as_posix()}'

[metadata]
sheet_name = "Cust_DER_Network Data"
id_column = "Unique Number ID"

[telemetry]
site_column = "serial"
timestamp_column = "MeasureTime"
phase_column = "Vphase"
voltage_column = "Volts"
current_column = "Curr"
reactive_power_column = "ReactPow"
active_power_column = "ActivePow"
source_month_column = "month"
source_file_column = "source_file"

[assumptions]
active_export_sign = {active_sign}
reactive_absorbing_sign = -1
source_timezone = "UTC"
local_timezone = "Australia/Sydney"
power_sample_type = "instantaneous"
measurement_location = "unknown"

[quality]
voltage_min_v = 0.0
voltage_max_v = 300.0
duplicate_float_tolerance = 0.000001

[processing]
site_bucket_count = 32
threads = 4
memory_limit = "8GB"
parquet_compression = "zstd"
"""


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "analysis.toml"
    path.write_text(_toml(tmp_path), encoding="utf-8")
    config = load_config(path)
    assert config.assumptions.active_export_sign == -1
    assert config.processing.site_bucket_count == 32
    assert config.paths.derived_root == tmp_path / "derived"


def test_invalid_sign_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "analysis.toml"
    path.write_text(_toml(tmp_path, active_sign=0), encoding="utf-8")
    with pytest.raises(ValueError, match="active_export_sign"):
        load_config(path)


def test_scope_validation_and_labels() -> None:
    scope = SourceScope(month="2025-04", site_bucket=2, bucket_count=32).validate()
    assert scope.label == "month_2025_04__bucket_2_of_32"
    assert not scope.is_full
    with pytest.raises(ValueError, match="YYYY-MM"):
        SourceScope(month="April-2025").validate()

