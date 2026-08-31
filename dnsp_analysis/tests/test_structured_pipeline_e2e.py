from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pandas as pd

from ausgrid_analysis.config import (
    AssumptionConfig,
    FoundationConfig,
    MetadataConfig,
    PathConfig,
    ProcessingConfig,
    QualityConfig,
    SourceScope,
    TelemetryConfig,
)
from ausgrid_analysis.db import (
    canonical_output_path,
    site_profile_path,
    structured_site_output_path,
)
from ausgrid_analysis.telemetry_profiles import build_site_profiles
from ausgrid_analysis.structured_intervals import (
    build_structured_phase,
    build_structured_site,
)
from ausgrid_analysis.structured_validation import validate_structured_telemetry
from ausgrid_analysis.metadata import metadata_output_path


def _config(tmp_path) -> FoundationConfig:
    return FoundationConfig(
        paths=PathConfig(tmp_path / "raw.parquet", tmp_path / "meta.xlsx", tmp_path / "derived"),
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


def _write_inputs(config: FoundationConfig, scope: SourceScope) -> None:
    canonical = pd.DataFrame(
        [
            ["1", datetime(2025, 4, 1, 0, tzinfo=timezone.utc), datetime(2025, 4, 1, 11), "A", 240.0, 1.0, -500.0, -50.0, 500.0, 50.0, -50.0],
            ["1", datetime(2025, 4, 1, 0, tzinfo=timezone.utc), datetime(2025, 4, 1, 11), "B", 241.0, None, None, None, None, None, None],
            ["1", datetime(2025, 4, 1, 1, tzinfo=timezone.utc), datetime(2025, 4, 1, 12), "A", 0.0, 1.0, -600.0, -60.0, 600.0, 60.0, -60.0],
            ["2", datetime(2025, 4, 1, 1, tzinfo=timezone.utc), datetime(2025, 4, 1, 12), "C", 245.0, 2.0, -700.0, -70.0, 700.0, 70.0, -70.0],
        ],
        columns=[
            "serial", "timestamp_utc", "timestamp_local", "phase", "voltage_v",
            "current_a", "active_power_raw_w", "reactive_power_raw_var",
            "p_export_w", "q_absorbing_var", "q_generator_var",
        ],
    )
    canonical = canonical.assign(
        source_month="2025-04", source_file="source", duplicate_count=1,
        duplicate_source_file_count=1, duplicate_status="unique",
        row_has_null_measurement=False,
        voltage_physical_ok=canonical.voltage_v.gt(0),
        metadata_available=True, analysis_cohort="solar_only",
        install_phase_count=1, year_utc=2025, month_utc=4, site_bucket=0,
    )
    metadata = pd.DataFrame(
        [
            ["1", "solar_only", False, 1, 6.6, 5.0, None],
            ["2", "solar_battery", True, 1, 8.0, 5.0, 5.0],
        ],
        columns=[
            "serial", "analysis_cohort", "has_battery", "install_phase_count",
            "solar_capacity_kw", "approved_capacity_kw",
            "battery_inverter_capacity_kw",
        ],
    )
    connection = duckdb.connect()
    connection.register("canonical_frame", canonical)
    canonical_output_path(config, scope).parent.mkdir(parents=True)
    connection.execute(
        f"COPY canonical_frame TO '{canonical_output_path(config, scope)}' "
        "(FORMAT PARQUET, PARTITION_BY (year_utc, month_utc, site_bucket))"
    )
    metadata_output_path(config).parent.mkdir(parents=True)
    connection.register("metadata_frame", metadata)
    connection.execute(
        f"COPY metadata_frame TO '{metadata_output_path(config)}' (FORMAT PARQUET)"
    )
    connection.close()


def test_structured_telemetry_structures_and_validates_without_claiming_conformance(tmp_path) -> None:
    config = _config(tmp_path)
    scope = SourceScope(bucket_count=4)
    _write_inputs(config, scope)

    build_site_profiles(config, scope)
    build_structured_phase(config, scope)
    build_structured_site(config, scope)
    validation = validate_structured_telemetry(config, scope)

    assert validation["status"] == "pass"
    assert validation["structured_phase_rows"] == 4
    assert validation["structured_site_rows"] == 3
    assert validation["formal_conformance_rows"] == 0
    profile = duckdb.sql(
        f"SELECT * FROM read_parquet('{site_profile_path(config, scope)}') "
        "WHERE serial='2'"
    ).fetchone()
    assert profile is not None
    site_glob = str(structured_site_output_path(config, scope) / "**" / "*.parquet")
    result = duckdb.sql(
        f"SELECT count_if(voltage_min_valid_v IS NULL) FROM "
        f"read_parquet('{site_glob}', hive_partitioning=true)"
    ).fetchone()
    assert result == (1,)
