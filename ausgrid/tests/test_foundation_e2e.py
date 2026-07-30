from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd

from ausgrid_analysis.canonical import build_canonical_phase
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
from ausgrid_analysis.duplicates import run_duplicate_audit
from ausgrid_analysis.inventory import run_inventory
from ausgrid_analysis.metadata import prepare_metadata
from ausgrid_analysis.validation import validate_canonical_phase


def _config(tmp_path) -> FoundationConfig:
    return FoundationConfig(
        paths=PathConfig(
            telemetry_parquet=tmp_path / "telemetry.parquet",
            metadata_workbook=tmp_path / "metadata.xlsx",
            derived_root=tmp_path / "derived",
        ),
        metadata=MetadataConfig(
            sheet_name="Cust_DER_Network Data",
            id_column="Unique Number ID",
        ),
        telemetry=TelemetryConfig(
            site_column="serial",
            timestamp_column="MeasureTime",
            phase_column="Vphase",
            voltage_column="Volts",
            current_column="Curr",
            reactive_power_column="ReactPow",
            active_power_column="ActivePow",
            source_month_column="month",
            source_file_column="source_file",
        ),
        assumptions=AssumptionConfig(
            active_export_sign=-1,
            reactive_absorbing_sign=-1,
            source_timezone="UTC",
            local_timezone="Australia/Sydney",
            power_sample_type="instantaneous",
            measurement_location="unknown",
        ),
        quality=QualityConfig(
            voltage_min_v=0.0,
            voltage_max_v=300.0,
            duplicate_float_tolerance=0.000001,
        ),
        processing=ProcessingConfig(
            site_bucket_count=4,
            threads=1,
            memory_limit="1GB",
            parquet_compression="zstd",
        ),
    ).validate()


def _write_telemetry(config: FoundationConfig) -> None:
    frame = pd.DataFrame(
        [
            [101, datetime(2025, 4, 2, 0, 0), "A", 235.0, 2.0, -100.0, -450.0, "2025-04", "one"],
            [101, datetime(2025, 4, 2, 0, 5), "A", 236.0, 2.1, -110.0, -460.0, "2025-04", "one"],
            [101, datetime(2025, 4, 2, 0, 5), "A", 236.0, 2.1, -110.0, -460.0, "2025-04", "two"],
            [202, datetime(2025, 4, 2, 0, 10), "B", 240.0, 1.0, -50.0, -200.0, "2025-04", "one"],
            [202, datetime(2025, 4, 2, 0, 10), "B", 240.0, 1.0, -50.0, -220.0, "2025-04", "two"],
            [303, datetime(2025, 4, 2, 0, 15), "C", 245.0, 1.2, 25.0, 180.0, "2025-04", "one"],
        ],
        columns=[
            "serial",
            "MeasureTime",
            "Vphase",
            "Volts",
            "Curr",
            "ReactPow",
            "ActivePow",
            "month",
            "source_file",
        ],
    )
    connection = duckdb.connect()
    connection.register("telemetry", frame)
    connection.execute(
        "COPY telemetry TO ? (FORMAT PARQUET)",
        [str(config.paths.telemetry_parquet)],
    )
    connection.close()


def _write_metadata(config: FoundationConfig) -> None:
    frame = pd.DataFrame(
        [
            [101, "No", "Solar", 1, 10, 6.6, "Maker", "Model", None, None, None, None, 2024, None, 1, 10, -33.8, 151.2, 0],
            [999, "No", "Solar", 3, 30, 15.0, "Maker", "Model", None, None, None, None, 2023, None, 2, 20, -33.9, 151.1, 0],
        ],
        columns=[
            "Unique Number ID",
            "Controlled Load",
            "DER_Type",
            "Install Phase",
            "Approved Capacity (kW)",
            "Solar_kW (total capacity)",
            "Solar Manufacturer",
            "Solar Model",
            "Battery_kWh",
            "Battery Manufacturer",
            "Battery Model",
            "Battery Inverter Capacity (kW)",
            "Solar Install Year",
            "Battery Install Year",
            "Zone External ID",
            "Sub External ID",
            "Sub Lat",
            "Sub Long",
            "FY26 EVM Test",
        ],
    )
    with pd.ExcelWriter(config.paths.metadata_workbook, engine="openpyxl") as writer:
        frame.to_excel(
            writer,
            sheet_name=config.metadata.sheet_name,
            index=False,
        )


def test_foundation_stages_reconcile_and_account_for_rows(tmp_path) -> None:
    config = _config(tmp_path)
    _write_telemetry(config)
    _write_metadata(config)
    scope = SourceScope(bucket_count=config.processing.site_bucket_count)

    inventory = run_inventory(config, scope)
    metadata = prepare_metadata(config)
    duplicates = run_duplicate_audit(config, scope)
    canonical = build_canonical_phase(config, scope)
    validation = validate_canonical_phase(config, scope)

    assert inventory["overall"]["n_rows"] == 6
    assert metadata["reconciliation_counts"] == {
        "telemetry_only": 2,
        "matched": 1,
        "metadata_only": 1,
    }
    assert duplicates["identical_duplicate_keys"] == 1
    assert duplicates["conflicting_duplicate_keys"] == 1
    assert canonical["n_rows"] == 3
    assert validation["status"] == "pass"
    assert validation["expected_canonical_rows"] == 3
    assert validation["canonical_rows"] == 3
    assert validation["accounting_difference"] == 0
    assert validation["duplicate_keys_remaining"] == 0

    empty_scope = SourceScope(
        month="2025-05",
        bucket_count=config.processing.site_bucket_count,
    )
    scoped_metadata = prepare_metadata(config, empty_scope)
    assert scoped_metadata["metadata_parquet_status"] == "reused"
    assert scoped_metadata["reconciliation_counts"] == {"metadata_only": 2}
    assert "samples" in scoped_metadata["reconciliation_csv"]
