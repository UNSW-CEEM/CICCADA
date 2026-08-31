from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd

from dnsp_analysis.canonical import build_canonical_phase
from dnsp_analysis.config import (
    AssumptionConfig,
    FoundationConfig,
    MetadataConfig,
    PathConfig,
    ProcessingConfig,
    QualityConfig,
    SourceScope,
    TelemetryConfig,
)
from dnsp_analysis.db import canonical_output_path
from dnsp_analysis.duplicates import run_duplicate_audit
from dnsp_analysis.metadata import prepare_metadata
from dnsp_analysis.schemas import (
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


# ---------------------------------------------------------------------------
# Integration-level check: canonical.py's SQL, not just the pure-Python twin.
#
# 2026-08-04: reactive_absorbing_sign moved from -1 to +1 in analysis.toml
# (see that file's comments for the full reasoning -- under -1, the two
# negations in canonical_query's q_generator_var formula canceled and Q was
# silently left unconverted from raw). This test locks in the corrected,
# single-negation contract at the level that actually runs in production:
# the SQL query, not just normalize_reactive_power's pure-Python twin above.
# ---------------------------------------------------------------------------


def _config(tmp_path) -> FoundationConfig:
    return FoundationConfig(
        paths=PathConfig(
            telemetry_parquet=tmp_path / "telemetry.parquet",
            metadata_workbook=tmp_path / "metadata.xlsx",
            derived_root=tmp_path / "derived",
        ),
        metadata=MetadataConfig(sheet_name="Cust_DER_Network Data", id_column="Unique Number ID"),
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
        # active_export_sign=-1, reactive_absorbing_sign=+1: the corrected
        # production pair as of 2026-08-04.
        assumptions=AssumptionConfig(
            active_export_sign=-1,
            reactive_absorbing_sign=1,
            source_timezone="UTC",
            local_timezone="Australia/Sydney",
            power_sample_type="instantaneous",
            measurement_location="revenue_meter",
        ),
        quality=QualityConfig(voltage_min_v=0.0, voltage_max_v=300.0, duplicate_float_tolerance=0.000001),
        processing=ProcessingConfig(site_bucket_count=4, threads=1, memory_limit="1GB", parquet_compression="zstd"),
    ).validate()


def _write_telemetry(config: FoundationConfig) -> None:
    frame = pd.DataFrame(
        [
            # Raw ActivePow=800 (importing, per DNSP's documented load
            # convention) with ReactPow=300 (absorbing).
            [101, datetime(2025, 4, 2, 0, 0), "A", 250.0, 2.0, 300.0, 800.0, "2025-04", "one"],
            # Raw ActivePow=-1200 (exporting) with ReactPow=-450 (supplying).
            [101, datetime(2025, 4, 2, 0, 5), "A", 251.0, 2.1, -450.0, -1200.0, "2025-04", "one"],
        ],
        columns=[
            "serial", "MeasureTime", "Vphase", "Volts", "Curr",
            "ReactPow", "ActivePow", "month", "source_file",
        ],
    )
    connection = duckdb.connect()
    connection.register("telemetry", frame)
    connection.execute("COPY telemetry TO ? (FORMAT PARQUET)", [str(config.paths.telemetry_parquet)])
    connection.close()


def _write_metadata(config: FoundationConfig) -> None:
    frame = pd.DataFrame(
        [[101, "No", "Solar", 1, 10, 6.6, "Maker", "Model", None, None, None, None, 2024, None, 1, 10, -33.8, 151.2, 0]],
        columns=[
            "Unique Number ID", "Controlled Load", "DER_Type", "Install Phase",
            "Approved Capacity (kW)", "Solar_kW (total capacity)", "Solar Manufacturer",
            "Solar Model", "Battery_kWh", "Battery Manufacturer", "Battery Model",
            "Battery Inverter Capacity (kW)", "Solar Install Year", "Battery Install Year",
            "Zone External ID", "Sub External ID", "Sub Lat", "Sub Long", "FY26 EVM Test",
        ],
    )
    with pd.ExcelWriter(config.paths.metadata_workbook, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=config.metadata.sheet_name, index=False)


def test_canonical_query_applies_single_negation_to_p_and_q(tmp_path) -> None:
    config = _config(tmp_path)
    _write_telemetry(config)
    _write_metadata(config)
    scope = SourceScope(bucket_count=config.processing.site_bucket_count)

    run_duplicate_audit(config, scope)
    prepare_metadata(config)
    build_canonical_phase(config, scope)

    frame = duckdb.sql(
        f"""
        SELECT active_power_raw_w, reactive_power_raw_var,
               p_export_w, q_absorbing_var, q_generator_var
        FROM read_parquet({str(canonical_output_path(config, scope) / '**' / '*.parquet')!r},
                           hive_partitioning=true)
        ORDER BY active_power_raw_w
        """
    ).fetchdf()

    exporting = frame.loc[frame["active_power_raw_w"] == -1200.0].iloc[0]
    assert exporting["p_export_w"] == 1200.0  # raw import-negative -> export-positive
    assert exporting["reactive_power_raw_var"] == -450.0
    assert exporting["q_absorbing_var"] == -450.0  # unchanged: absorbing_sign=+1
    assert exporting["q_generator_var"] == 450.0  # single negation: supplying -> +Q

    importing = frame.loc[frame["active_power_raw_w"] == 800.0].iloc[0]
    assert importing["p_export_w"] == -800.0  # raw import-positive -> export-negative
    assert importing["reactive_power_raw_var"] == 300.0
    assert importing["q_absorbing_var"] == 300.0  # unchanged: absorbing_sign=+1
    assert importing["q_generator_var"] == -300.0  # single negation: absorbing -> -Q

    # The identity this whole fix hinges on: generator Q is always the exact
    # negation of absorbing-convention Q, for every row, regardless of sign.
    assert (frame["q_generator_var"] == -frame["q_absorbing_var"]).all()

