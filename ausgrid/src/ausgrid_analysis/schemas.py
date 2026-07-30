"""Column contracts, SQL helpers and small pure transformations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import FoundationConfig, SourceScope


METADATA_RENAME_MAP = {
    "Unique Number ID": "serial",
    "Controlled Load": "controlled_load",
    "DER_Type": "der_type",
    "Install Phase": "install_phase_count",
    "Approved Capacity (kW)": "approved_capacity_kw",
    "Solar_kW (total capacity)": "solar_capacity_kw",
    "Solar Manufacturer": "solar_manufacturer",
    "Solar Model": "solar_model",
    "Battery_kWh": "battery_kwh",
    "Battery Manufacturer": "battery_manufacturer",
    "Battery Model": "battery_model",
    "Battery Inverter Capacity (kW)": "battery_inverter_capacity_kw",
    "Solar Install Year": "solar_install_year",
    "Battery Install Year": "battery_install_year",
    "Zone External ID": "zone_external_id",
    "Sub External ID": "sub_external_id",
    "Sub Lat": "sub_lat",
    "Sub Long": "sub_long",
    "FY26 EVM Test": "fy26_evm_test",
}


NUMERIC_METADATA_COLUMNS = (
    "install_phase_count",
    "approved_capacity_kw",
    "solar_capacity_kw",
    "battery_kwh",
    "battery_inverter_capacity_kw",
    "solar_install_year",
    "battery_install_year",
    "zone_external_id",
    "sub_external_id",
    "sub_lat",
    "sub_long",
    "fy26_evm_test",
)


def sql_string(value: str | Path) -> str:
    """Return a safely quoted DuckDB string literal."""

    return "'" + str(value).replace("'", "''") + "'"


def quote_identifier(value: str) -> str:
    """Return a safely quoted SQL identifier."""

    return '"' + value.replace('"', '""') + '"'


def source_relation_sql(config: FoundationConfig) -> str:
    return f"read_parquet({sql_string(config.paths.telemetry_parquet)})"


def source_projection_sql(config: FoundationConfig) -> str:
    """Canonical source projection with provider columns renamed once."""

    c = config.telemetry
    return f"""
        SELECT
            CAST({quote_identifier(c.site_column)} AS VARCHAR) AS serial,
            CAST({quote_identifier(c.timestamp_column)} AS TIMESTAMP) AS measure_time,
            CAST({quote_identifier(c.phase_column)} AS VARCHAR) AS phase,
            CAST({quote_identifier(c.voltage_column)} AS DOUBLE) AS voltage_v,
            CAST({quote_identifier(c.current_column)} AS DOUBLE) AS current_a,
            CAST({quote_identifier(c.reactive_power_column)} AS DOUBLE)
                AS reactive_power_raw_var,
            CAST({quote_identifier(c.active_power_column)} AS DOUBLE)
                AS active_power_raw_w,
            CAST({quote_identifier(c.source_month_column)} AS VARCHAR) AS source_month,
            CAST({quote_identifier(c.source_file_column)} AS VARCHAR) AS source_file
        FROM {source_relation_sql(config)}
    """.strip()


def scope_predicate_sql(scope: SourceScope, alias: str = "s") -> str:
    predicates = ["1 = 1"]
    if scope.month is not None:
        predicates.append(f"{alias}.source_month = {sql_string(scope.month)}")
    if scope.site_bucket is not None:
        predicates.append(
            f"mod(hash({alias}.serial), {scope.bucket_count}) = {scope.site_bucket}"
        )
    return " AND ".join(predicates)


def scoped_source_sql(config: FoundationConfig, scope: SourceScope) -> str:
    return f"""
        SELECT *
        FROM ({source_projection_sql(config)}) AS s
        WHERE {scope_predicate_sql(scope, "s")}
    """.strip()


def physical_payload_hash_sql(tolerance: float, alias: str = "s") -> str:
    """Hash physical values after tolerance-based quantisation."""

    scale = 1.0 / float(tolerance)
    cols = (
        f"round({alias}.voltage_v * {scale})",
        f"round({alias}.current_a * {scale})",
        f"round({alias}.reactive_power_raw_var * {scale})",
        f"round({alias}.active_power_raw_w * {scale})",
    )
    return f"hash({', '.join(cols)})"


def normalize_active_power(raw_power: float | None, export_sign: int) -> float | None:
    return None if raw_power is None else raw_power * export_sign


def normalize_reactive_power(
    raw_reactive_power: float | None,
    absorbing_sign: int,
) -> tuple[float | None, float | None]:
    """Return `(positive_absorption, generator_convention_q)`."""

    if raw_reactive_power is None:
        return None, None
    absorbing = raw_reactive_power * absorbing_sign
    return absorbing, -absorbing


def utc_to_local(value: datetime, local_timezone: str) -> datetime:
    """Interpret a naive timestamp as UTC and convert it to local time."""

    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return aware.astimezone(ZoneInfo(local_timezone))

