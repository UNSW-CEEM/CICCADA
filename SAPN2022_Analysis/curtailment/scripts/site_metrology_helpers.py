from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

CONFORMANCE_DIR = Path(__file__).resolve().parents[2] / "conformance"
if str(CONFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_DIR))

from funcs import (
    addLocalTStamp,
    addPolarityToPower,
    addValidVoltage,
    mapCircuitDataToSite,
)

ADELAIDE_TZ = "Australia/Adelaide"
POWER_W_TO_KW = 1000


def empty_site_metrology_frame():
    return pl.DataFrame(
        schema={
            "dataset_role": pl.Utf8,
            "site_id": pl.Int64,
            "t_stamp": pl.Datetime,
            "P_kw": pl.Float64,
            "V": pl.Float64,
        }
    )


def _prepare_cleaned_metrology(raw, circuitDetails, power_divisor, start_date, end_date):
    cleaned = (
        raw.with_columns([
            pl.col("c_id").cast(pl.Int64),
            pl.col("utc_tstamp").cast(pl.Utf8),
            pl.col("power").cast(pl.Float64, strict=False),
            pl.col("voltage").cast(pl.Float64, strict=False),
            pl.col("vmean").cast(pl.Float64, strict=False),
            pl.col("duration").cast(pl.Int64, strict=False),
        ])
        .with_columns((pl.col("power") / power_divisor).alias("power"))
    )
    cleaned = addLocalTStamp(cleaned, add=True)
    cleaned = addValidVoltage(cleaned)
    cleaned = addPolarityToPower(cleaned, circuitDetails)

    if start_date is not None and end_date is not None:
        cleaned = cleaned.filter(
            pl.col("local_tstamp").dt.date().is_between(start_date, end_date, closed="both")
        )

    return cleaned.select([
        "c_id",
        "utc_tstamp",
        "duration",
        "power",
        "voltage_valid",
        "local_tstamp",
    ])


def prepare_sapn_metrology(sapn_data_path, start_date, end_date):
    """Read the cleaned SAPN validation parquet produced by data_processing.py."""
    sapn_data_path = Path(sapn_data_path)
    if not sapn_data_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned SAPN validation parquet at {sapn_data_path}. "
            "Run conformance/data_processing.py first or update "
            "SAPN_CLEANED_DATA_PATH."
        )

    prepared = pl.scan_parquet(sapn_data_path).with_columns([
        pl.col("c_id").cast(pl.Int64),
        pl.col("utc_tstamp").cast(pl.Utf8),
        pl.col("duration").cast(pl.Int64, strict=False),
        pl.col("power").cast(pl.Float64, strict=False),
        pl.col("voltage_valid").cast(pl.Float64, strict=False),
        pl.col("local_tstamp").cast(pl.Datetime(time_zone=ADELAIDE_TZ)),
    ])

    if start_date is not None and end_date is not None:
        prepared = prepared.filter(
            pl.col("local_tstamp")
            .dt.date()
            .is_between(start_date, end_date, closed="both")
        )

    return prepared.select([
        "c_id",
        "utc_tstamp",
        "duration",
        "power",
        "voltage_valid",
        "local_tstamp",
    ])


def prepare_evm_metrology(evm_parquets, evm_circuits, start_date, end_date):
    if not evm_parquets:
        raise FileNotFoundError("No EVM training parquet files found for the selected date range")

    raw = pl.scan_parquet([str(path) for path in evm_parquets])
    return _prepare_cleaned_metrology(
        raw,
        evm_circuits,
        POWER_W_TO_KW,
        start_date,
        end_date,
    )


def _build_structured_site_rows(wide, dataset_role):
    power_cols = [
        col for col in wide.columns
        if col.startswith("power")
        and not col.endswith("_next")
        and not col.endswith("_logic")
    ]
    voltage_cols = [col for col in wide.columns if col.startswith("voltage_valid")]

    if not power_cols or not voltage_cols:
        return empty_site_metrology_frame()

    df = wide.clone()
    for col in voltage_cols:
        rolled_name = f"vmean_rolling_10m{col.replace('voltage_valid', '', 1)}"
        rolled = (
            df.filter(pl.col(col).is_not_null())
            .with_columns(
                pl.col(col).rolling_mean_by(
                    by="local_tstamp",
                    window_size="10m",
                ).alias(rolled_name)
            )
            .select(["local_tstamp", rolled_name])
        )
        df = df.join(rolled, on="local_tstamp", how="left")

    vmean_cols = [col for col in df.columns if col.startswith("vmean_rolling_10m")]
    if not vmean_cols:
        return empty_site_metrology_frame()

    df = df.with_columns([
        pl.mean_horizontal([pl.col(col) for col in voltage_cols]).alias("V"),
        pl.mean_horizontal([pl.col(col) for col in vmean_cols]).alias("v10m_avg"),
        pl.sum_horizontal([
            pl.when(pl.col(col).cast(pl.Float64, strict=False).fill_null(0) < 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col(col).cast(pl.Float64, strict=False).fill_null(0))
            for col in power_cols
        ]).alias("site_power_kw"),
        pl.any_horizontal([pl.col(col).is_not_null() for col in power_cols]).alias("_has_power"),
    ])

    return (
        df.filter(pl.col("_has_power") & pl.col("v10m_avg").is_not_null())
        .select([
            pl.lit(dataset_role).alias("dataset_role"),
            pl.col("site_id").cast(pl.Int64),
            pl.col("utc_tstamp")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f", strict=False)
            .alias("t_stamp"),
            pl.col("site_power_kw").cast(pl.Float64).alias("P_kw"),
            pl.col("V").cast(pl.Float64),
        ])
        .filter(pl.col("t_stamp").is_not_null())
        .sort("t_stamp")
    )


def aggregate_site_metrology(allData, circuitDetails, site_ids, start_date, end_date, dataset_role):
    local_tz = ZoneInfo(ADELAIDE_TZ)
    start_day = datetime.combine(start_date, time.min, tzinfo=local_tz)
    end_day = datetime.combine(end_date, time.max, tzinfo=local_tz)
    site_frames = []

    for site_id in sorted(set(site_ids)):
        has_data, wide, _ = mapCircuitDataToSite(
            allData,
            circuitDetails,
            site_id,
            start_day,
            end_day,
        )
        if not has_data:
            continue

        site_frame = _build_structured_site_rows(wide, dataset_role)
        if not site_frame.is_empty():
            site_frames.append(site_frame)

    if not site_frames:
        return empty_site_metrology_frame()

    return pl.concat(site_frames, how="vertical")
