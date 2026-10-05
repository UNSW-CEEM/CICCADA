"""Build tier-based curtailment extracts from saved site compliance.

This script produces two reporting tables:

- a timestamp-level flag table showing when a site sits above either its final
  LOS or OV1 threshold (used for high reso curtaliment)
- a 5-minute bucket summary of the same signals for downstream curtailment work
  (used for 5m reso curtialment)

Inputs come from three places:

- ``site_compliance.csv`` for the assessed tier-based site list and
  final LOS/OV1 thresholds
- the cleaned metrology parquet plus ``circuit_details.csv`` so site-level
  power and voltage can be rebuilt from the underlying circuit data

Important difference from ``site_compliance_timestamp_detail``:

- this script does *not* read that file
- it rebuilds the timestamp flags directly from cleaned metrology
- it keeps assessed sites but does not apply ``main.py``'s day-coverage
  exclusion gate, so excluded low-coverage site-days can still appear here
- we will use all data
- no filtering for days with less than 80% coverage
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

CONFORMANCE_DIR = Path(__file__).resolve().parents[1]
if str(CONFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_DIR))

from sapn2022_workflow.config import (
    DAY_END,
    DAY_EXTRACTION_START,
    EVENT_DAYS,
    EVENT_MONTH,
    EVENT_YEAR,
    LOCAL_TIMEZONE,
    PRIMARY_PHASE_B_METHOD,
)
from sapn2022_workflow.loading import load_sapn_cleaned_data
from sapn2022_workflow.reporting import SITE_COMPLIANCE_NAME
from sapn2022_workflow.sapn_paths import (
    CIRCUIT_DETAILS_PATH,
    CLEANED_SITE_DATA_PATH,
    CONFORMANCE_OUTPUT_DIR,
)
from sapn2022_workflow.site_preparation import (
    calculate_site_day_voltage_signals,
    extract_site_day,
    map_circuit_data_to_site,
    select_site_pv_data,
    trim_site_day_analysis_window,
)

SITE_COMPLIANCE_PATH = CONFORMANCE_OUTPUT_DIR / SITE_COMPLIANCE_NAME
CLEANED_DATA_PATH = CLEANED_SITE_DATA_PATH  # this is the cleaned circuit data parquet
OUTPUT_DIR = (
    CONFORMANCE_DIR / "updated results" / "phase b info for curtailment" / "tier based"
)
TIMESTAMP_OUTPUT_PATH = OUTPUT_DIR / "tier_based_timestamp_flags.csv"
BUCKET_OUTPUT_PATH = OUTPUT_DIR / "tier_based_5min_buckets.csv"

TIMESTAMP_OUTPUT_SCHEMA = {
    "site_id": pl.Int64,
    "event_day": pl.Int64,
    "local_tstamp": pl.Datetime(time_zone="Australia/Adelaide"),
    "utc_tstamp": pl.Utf8,
    "site_power_kw": pl.Float64,
    "v10m_avg": pl.Float64,
    "vinst_max": pl.Float64,
    "los_or_ov1_flag": pl.Int8,
}

BUCKET_OUTPUT_SCHEMA = {
    "site_id": pl.Int64,
    "event_day": pl.Int64,
    "bucket_5min_local": pl.Utf8,
    "tod_bin": pl.Utf8,
    "site_power_kw_min": pl.Float64,
    "site_power_kw_avg": pl.Float64,
    "site_power_kw_max": pl.Float64,
    "v10m_avg_min": pl.Float64,
    "v10m_avg_avg": pl.Float64,
    "v10m_avg_max": pl.Float64,
    "vinst_max_min": pl.Float64,
    "vinst_max_avg": pl.Float64,
    "vinst_max_max": pl.Float64,
    "los_or_ov1_flag": pl.Int8,
}


def _load_assessed_sites() -> pl.DataFrame:
    """Load assessed sites plus the thresholds needed to rebuild curtailment flags."""
    if not SITE_COMPLIANCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing required input: {SITE_COMPLIANCE_PATH}. Run conformance "
            "first to generate site compliance."
        )

    compliance_df = pl.read_csv(SITE_COMPLIANCE_PATH, null_values=[""])
    required_columns = {
        "site_id",
        "threshold_method",
        "overall_pass",
        "los_threshold_used",
        "ov1_threshold_used",
    }
    missing_columns = required_columns.difference(compliance_df.columns)
    if missing_columns:
        raise ValueError(
            f"Site compliance is missing required columns: {sorted(missing_columns)}"
        )

    assessed_sites = (
        compliance_df.filter(
            (pl.col("threshold_method") == PRIMARY_PHASE_B_METHOD)
            & pl.col("overall_pass").is_not_null()
        )
        .select(["site_id", "los_threshold_used", "ov1_threshold_used"])
        .with_columns(
            [
                pl.col("site_id").cast(pl.Int64),
                pl.col("los_threshold_used").cast(pl.Float64),
                pl.col("ov1_threshold_used").cast(pl.Float64),
            ]
        )
        .sort("site_id")
    )

    if assessed_sites["site_id"].n_unique() != assessed_sites.height:
        raise ValueError(
            "Site compliance contains duplicate assessed tier-based sites."
        )

    missing_thresholds = assessed_sites.filter(
        pl.col("los_threshold_used").is_null() | pl.col("ov1_threshold_used").is_null()
    )
    if not missing_thresholds.is_empty():
        bad_sites = missing_thresholds["site_id"].to_list()
        raise ValueError(
            f"Missing LOS/OV1 threshold values for assessed sites: {bad_sites[:10]}"
        )

    return assessed_sites


def _build_day_flag_frame(
    wide: pl.DataFrame,
    *,
    los_threshold_used: float,
    ov1_threshold_used: float,
) -> pl.DataFrame:
    """Rebuild the timestamp-level tier-based curtailment flags for one site-day."""
    # Identify site-day power and voltage columns produced by
    # ``map_circuit_data_to_site``.
    power_cols = [
        col
        for col in wide.columns
        if col.startswith("power")
        and not col.endswith("_next")
        and not col.endswith("_logic")
    ]
    voltage_cols = [col for col in wide.columns if col.startswith("voltage_valid")]
    if not power_cols or not voltage_cols:
        return pl.DataFrame(schema=TIMESTAMP_OUTPUT_SCHEMA)

    site_day_df = calculate_site_day_voltage_signals(
        wide,
        voltage_prefix="voltage_valid",
    )
    site_day_df = site_day_df.with_columns(
        [
            pl.sum_horizontal(
                [
                    # Negative PV power is clipped to zero so the site total reflects
                    # export/available generation rather than import artefacts.
                    pl.when(pl.col(col).cast(pl.Float64, strict=False).fill_null(0) < 0)
                    .then(pl.lit(0.0))
                    .otherwise(pl.col(col).cast(pl.Float64, strict=False).fill_null(0))
                    for col in power_cols
                ]
            ).alias("site_power_kw"),
            pl.any_horizontal([pl.col(col).is_not_null() for col in power_cols]).alias(
                "_has_power"
            ),
        ]
    )
    site_day_df = trim_site_day_analysis_window(site_day_df)
    site_day_df = site_day_df.with_columns(
        [
            # Keep only rows where we can compare site power against a valid 10-minute
            # voltage mean. Rows with power but no usable voltage stay out of the
            # curtailment outputs.
            (pl.col("_has_power") & pl.col("v10m_avg").is_not_null()).alias(
                "_keep_row"
            ),
            (pl.col("vinst_max") >= ov1_threshold_used)
            .fill_null(False)
            .alias("_ov1_kicks_in"),
        ]
    ).with_columns(
        [
            # OV1 takes priority over LOS at the same timestamp, matching the Phase B
            # responsibility ordering.
            (pl.col("_keep_row") & pl.col("_ov1_kicks_in")).alias("_ov1_responsible"),
            (
                pl.col("_keep_row")
                & ~pl.col("_ov1_kicks_in")
                & (pl.col("v10m_avg") >= los_threshold_used)
            )
            .fill_null(False)
            .alias("_los_responsible"),
        ]
    )

    timestamp_dtype = site_day_df.schema["local_tstamp"]
    if isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_zone:
        local_timestamp = pl.col("local_tstamp").dt.convert_time_zone(LOCAL_TIMEZONE)
    else:
        local_timestamp = pl.col("local_tstamp").dt.replace_time_zone(LOCAL_TIMEZONE)

    return (
        site_day_df.filter(pl.col("_keep_row"))
        .select(
            [
                pl.col("site_id").cast(pl.Int64),
                local_timestamp.alias("local_tstamp"),
                pl.col("utc_tstamp").cast(pl.Utf8),
                local_timestamp.dt.day().cast(pl.Int64).alias("event_day"),
                pl.col("site_power_kw").cast(pl.Float64),
                pl.col("v10m_avg").cast(pl.Float64),
                pl.col("vinst_max").cast(pl.Float64),
                (pl.col("_ov1_responsible") | pl.col("_los_responsible"))
                .cast(pl.Int8)
                .alias("los_or_ov1_flag"),
            ]
        )
        .sort("local_tstamp")
    )


def _build_bucket_frame(timestamp_df: pl.DataFrame) -> pl.DataFrame:
    """Summarise timestamp-level curtailment signals into 5-minute buckets."""
    if timestamp_df.is_empty():
        return pl.DataFrame(schema=BUCKET_OUTPUT_SCHEMA)

    return (
        timestamp_df.with_columns(
            pl.col("local_tstamp").dt.truncate("5m").alias("_bucket_floor")
        )
        .with_columns(
            # First compute the 5-minute floor, then label each row by the end
            # of that reporting bucket. Exact boundary timestamps keep their
            # own time; all other timestamps roll forward to the next 5-minute mark.
            pl.when(pl.col("local_tstamp") == pl.col("_bucket_floor"))
            .then(pl.col("_bucket_floor"))
            .otherwise(pl.col("_bucket_floor") + pl.duration(minutes=5))
            .alias("bucket_5min_local")
        )
        .with_columns(
            [
                pl.col("bucket_5min_local").dt.strftime("%H:%M:%S").alias("tod_bin"),
                pl.col("bucket_5min_local")
                .dt.strftime("%Y-%m-%d %H:%M:%S%z")
                .alias("_bucket_text_raw"),
            ]
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("_bucket_text_raw").str.slice(0, 22),
                    pl.lit(":"),
                    pl.col("_bucket_text_raw").str.slice(22, 2),
                ]
            ).alias("bucket_5min_local")
        )
        .group_by(["site_id", "event_day", "bucket_5min_local"])
        .agg(
            [
                pl.col("tod_bin").first().alias("tod_bin"),
                pl.col("site_power_kw").min().alias("site_power_kw_min"),
                pl.col("site_power_kw").mean().alias("site_power_kw_avg"),
                pl.col("site_power_kw").max().alias("site_power_kw_max"),
                pl.col("v10m_avg").min().alias("v10m_avg_min"),
                pl.col("v10m_avg").mean().alias("v10m_avg_avg"),
                pl.col("v10m_avg").max().alias("v10m_avg_max"),
                pl.col("vinst_max").min().alias("vinst_max_min"),
                pl.col("vinst_max").mean().alias("vinst_max_avg"),
                pl.col("vinst_max").max().alias("vinst_max_max"),
                pl.col("los_or_ov1_flag").max().cast(pl.Int8).alias("los_or_ov1_flag"),
            ]
        )
        .sort(["site_id", "event_day", "bucket_5min_local"])
    )


def main() -> None:
    """Build timestamp and 5-minute curtailment extracts for assessed tier-based sites."""
    assessed_sites = _load_assessed_sites()

    # Load the circuit metadata and cleaned metrology needed for site rebuilds.
    if not CIRCUIT_DETAILS_PATH.exists():
        raise FileNotFoundError(
            f"Missing required input: {CIRCUIT_DETAILS_PATH}. Run conformance first "
            "to generate the site summary."
        )
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing cleaned site data at {CLEANED_DATA_PATH}. "
            "Run run_sapn2022_preprocessing.py first."
        )
    circuit_details = pl.read_csv(CIRCUIT_DETAILS_PATH)
    all_data = load_sapn_cleaned_data(CLEANED_DATA_PATH)

    timestamp_frames: list[pl.DataFrame] = []
    event_timezone = ZoneInfo(LOCAL_TIMEZONE)

    for idx, site_row in enumerate(assessed_sites.iter_rows(named=True), start=1):
        site_id = int(site_row["site_id"])
        los_threshold_used = float(site_row["los_threshold_used"])
        ov1_threshold_used = float(site_row["ov1_threshold_used"])
        site_data = select_site_pv_data(all_data, circuit_details, site_id)

        site_day_frames: list[pl.DataFrame] = []
        for event_day in EVENT_DAYS:
            start_day = datetime(
                EVENT_YEAR,
                EVENT_MONTH,
                event_day,
                DAY_EXTRACTION_START.hour,
                DAY_EXTRACTION_START.minute,
                DAY_EXTRACTION_START.second,
                tzinfo=event_timezone,
            )
            end_day = datetime(
                EVENT_YEAR,
                EVENT_MONTH,
                event_day,
                DAY_END.hour,
                DAY_END.minute,
                DAY_END.second,
                tzinfo=event_timezone,
            )
            site_day_long = extract_site_day(
                site_data,
                start_day,
                end_day,
            )
            if site_day_long.is_empty():
                continue
            wide = map_circuit_data_to_site(site_day_long, site_id)

            # Unlike ``main.py``, this script intentionally keeps any assessed
            # site-day that has raw mapped data. It does not remove days for
            # low common power/v10m coverage before building the output tables.
            day_frame = _build_day_flag_frame(
                wide,
                los_threshold_used=los_threshold_used,
                ov1_threshold_used=ov1_threshold_used,
            )
            if not day_frame.is_empty():
                site_day_frames.append(day_frame)

        if site_day_frames:
            timestamp_frames.append(pl.concat(site_day_frames, how="vertical"))

        if idx % 25 == 0 or idx == assessed_sites.height:
            print(f"[{idx}/{assessed_sites.height}] processed site {site_id}")

    timestamp_df = (
        pl.concat(timestamp_frames, how="vertical")
        if timestamp_frames
        else pl.DataFrame(schema=TIMESTAMP_OUTPUT_SCHEMA)
    ).sort(["site_id", "event_day", "local_tstamp"])
    bucket_df = _build_bucket_frame(timestamp_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_df.write_csv(TIMESTAMP_OUTPUT_PATH)
    bucket_df.write_csv(BUCKET_OUTPUT_PATH)

    print(f"Assessed tier-based sites: {assessed_sites.height}")
    print(f"Timestamp rows written: {timestamp_df.height}")
    print(f"5-minute bucket rows written: {bucket_df.height}")
    print(f"Saved timestamp flags to {TIMESTAMP_OUTPUT_PATH}")
    print(f"Saved 5-minute buckets to {BUCKET_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
