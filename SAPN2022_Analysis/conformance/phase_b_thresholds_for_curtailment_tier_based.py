from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from checkPVBehaviour import CheckPVBehaviour
from funcs import (
    CLEANED_SITE_DATA_PATH,
    loadCleanedSiteData,
    mapCircuitDataToSite,
)


PHASE_B_SUMMARY_PATH = Path("updated results/site_compliance/phase_b_site_summary.csv")
SITE_THRESHOLDS_PATH = Path("updated results/site_compliance/site_thresholds.csv")
CIRCUIT_DETAILS_PATH = Path("Nov2022/ebm_1_20221112_20221119_circuit_details.csv")
CLEANED_DATA_PATH = CLEANED_SITE_DATA_PATH
OUTPUT_DIR = Path("updated results/phase b info for curtailment/tier based")
TIMESTAMP_OUTPUT_PATH = OUTPUT_DIR / "tier_based_timestamp_flags.csv"
BUCKET_OUTPUT_PATH = OUTPUT_DIR / "tier_based_5min_buckets.csv"
DAYS_TO_CHECK = (13, 14, 15, 16, 17, 19)
LOCAL_TZ = ZoneInfo("Australia/Adelaide")
TAU = 0.3


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}. Run main.py first to generate the Phase B summary and thresholds."
        )


def _empty_timestamp_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "site_id": pl.Int64,
            "event_day": pl.Int64,
            "local_tstamp": pl.Datetime(time_zone="Australia/Adelaide"),
            "utc_tstamp": pl.Utf8,
            "site_power_kw": pl.Float64,
            "v10m_avg": pl.Float64,
            "vinst_max": pl.Float64,
            "los_or_ov1_flag": pl.Int8,
        }
    )


def _empty_bucket_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
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
    )


def _load_assessed_sites() -> pl.DataFrame:
    _require_file(PHASE_B_SUMMARY_PATH)
    _require_file(SITE_THRESHOLDS_PATH)

    summary_df = pl.read_csv(PHASE_B_SUMMARY_PATH, null_values=[""])
    thresholds_df = pl.read_csv(SITE_THRESHOLDS_PATH, null_values=[""])

    assessed_summary = summary_df.filter(pl.col("overall_pass").is_not_null())
    assessed_sites = (
        assessed_summary
        .select(["site_id", "los_threshold_used"])
        .join(
            thresholds_df.select(["site_id", "ov1_work_site"]),
            on="site_id",
            how="left",
        )
        .with_columns([
            pl.col("site_id").cast(pl.Int64),
            pl.col("los_threshold_used").cast(pl.Float64),
            pl.col("ov1_work_site").cast(pl.Float64),
        ])
        .sort("site_id")
    )

    if assessed_sites.height != assessed_summary.height:
        raise ValueError(
            "Tier-based assessed site rows do not line up with threshold rows. "
            "Re-run main.py so the summary and thresholds match."
        )

    missing_thresholds = assessed_sites.filter(
        pl.col("los_threshold_used").is_null() | pl.col("ov1_work_site").is_null()
    )
    if not missing_thresholds.is_empty():
        bad_sites = missing_thresholds["site_id"].to_list()
        raise ValueError(
            f"Missing LOS/OV1 threshold values for assessed sites: {bad_sites[:10]}"
        )

    return assessed_sites


def _prepare_inputs() -> tuple[pl.DataFrame, pl.LazyFrame]:
    _require_file(CIRCUIT_DETAILS_PATH)
    if not Path(f"{CLEANED_DATA_PATH}.parquet").exists():
        raise FileNotFoundError(
            f"Missing cleaned site data at {CLEANED_DATA_PATH}.parquet. Run data_processing.py first."
        )

    circuit_details = pl.read_csv(CIRCUIT_DETAILS_PATH)
    all_data = loadCleanedSiteData(CLEANED_DATA_PATH)

    return circuit_details, all_data


def _window_for_day(day: int) -> tuple[datetime, datetime]:
    start_day = datetime(2022, 11, day, 6, 0, 0, tzinfo=LOCAL_TZ)
    end_day = datetime(2022, 11, day, 18, 0, 0, tzinfo=LOCAL_TZ)
    return start_day, end_day


def _build_day_flag_frame(
    wide: pl.DataFrame,
    *,
    los_threshold_used: float,
    ov1_work_threshold: float,
) -> pl.DataFrame:
    behaviour = CheckPVBehaviour(wide, volCol="voltage_valid")
    df = behaviour.circuitData.clone()

    power_cols = [
        col for col in df.columns
        if col.startswith("power")
        and not col.endswith("_next")
        and not col.endswith("_logic")
    ]
    voltage_cols = [col for col in df.columns if col.startswith("voltage_valid")]

    if not power_cols or not voltage_cols:
        return _empty_timestamp_frame()

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
        return _empty_timestamp_frame()

    df = df.with_columns([
        pl.mean_horizontal([pl.col(col) for col in vmean_cols]).alias("v10m_avg"),
        pl.max_horizontal([pl.col(col) for col in voltage_cols]).alias("vinst_max"),
        pl.sum_horizontal([
            pl.when(pl.col(col).cast(pl.Float64, strict=False).fill_null(0) < 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col(col).cast(pl.Float64, strict=False).fill_null(0))
            for col in power_cols
        ]).alias("site_power_kw"),
        pl.any_horizontal([pl.col(col).is_not_null() for col in power_cols]).alias("_has_power"),
    ])

    df = df.with_columns([
        (pl.col("_has_power") & pl.col("v10m_avg").is_not_null()).alias("_keep_row"),
        (pl.col("vinst_max") >= (ov1_work_threshold - TAU)).fill_null(False).alias("_ov1_kicks_in"),
    ])

    df = df.with_columns([
        (pl.col("_keep_row") & pl.col("_ov1_kicks_in")).alias("_ov1_responsible"),
        (
            pl.col("_keep_row")
            & ~pl.col("_ov1_kicks_in")
            & (pl.col("v10m_avg") > los_threshold_used)
        ).fill_null(False).alias("_los_responsible"),
    ])

    return (
        df
        .filter(pl.col("_keep_row"))
        .select([
            pl.col("site_id").cast(pl.Int64),
            pl.col("local_tstamp").cast(pl.Datetime(time_zone="Australia/Adelaide")),
            pl.col("utc_tstamp").cast(pl.Utf8),
            pl.col("local_tstamp").dt.day().cast(pl.Int64).alias("event_day"),
            pl.col("site_power_kw").cast(pl.Float64),
            pl.col("v10m_avg").cast(pl.Float64),
            pl.col("vinst_max").cast(pl.Float64),
            (pl.col("_ov1_responsible") | pl.col("_los_responsible"))
            .cast(pl.Int8)
            .alias("los_or_ov1_flag"),
        ])
        .sort("local_tstamp")
    )


def _build_bucket_frame(timestamp_df: pl.DataFrame) -> pl.DataFrame:
    if timestamp_df.is_empty():
        return _empty_bucket_frame()

    return (
        timestamp_df
        .with_columns(
            pl.col("local_tstamp").dt.truncate("5m").alias("_bucket_floor")
        )
        .with_columns(
            pl.when(pl.col("local_tstamp") == pl.col("_bucket_floor"))
            .then(pl.col("_bucket_floor"))
            .otherwise(pl.col("_bucket_floor") + pl.duration(minutes=5))
            .alias("bucket_5min_local")
        )
        .with_columns([
            pl.col("bucket_5min_local").dt.strftime("%H:%M:%S").alias("tod_bin"),
            pl.col("bucket_5min_local").dt.strftime("%Y-%m-%d %H:%M:%S%z").alias("_bucket_text_raw"),
        ])
        .with_columns(
            pl.concat_str([
                pl.col("_bucket_text_raw").str.slice(0, 22),
                pl.lit(":"),
                pl.col("_bucket_text_raw").str.slice(22, 2),
            ]).alias("bucket_5min_local")
        )
        .group_by(["site_id", "event_day", "bucket_5min_local"])
        .agg([
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
        ])
        .sort(["site_id", "event_day", "bucket_5min_local"])
    )


def main() -> None:
    assessed_sites = _load_assessed_sites()
    circuit_details, all_data = _prepare_inputs()

    timestamp_frames: list[pl.DataFrame] = []

    for idx, row in enumerate(assessed_sites.iter_rows(named=True), start=1):
        site_id = int(row["site_id"])
        los_threshold_used = float(row["los_threshold_used"])
        ov1_work_threshold = float(row["ov1_work_site"])

        site_day_frames: list[pl.DataFrame] = []
        for day in DAYS_TO_CHECK:
            start_day, end_day = _window_for_day(day)
            has_data, wide, _ = mapCircuitDataToSite(
                all_data,
                circuit_details,
                site_id,
                start_day,
                end_day,
            )
            if not has_data:
                continue

            day_frame = _build_day_flag_frame(
                wide,
                los_threshold_used=los_threshold_used,
                ov1_work_threshold=ov1_work_threshold,
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
        else _empty_timestamp_frame()
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
