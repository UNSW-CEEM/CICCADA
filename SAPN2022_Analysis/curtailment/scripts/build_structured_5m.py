"""
Build script 1b for the local SAPN2022 curtailment workflow.

This creates a separate 5-minute canonical structured dataset that stays
closer to the original Trino structured-data workflow than the
high-frequency builder, while now sharing the same SAPN2022_updated
exact-timestamp site-power construction first.

Key choices in this 5-minute builder:
- raw EVM metrology and cleaned SAPN validation metrology are read at native
  timestamp grain first;
- site rows are then assigned to SAPN end-labeled 5-minute bins and collapsed
  to one row per site/dataset_role/5-minute bin using mean P_kw and mean V;
- Adelaide local time is still used for day/time-of-day features;
- clear-sky candidate days must be observed train days for that site;
- the previous 45-day clear-sky distance cutoff is removed;
- reactive power is unavailable locally, so Q_kvar_norm is omitted and S_norm
  is fixed at 1.0;
- resolved local capacity is still written as both ac_capacity_kw and S_99 so
  downstream logic can keep using the Trino-style denominator name;
- the cleaned SAPN validation parquet should already include power conversion,
  local timestamps, `voltage_valid`, and polarity, and this script fails fast
  if that file is missing.
"""

from datetime import date
from pathlib import Path

import polars as pl

import site_metrology_helpers as sapn_funcs
from path_config import require_local_path
from structured_data_shared_params import (
    BOM_CLEAR_SKY_CANDIDATES,
    EXCLUDED_LOCAL_DAYS,
    MIN_VALID_CLEAR_SKY_P_KW_NORM,
    S_NORM_REACTIVE_POWER_UNAVAILABLE,
    add_capacity_and_grid,
    adelaide_datetime_expr,
    as_lazy,
    bom10_to_5min,
    bom_daily_parquets,
    build_eligible_sites,
    clear_sky_days,
    end_labeled_5min_utc_expr,
    evm_training_parquets,
    map_sites_to_bom_grid,
    prepare_bom10min,
    pv_circuits_for_sites,
    read_evm_circuit_metadata,
    read_evm_site_metadata,
    read_sapn_circuit_details,
    read_sapn_site_details,
    read_site_cohort,
    resolve_capacity,
    time_of_day_5min_expr,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SAPN/EVM/BOM inputs live outside this repo, so their machine-specific root
# folders are defined in the ignored `local_paths.py` file instead of here.
SAPN_ROOT = require_local_path(
    "SAPN_ROOT",
    "root folder containing `Nov2022/`, `All Results/`, and `updated results/`.",
)
EVM_ROOT = require_local_path(
    "EVM_ROOT",
    "root folder containing `site_metadata.csv`, `circuit_metadata.csv`, and the EVM training parquet directory.",
)
BOM_ROOT = require_local_path(
    "BOM_ROOT",
    "root folder containing the BOM daily parquet files.",
)
BOM_POINTS_CSV = require_local_path(
    "BOM_POINTS_CSV",
    "CSV mapping BOM postcodes to point locations.",
)

# The cleaned SAPN validation parquet should already exist locally before this
# build runs; these derived paths intentionally stay machine-local.
SAPN_SITE_DETAILS_PATH = SAPN_ROOT / "Nov2022" / "ebm_1_20221112_20221119_site_details.csv"
SAPN_CIRCUIT_DETAILS_PATH = SAPN_ROOT / "Nov2022" / "ebm_1_20221112_20221119_circuit_details.csv"
SAPN_CLEANED_DATA_PATH = SAPN_ROOT / "Nov2022" / "ebm_1_20221112_20221119_data_cleaned_sa.parquet"
EVM_SITE_METADATA_PATH = EVM_ROOT / "site_metadata.csv"
EVM_CIRCUIT_METADATA_PATH = EVM_ROOT / "circuit_metadata.csv"
EVM_TRAINING_DIR = EVM_ROOT / "curtailment training data parquet"
SITE_COHORT_CSV = PROJECT_ROOT / "confidence_tier_site_ids.csv"
USE_SITE_COHORT = True
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "all_structured_data_5m"
OUTPUT_NAME = "structured_data_5m.parquet"
TRAIN_START_DATE = date(2022, 10, 12)
TRAIN_END_DATE = date(2022, 11, 11)
VALIDATION_START_DATE = date(2022, 11, 12)
VALIDATION_END_DATE = date(2022, 11, 19)
BOM_START_DATE = None
BOM_END_DATE = None
LIMIT_SITES = None
SITE_ID = None

STRUCTURED_5M_COLUMNS = [
    "site_id",
    "t_stamp",
    "actual_day",
    "actual_tod",
    "V",
    "P_kw_norm",
    "S_norm",
    "GHI",
    "cloud_type",
    "cs_day",
    "cs_tod",
    "P_kw_norm_cs",
    "GHI_cs",
    "cloud_type_cs",
    "S_99",
    "year",
    "month",
    "dataset_role",
    "ac_capacity_kw",
    "n_lat",
    "n_long",
]


def canonicalise_site_metrology(site_metrology):
    """Collapse native site rows to one canonical row per 5-minute timestamp."""
    return (
        as_lazy(site_metrology)
        .with_columns(end_labeled_5min_utc_expr("t_stamp").alias("t_stamp_5m"))
        .group_by(["dataset_role", "site_id", "t_stamp_5m"])
        .agg([
            pl.col("P_kw").mean().alias("P_kw"),
            pl.col("V").mean().alias("V"),
            pl.col("ac_capacity_kw").first().alias("ac_capacity_kw"),
            pl.col("S_99").first().alias("S_99"),
            pl.col("capacity_source").first().alias("capacity_source"),
            pl.col("n_lat").first().alias("n_lat"),
            pl.col("n_long").first().alias("n_long"),
        ])
        .rename({"t_stamp_5m": "t_stamp"})
        .with_columns([
            adelaide_datetime_expr("t_stamp").dt.date().alias("actual_day"),
            time_of_day_5min_expr("t_stamp").alias("actual_tod"),
            (pl.col("P_kw") / pl.col("S_99")).alias("P_kw_norm"),
            pl.lit(S_NORM_REACTIVE_POWER_UNAVAILABLE).cast(pl.Float64).alias("S_norm"),
        ])
        .filter(~pl.col("actual_day").is_in(sorted(EXCLUDED_LOCAL_DAYS)))
    )


def observed_train_clear_sky_days(site_metrology, bom10min):
    """Keep only BOM clear-sky days that were actually observed for each train site."""
    observed_train_days = (
        as_lazy(site_metrology)
        .filter(pl.col("dataset_role") == "train")
        .select(["site_id", "n_lat", "n_long", pl.col("actual_day").alias("clear_sky_day")])
        .unique()
    )

    return (
        observed_train_days
        .join(
            as_lazy(clear_sky_days(bom10min)),
            left_on=["n_lat", "n_long", "clear_sky_day"],
            right_on=["latitude", "longitude", "clear_sky_day"],
            how="inner",
        )
        .select([
            "site_id",
            "n_lat",
            "n_long",
            "clear_sky_day",
            "cloud_sum",
            "max_GHI",
            "rn",
        ])
        .unique()
    )


def nearest_clear_sky_candidates(site_metrology, bom10min):
    """Rank the nearest observed-train clear-sky candidates for each actual day."""
    site_days = (
        as_lazy(site_metrology)
        .select(["dataset_role", "site_id", "n_lat", "n_long", "actual_day"])
        .unique()
    )
    cs_days = observed_train_clear_sky_days(site_metrology, bom10min)

    return (
        site_days
        .join(cs_days, on=["site_id", "n_lat", "n_long"], how="inner")
        .with_columns([
            (pl.col("actual_day").cast(pl.Int32) - pl.col("clear_sky_day").cast(pl.Int32)).alias("signed_day_diff"),
            (pl.col("actual_day").cast(pl.Int32) - pl.col("clear_sky_day").cast(pl.Int32)).abs().alias("abs_day_diff"),
        ])
        .sort(
            ["dataset_role", "site_id", "actual_day", "abs_day_diff", "signed_day_diff", "clear_sky_day"]
        )
        .with_columns(
            pl.col("clear_sky_day")
            .cum_count()
            .over(["dataset_role", "site_id", "actual_day"])
            .alias("candidate_rank")
        )
        .filter(pl.col("candidate_rank") <= BOM_CLEAR_SKY_CANDIDATES)
        .select([
            "dataset_role",
            "site_id",
            "n_lat",
            "n_long",
            "actual_day",
            "clear_sky_day",
            "cloud_sum",
            "max_GHI",
            "rn",
            "abs_day_diff",
            "signed_day_diff",
            "candidate_rank",
        ])
    )


def segmented_site_metrology(site_metrology):
    """Split canonical 5-minute train days into continuous segments."""
    return (
        as_lazy(site_metrology)
        .sort(["site_id", "actual_day", "t_stamp"])
        .with_columns(
            pl.col("t_stamp").shift(1).over(["site_id", "actual_day"]).alias("prev_ts")
        )
        .with_columns(
            pl.when(pl.col("prev_ts").is_null())
            .then(pl.lit(0))
            .when((pl.col("t_stamp") - pl.col("prev_ts")).dt.total_minutes() > 30)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("gap_start")
        )
        .with_columns(
            pl.col("gap_start").cum_sum().over(["site_id", "actual_day"]).alias("segment_id")
        )
    )


def nearest_clear_sky_profiles(site_metrology, clear_sky_candidates, bom5min):
    """Build 5-minute clear-sky PV/GHI profiles from observed train clear-sky days."""
    candidate_cs_days = (
        as_lazy(clear_sky_candidates)
        .select(["site_id", "n_lat", "n_long", pl.col("clear_sky_day").alias("cs_day")])
        .unique()
    )

    return (
        segmented_site_metrology(
            as_lazy(site_metrology).filter(pl.col("dataset_role") == "train")
        )
        .join(
            candidate_cs_days,
            left_on=["site_id", "n_lat", "n_long", "actual_day"],
            right_on=["site_id", "n_lat", "n_long", "cs_day"],
            how="inner",
        )
        .with_columns([
            pl.col("actual_day").alias("cs_day"),
            pl.col("actual_tod").alias("cs_tod"),
        ])
        .join(
            as_lazy(bom5min),
            left_on=["n_lat", "n_long", "t_stamp"],
            right_on=["latitude", "longitude", "time_5min"],
            how="inner",
        )
        .group_by(["site_id", "cs_day", "segment_id", "cs_tod"])
        .agg([
            pl.col("P_kw_norm").first().alias("P_kw_norm_bin"),
            pl.col("GHI").first().alias("GHI_bin"),
            pl.col("cloud_type").first().alias("cloud_type_cs"),
        ])
        .sort(["site_id", "cs_day", "segment_id", "cs_tod"])
        .with_columns([
            pl.col("P_kw_norm_bin")
            .rolling_quantile(0.6, window_size=7, min_samples=1, center=True)
            .over(["site_id", "cs_day", "segment_id"])
            .alias("P_kw_norm_cs"),
            pl.col("GHI_bin")
            .rolling_quantile(0.6, window_size=7, min_samples=1, center=True)
            .over(["site_id", "cs_day", "segment_id"])
            .alias("GHI_cs"),
        ])
        .select(["site_id", "cs_day", "cs_tod", "P_kw_norm_cs", "GHI_cs", "cloud_type_cs"])
    )


def attach_clear_sky_features(site_metrology, bom10min):
    """Join canonical 5-minute rows to BOM and the nearest observed-train clear-sky profile."""
    bin_key_cols = ["dataset_role", "site_id", "actual_day", "actual_tod"]
    bom5min = bom10_to_5min(bom10min)
    clear_sky_candidates = nearest_clear_sky_candidates(site_metrology, bom10min)
    cs_profiles = nearest_clear_sky_profiles(site_metrology, clear_sky_candidates, bom5min)

    actual_with_bom = (
        as_lazy(site_metrology)
        .join(
            as_lazy(bom5min),
            left_on=["n_lat", "n_long", "t_stamp"],
            right_on=["latitude", "longitude", "time_5min"],
            how="inner",
        )
    )

    actual_bin_candidates = (
        actual_with_bom
        .select(["dataset_role", "site_id", "n_lat", "n_long", "actual_day", "actual_tod"])
        .unique()
        .join(
            clear_sky_candidates,
            on=["dataset_role", "site_id", "n_lat", "n_long", "actual_day"],
            how="left",
        )
        .join(
            cs_profiles,
            left_on=["site_id", "clear_sky_day", "actual_tod"],
            right_on=["site_id", "cs_day", "cs_tod"],
            how="left",
        )
        .with_columns([
            (pl.col("P_kw_norm_cs") > MIN_VALID_CLEAR_SKY_P_KW_NORM)
            .fill_null(False)
            .alias("candidate_valid"),
        ])
        .with_columns(
            pl.when(pl.col("candidate_valid"))
            .then(pl.col("candidate_rank"))
            .otherwise(None)
            .min()
            .over(bin_key_cols)
            .alias("selected_candidate_rank")
        )
    )

    primary_bin_profiles = (
        actual_bin_candidates
        .group_by(bin_key_cols)
        .agg([
            pl.col("clear_sky_day").filter(pl.col("candidate_rank") == 1).first().alias("primary_clear_sky_day"),
            pl.col("GHI_cs").filter(pl.col("candidate_rank") == 1).first().alias("primary_GHI_cs"),
            pl.col("cloud_type_cs").filter(pl.col("candidate_rank") == 1).first().alias("primary_cloud_type_cs"),
            pl.col("selected_candidate_rank").first().alias("selected_candidate_rank"),
        ])
    )

    selected_bin_profiles = (
        actual_bin_candidates
        .filter(pl.col("candidate_rank") == pl.col("selected_candidate_rank"))
        .group_by(bin_key_cols)
        .agg([
            pl.col("clear_sky_day").first().alias("selected_clear_sky_day"),
            pl.col("P_kw_norm_cs").first().alias("selected_P_kw_norm_cs"),
            pl.col("GHI_cs").first().alias("selected_GHI_cs"),
            pl.col("cloud_type_cs").first().alias("selected_cloud_type_cs"),
        ])
    )

    attached_clear_sky_bins = (
        primary_bin_profiles
        .join(selected_bin_profiles, on=bin_key_cols, how="left")
        .with_columns([
            pl.coalesce(["selected_clear_sky_day", "primary_clear_sky_day"]).alias("cs_day"),
            pl.col("actual_tod").alias("cs_tod"),
            pl.col("selected_P_kw_norm_cs").alias("P_kw_norm_cs"),
            pl.coalesce(["selected_GHI_cs", "primary_GHI_cs"]).alias("GHI_cs"),
            pl.coalesce(["selected_cloud_type_cs", "primary_cloud_type_cs"]).alias("cloud_type_cs"),
        ])
    )

    return (
        actual_with_bom
        .join(attached_clear_sky_bins, on=bin_key_cols, how="left")
    )


def run(
    *,
    output_dir=OUTPUT_DIR,
    sapn_validation_data_path=SAPN_CLEANED_DATA_PATH,
    site_id=SITE_ID,
    limit_sites=LIMIT_SITES,
):
    """Run the separate 5-minute canonical structured-data build with a few explicit overrides."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sapn_validation_data_path = Path(sapn_validation_data_path)
    if not sapn_validation_data_path.exists():
        raise FileNotFoundError(
            f"SAPN validation data file not found: {sapn_validation_data_path}. "
            "Run data_processing.py to create the cleaned parquet or pass another path to run(...)."
        )
    train_start = TRAIN_START_DATE
    train_end = TRAIN_END_DATE
    validation_start = VALIDATION_START_DATE
    validation_end = VALIDATION_END_DATE
    bom_start = BOM_START_DATE if BOM_START_DATE else min(train_start, validation_start)
    bom_end = BOM_END_DATE if BOM_END_DATE else max(train_end, validation_end)

    print("Loading metadata")
    sapn_sites = read_sapn_site_details(SAPN_SITE_DETAILS_PATH)
    sapn_circuits = read_sapn_circuit_details(SAPN_CIRCUIT_DETAILS_PATH)
    evm_sites = read_evm_site_metadata(EVM_SITE_METADATA_PATH)
    evm_circuits = read_evm_circuit_metadata(EVM_CIRCUIT_METADATA_PATH)

    site_cohort = None
    if USE_SITE_COHORT:
        if not SITE_COHORT_CSV.exists():
            raise FileNotFoundError(f"Site cohort CSV not found: {SITE_COHORT_CSV}")
        site_cohort = read_site_cohort(SITE_COHORT_CSV)
        print(f"Using site cohort: {SITE_COHORT_CSV} ({site_cohort.height} sites)")

    eligible_sites, _ = build_eligible_sites(
        sapn_sites,
        sapn_circuits,
        evm_sites,
        site_cohort=site_cohort,
        limit_sites=None if SITE_ID is not None else LIMIT_SITES,
    )
    if SITE_ID is not None:
        eligible_sites = eligible_sites.filter(pl.col("site_id") == SITE_ID)
        if eligible_sites.is_empty():
            raise ValueError(f"Requested site_id is not eligible or not in the selected cohort: {SITE_ID}")
    print(f"Eligible sites: {eligible_sites.height}")

    sapn_pv_circuits = pv_circuits_for_sites(sapn_circuits, eligible_sites)
    evm_pv_circuits = pv_circuits_for_sites(evm_circuits, eligible_sites)
    bom_mapping = map_sites_to_bom_grid(eligible_sites, BOM_POINTS_CSV)

    evm_parquets = evm_training_parquets(EVM_TRAINING_DIR, train_start, train_end)
    bom_files = bom_daily_parquets(BOM_ROOT, bom_start, bom_end)
    print(f"EVM training parquet files: {len(evm_parquets)}")
    print(f"BOM files: {len(bom_files)}")

    print("Preparing native metrology")
    site_ids = eligible_sites["site_id"].to_list()
    train_raw = sapn_funcs.prepare_evm_metrology(
        evm_parquets,
        evm_pv_circuits,
        train_start,
        train_end,
    )
    validation_raw = sapn_funcs.prepare_sapn_metrology(
        sapn_validation_data_path,
        validation_start,
        validation_end,
    )
    train_site_metrology = sapn_funcs.aggregate_site_metrology(
        train_raw,
        evm_pv_circuits,
        site_ids,
        train_start,
        train_end,
        dataset_role="train",
    )
    validation_site_metrology = sapn_funcs.aggregate_site_metrology(
        validation_raw,
        sapn_pv_circuits,
        site_ids,
        validation_start,
        validation_end,
        dataset_role="validation",
    )
    native_site_metrology = (
        pl.concat([train_site_metrology, validation_site_metrology], how="vertical")
        .lazy()
        .with_columns([
            adelaide_datetime_expr("t_stamp").dt.date().alias("actual_day"),
            time_of_day_5min_expr("t_stamp").alias("actual_tod"),
        ])
        .filter(~pl.col("actual_day").is_in(sorted(EXCLUDED_LOCAL_DAYS)))
        .collect()
    )

    print("Resolving capacity")
    capacity = resolve_capacity(eligible_sites, native_site_metrology.lazy()).collect()
    native_with_capacity = add_capacity_and_grid(
        native_site_metrology.lazy(),
        capacity,
        bom_mapping,
    ).collect()

    print("Canonicalising to 5-minute site rows")
    site_metrology_5m = canonicalise_site_metrology(native_with_capacity).collect()

    print("Preparing BOM and clear-sky features")
    bom10min = prepare_bom10min(bom_files, bom_mapping).collect()
    structured = attach_clear_sky_features(site_metrology_5m, bom10min)

    structured_df = (
        structured
        .with_columns([
            pl.col("t_stamp").dt.year().alias("year"),
            pl.col("t_stamp").dt.month().alias("month"),
        ])
        .select(STRUCTURED_5M_COLUMNS)
        .sort(["dataset_role", "site_id", "t_stamp"])
        .collect()
    )

    output_path = output_dir / OUTPUT_NAME
    print(f"Writing {output_path}")
    structured_df.write_parquet(output_path, compression="zstd")
    print("Done")

def main():
    """Run the separate 5-minute canonical structured-data build."""
    return run(
        output_dir=OUTPUT_DIR,
        sapn_validation_data_path=SAPN_CLEANED_DATA_PATH,
        # site_id=SITE_ID,
        # limit_sites=LIMIT_SITES,
    )

if __name__ == "__main__":
    main()
