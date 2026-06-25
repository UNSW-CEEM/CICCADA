"""
Build script 1 for the local SAPN2022 curtailment workflow.

This creates the structured dataset used by the later GHI-normalised model
scripts. It follows the original Solar Analytics structured-data workflow, but
reads local EVM/SAPN/BOM files with Polars instead of querying Trino.
EVM training metrology is expected to have already been converted from the
original CSVs into parquet outside this workflow, without filtering,
aggregation, timestamp rounding, or schema changes.

Important choices kept explicit in this script:
- metrology and BOM are joined on exact UTC timestamps;
- Adelaide local time is used only for day/time-of-day features, matching the
  SAPN2022 local-data handling rather than the original fixed +10 hour shortcut;
- sites are restricted to the explicit confidence-tier assessed cohort CSV,
  because models are per-site and unused site models add work only;
- EVM rows are labelled dataset_role="train";
- SAPN2022 validation rows are read from the cleaned parquet produced by
  `data_processing.py`, so power conversion, local timestamps, `voltage_valid`,
  and polarity have already been applied before this script runs;
- the script fails fast if that cleaned SAPN validation parquet is missing;
- local files do not include reactive power, so S_norm is set to 1.0.
"""

from datetime import date, time, timedelta
from pathlib import Path

import polars as pl

import funcs_from_SAPN2022updated as sapn_funcs
from path_config import require_local_path
from structured_data_shared_params import (
    BOM_CLEAR_SKY_CANDIDATES,
    EXCLUDED_LOCAL_DAYS,
    MIN_VALID_CLEAR_SKY_P_KW_NORM,
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

# SAPN/EVM/BOM inputs live outside this repo, so their machine-specific paths
# are loaded from the ignored `local_paths.py` file instead of being committed.
DEFAULT_SAPN_ROOT = require_local_path(
    "SAPN_ROOT",
    "root folder containing `Nov2022/`, `All Results/`, and `updated results/`.",
)
DEFAULT_EVM_ROOT = require_local_path(
    "EVM_ROOT",
    "root folder containing `site_metadata.csv`, `circuit_metadata.csv`, and the EVM training parquet directory.",
)
DEFAULT_BOM_ROOT = require_local_path(
    "BOM_ROOT",
    "root folder containing the BOM daily parquet files.",
)
DEFAULT_BOM_POINTS = require_local_path(
    "BOM_POINTS_CSV",
    "CSV mapping BOM postcodes to point locations.",
)

# Fixed local copy of the SAPN2022 confidence-tier assessed cohort.
DEFAULT_SITE_COHORT = PROJECT_ROOT / "confidence_tier_site_ids.csv"

# Edit this block when changing local inputs or the build window.
# `SAPN_CLEANED_DATA_PATH` should point at the cleaned validation parquet
# created by `data_processing.py`. This build reads that file directly and
# raises an error immediately if it is missing.
SAPN_SITE_DETAILS_PATH = DEFAULT_SAPN_ROOT / "Nov2022" / "ebm_1_20221112_20221119_site_details.csv"
SAPN_CIRCUIT_DETAILS_PATH = DEFAULT_SAPN_ROOT / "Nov2022" / "ebm_1_20221112_20221119_circuit_details.csv"
SAPN_CLEANED_DATA_PATH = DEFAULT_SAPN_ROOT / "Nov2022" / "ebm_1_20221112_20221119_data_cleaned_sa.parquet"
EVM_SITE_METADATA_PATH = DEFAULT_EVM_ROOT / "site_metadata.csv"
EVM_CIRCUIT_METADATA_PATH = DEFAULT_EVM_ROOT / "circuit_metadata.csv"
EVM_TRAINING_DIR = DEFAULT_EVM_ROOT / "curtailment training data parquet"
BOM_ROOT = DEFAULT_BOM_ROOT
BOM_POINTS_CSV = DEFAULT_BOM_POINTS
SITE_COHORT_CSV = DEFAULT_SITE_COHORT
USE_SITE_COHORT = True
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "all_structured_data_test"
TRAIN_START_DATE = date(2022, 10, 12)
TRAIN_END_DATE = date(2022, 11, 11)
VALIDATION_START_DATE = date(2022, 11, 12)
VALIDATION_END_DATE = date(2022, 11, 19)
BOM_START_DATE = None
BOM_END_DATE = None
LIMIT_SITES = None
SITE_ID = None
WRITE_DIAGNOSTICS = False

CLEAR_SKY_MAX_DISTANCE_DAYS = 45
# Only the high-resolution builder uses the nearest-5-minute BOM join with a
# tolerance window, so this stays local instead of moving to the shared module.
BOM_JOIN_TOLERANCE_MINUTES = 5
DIAGNOSTIC_DAYLIGHT_START = time(6, 0)
DIAGNOSTIC_DAYLIGHT_END = time(18, 0)

# Final structured-data schema. Later scripts should depend on these columns,
# not on the raw SAPN/EVM/BOM input schemas.
STRUCTURED_COLUMNS = [
    "site_id",
    "t_stamp",
    "actual_day",
    "actual_tod",
    "P_kw",
    "V",
    "GHI",
    "cloud_type",
    "Q_kvar_norm",
    "P_kw_norm",
    "S_norm",
    "P_kw_norm_cs",
    "GHI_cs",
    "S_99",
    "ac_capacity_kw",
    "n_lat",
    "n_long",
    "dataset_role",
]

def add_bom_join_timestamp(site_metrology):
    """Add the temporary timestamp used only for BOM joins.

    The source metrology timestamp is kept unchanged as t_stamp. BOM data is on
    a 5-minute grid after bom10_to_5min(), so local rows are joined to the
    nearest 5-minute BOM timestamp when it is within the configured tolerance.
    """
    return (
        site_metrology
        .with_columns(end_labeled_5min_utc_expr("t_stamp").alias("bom_join_ts"))
        .with_columns(
            (pl.col("t_stamp") - pl.col("bom_join_ts"))
            .dt.total_seconds()
            .abs()
            .alias("bom_join_delta_seconds")
        )
        .filter(pl.col("bom_join_delta_seconds") <= BOM_JOIN_TOLERANCE_MINUTES * 60)
    )


def train_window_bounds(site_metrology):
    """Return the calendar window covered by the training cohort."""
    train_bounds = (
        as_lazy(site_metrology)
        .filter(pl.col("dataset_role") == "train")
        .select([
            pl.col("actual_day").min().alias("train_start_day"),
            pl.col("actual_day").max().alias("train_end_day"),
        ])
        .collect()
    )
    train_start_day, train_end_day = train_bounds.row(0)
    if train_start_day is None or train_end_day is None:
        raise ValueError("No train rows available to define the clear-sky search window")
    return train_start_day, train_end_day


def nearest_clear_sky_candidates(site_metrology, bom10min):
    """Rank the nearest train-window BOM clear-sky candidates for each actual day."""
    train_start_day, train_end_day = train_window_bounds(site_metrology)
    site_days = site_metrology.select(["n_lat", "n_long", "actual_day"]).unique()
    cs_days = clear_sky_days(
        bom10min,
        start_day=train_start_day,
        end_day=train_end_day,
    )

    return (
        site_days
        .join(
            cs_days,
            left_on=["n_lat", "n_long"],
            right_on=["latitude", "longitude"],
            how="inner",
        )
        .with_columns([
            (pl.col("actual_day").cast(pl.Int32) - pl.col("clear_sky_day").cast(pl.Int32)).alias("signed_day_diff"),
            (pl.col("actual_day").cast(pl.Int32) - pl.col("clear_sky_day").cast(pl.Int32)).abs().alias("abs_day_diff"),
        ])
        .filter(pl.col("abs_day_diff") < CLEAR_SKY_MAX_DISTANCE_DAYS)
        .sort(["n_lat", "n_long", "actual_day", "abs_day_diff", "signed_day_diff"])
        .with_columns(
            pl.col("clear_sky_day").cum_count().over(["n_lat", "n_long", "actual_day"]).alias("candidate_rank")
        )
        .filter(pl.col("candidate_rank") <= BOM_CLEAR_SKY_CANDIDATES)
        .select([
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


def nearest_clear_sky_days(site_metrology, bom10min):
    """Choose the nearest clear-sky day for each actual site/grid/day."""
    return (
        # Keep the original nearest-day selection as the primary fallback-free
        # result while exposing additional candidates through the helper above.
        nearest_clear_sky_candidates(site_metrology, bom10min)
        .filter(pl.col("candidate_rank") == 1)
        .select(["n_lat", "n_long", "actual_day", "clear_sky_day", "abs_day_diff"])
    )


def segmented_site_metrology(site_metrology):
    """Split days into continuous segments, matching the original 30-minute gap rule."""
    return (
        site_metrology
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
            # The rolling clear-sky profile should not bridge large data gaps.
            pl.col("gap_start").cum_sum().over(["site_id", "actual_day"]).alias("segment_id")
        )
    )


def nearest_clear_sky_profiles(site_metrology, clear_sky_candidates, bom5min):
    """Build clear-sky PV/GHI reference profiles from selected clear-sky days.

    The original workflow assumed the site data itself was already at 5-minute
    grain. Local extracts can have multiple site rows inside one 5-minute bin,
    so the site power profile is collapsed to one 5-minute representative first
    and only then smoothed with the original centred 7-bin rolling percentile.
    """
    candidate_cs_days = (
        # Build profiles once per site and candidate clear-sky day, then re-use
        # those profiles for actual days that map to the same local time-of-day.
        clear_sky_candidates
        .join(
            site_metrology.select(["site_id", "n_lat", "n_long"]).unique(),
            on=["n_lat", "n_long"],
            how="inner",
        )
        .select(["site_id", "n_lat", "n_long", pl.col("clear_sky_day").alias("cs_day")])
        .unique()
    )

    return (
        add_bom_join_timestamp(
            segmented_site_metrology(
                site_metrology.filter(pl.col("dataset_role") == "train")
            )
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
            bom5min,
            left_on=["n_lat", "n_long", "bom_join_ts"],
            right_on=["latitude", "longitude", "time_5min"],
            how="inner",
        )
        .group_by(["site_id", "cs_day", "segment_id", "cs_tod"])
        .agg([
            pl.col("P_kw").quantile(0.6).alias("P_kw_bin_q60"),
            pl.col("P_kw_norm").quantile(0.6).alias("P_kw_norm_bin_q60"),
            pl.col("GHI").first().alias("GHI_bin"),
            pl.col("cloud_type").first().alias("cloud_type_cs"),
            pl.len().alias("bin_source_rows"),
        ])
        .sort(["site_id", "cs_day", "segment_id", "cs_tod"])
        .with_columns([
            pl.col("P_kw_norm_bin_q60")
            .rolling_quantile(0.6, window_size=7, min_samples=1, center=True)
            .over(["site_id", "cs_day", "segment_id"])
            .alias("P_kw_norm_cs"),
            pl.col("GHI_bin")
            .rolling_quantile(0.6, window_size=7, min_samples=1, center=True)
            .over(["site_id", "cs_day", "segment_id"])
            .alias("GHI_cs"),
        ])
    )


def attach_clear_sky_features(site_metrology, bom10min):
    """Join actual rows to BOM and their nearest clear-sky profile.

    t_stamp remains the original source timestamp. For the BOM join only, rows
    use bom_join_ts, the nearest 5-minute timestamp within the configured
    tolerance, at the same n_lat/n_long. Local day/time columns are used after
    that for clear-sky day matching.
    """
    bin_key_cols = ["dataset_role", "site_id", "actual_day", "actual_tod"]
    bom5min = bom10_to_5min(bom10min)
    clear_sky_candidates = nearest_clear_sky_candidates(site_metrology, bom10min)
    cs_profiles = nearest_clear_sky_profiles(site_metrology, clear_sky_candidates, bom5min)

    actual_with_bom = (
        add_bom_join_timestamp(site_metrology)
        .join(
            # Join to the nearest BOM 5-minute timestamp without changing the
            # original site timestamp that is written to structured_data.
            bom5min,
            left_on=["n_lat", "n_long", "bom_join_ts"],
            right_on=["latitude", "longitude", "time_5min"],
            how="inner",
        )
    )

    actual_bin_candidates = (
        actual_with_bom
        .select(["dataset_role", "site_id", "n_lat", "n_long", "actual_day", "actual_tod"])
        .unique()
        .join(clear_sky_candidates, on=["n_lat", "n_long", "actual_day"], how="left")
        .join(
            # Clear-sky profiles are joined by local 5-minute time-of-day, not
            # by the original timestamp.
            cs_profiles,
            left_on=["site_id", "clear_sky_day", "actual_tod"],
            right_on=["site_id", "cs_day", "cs_tod"],
            how="left",
        )
        .with_columns([
            (pl.col("P_kw_norm_cs") > MIN_VALID_CLEAR_SKY_P_KW_NORM).fill_null(False).alias("candidate_valid"),
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
            pl.coalesce(["selected_clear_sky_day", "primary_clear_sky_day"]).alias("clear_sky_day"),
            pl.col("selected_P_kw_norm_cs").alias("P_kw_norm_cs"),
            pl.coalesce(["selected_GHI_cs", "primary_GHI_cs"]).alias("GHI_cs"),
            pl.coalesce(["selected_cloud_type_cs", "primary_cloud_type_cs"]).alias("cloud_type_cs"),
            pl.coalesce(["selected_clear_sky_day", "primary_clear_sky_day"]).alias("cs_day"),
            pl.col("actual_tod").alias("cs_tod"),
        ])
    )

    structured = (
        actual_with_bom
        .join(attached_clear_sky_bins, on=bin_key_cols, how="left")
    )

    return structured


def build_clear_sky_day_diagnostics(structured, site_metrology, bom10min):
    """Summarise the chosen clear-sky day and daylight-bin coverage per day."""
    structured = as_lazy(structured)
    site_metrology = as_lazy(site_metrology)
    daylight_window = (
        (pl.col("actual_tod") >= pl.lit(DIAGNOSTIC_DAYLIGHT_START))
        & (pl.col("actual_tod") < pl.lit(DIAGNOSTIC_DAYLIGHT_END))
    )
    day_key_cols = ["dataset_role", "site_id", "actual_day"]
    clear_sky_candidates = nearest_clear_sky_candidates(site_metrology, bom10min)

    actual_days = (
        structured
        .select(day_key_cols + ["n_lat", "n_long"])
        .unique()
    )

    primary_clear_sky_days = (
        actual_days
        .join(
            clear_sky_candidates
            .filter(pl.col("candidate_rank") == 1)
            .select([
                "n_lat",
                "n_long",
                "actual_day",
                pl.col("clear_sky_day").alias("nearest_clear_sky_day"),
                "cloud_sum",
                "max_GHI",
                pl.col("rn").alias("nearest_clear_sky_bom_rank"),
            ]),
            on=["n_lat", "n_long", "actual_day"],
            how="left",
        )
        .select(day_key_cols + [
            "nearest_clear_sky_day",
            "cloud_sum",
            "max_GHI",
            "nearest_clear_sky_bom_rank",
        ])
    )

    day_counts = (
        structured
        .filter(daylight_window)
        .group_by(day_key_cols)
        .agg([
            pl.col("actual_tod").n_unique().alias("tod_bins_0600_1800"),
            pl.when(pl.col("P_kw_norm_cs").is_not_null())
            .then(pl.col("actual_tod"))
            .otherwise(None)
            .n_unique()
            .alias("tod_bins_with_P_kw_norm_cs_0600_1800"),
        ])
    )

    return (
        primary_clear_sky_days
        .join(day_counts, on=day_key_cols, how="left")
        .with_columns([
            pl.col("tod_bins_0600_1800").fill_null(0),
            pl.col("tod_bins_with_P_kw_norm_cs_0600_1800").fill_null(0),
        ])
    )


def write_csv(df, path):
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    df.write_csv(path)


def write_diagnostic_files(output_dir, bom_mapping, structured,
                           clear_sky_day_diagnostics=None,
                           clear_sky_profile_diagnostics=None):
    """Write optional audit files for a sample/full run."""
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    write_csv(bom_mapping.sort("site_id"), diagnostics_dir / "bom_grid_mapping.csv")

    structured_lf = structured.lazy() if isinstance(structured, pl.DataFrame) else structured
    structured_summary = (
        # This is the first high-level check after a build: enough train and
        # validation rows should survive for the requested cohort.
        structured_lf
        .group_by("dataset_role")
        .agg([
            pl.len().alias("structured_rows"),
            pl.col("site_id").n_unique().alias("sites"),
            pl.col("actual_day").n_unique().alias("actual_days"),
            pl.col("S_norm").min().alias("min_S_norm"),
            pl.col("S_norm").max().alias("max_S_norm"),
        ])
    )
    write_csv(structured_summary, diagnostics_dir / "structured_summary.csv")

    if clear_sky_day_diagnostics is not None:
        write_csv(
            as_lazy(clear_sky_day_diagnostics).sort(["site_id", "actual_day"]),
            diagnostics_dir / "clear_sky_day_diagnostics.csv",
        )
    if clear_sky_profile_diagnostics is not None:
        write_csv(
            as_lazy(clear_sky_profile_diagnostics),
            diagnostics_dir / "clear_sky_profile_diagnostics.csv",
        )


def run(
    *,
    output_dir=OUTPUT_DIR,
    sapn_validation_data_path=SAPN_CLEANED_DATA_PATH,
    site_id=SITE_ID,
    limit_sites=LIMIT_SITES,
    write_diagnostics=WRITE_DIAGNOSTICS,
):
    """Run the full local structured-data build with a few explicit overrides."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sapn_validation_data_path = Path(sapn_validation_data_path)
    if not sapn_validation_data_path.exists():
        raise FileNotFoundError(
            f"SAPN validation data file not found: {sapn_validation_data_path}. "
            "Run data_processing.py to create the cleaned parquet or pass another path to run(...)."
        )

    # Training uses EVM data. Validation/test uses SAPN2022 data. BOM is read
    # over a wider window so clear-sky reference days can be found near both.
    train_start = TRAIN_START_DATE
    train_end = TRAIN_END_DATE
    validation_start = VALIDATION_START_DATE
    validation_end = VALIDATION_END_DATE
    bom_start = BOM_START_DATE if BOM_START_DATE else min(train_start, validation_start) - timedelta(days=45)
    bom_end = BOM_END_DATE if BOM_END_DATE else max(train_end, validation_end) + timedelta(days=45)

    print("Loading metadata")
    sapn_sites = read_sapn_site_details(SAPN_SITE_DETAILS_PATH)
    sapn_circuits = read_sapn_circuit_details(SAPN_CIRCUIT_DETAILS_PATH)
    evm_sites = read_evm_site_metadata(EVM_SITE_METADATA_PATH)
    evm_circuits = read_evm_circuit_metadata(EVM_CIRCUIT_METADATA_PATH)

    # Build the SAPN validation cohort, then reuse the same site IDs for EVM
    # training so capacity, BOM grid mapping, and later scoring are consistent.
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

    # Select only the PV circuits needed for site-level PV generation. SAPN and
    # EVM metadata use different source column names, but are standardised to
    # site_id/c_id/polarity before this point.
    sapn_pv_circuits = pv_circuits_for_sites(sapn_circuits, eligible_sites)
    evm_pv_circuits = pv_circuits_for_sites(evm_circuits, eligible_sites)
    bom_mapping = map_sites_to_bom_grid(eligible_sites, BOM_POINTS_CSV)

    evm_parquets = evm_training_parquets(EVM_TRAINING_DIR, train_start, train_end)
    bom_files = bom_daily_parquets(BOM_ROOT, bom_start, bom_end)
    print(f"EVM training parquet files: {len(evm_parquets)}")
    print(f"BOM files: {len(bom_files)}")

    # Read and normalise raw metrology. After this step, both sources use kW
    # power, UTC/source t_stamp, and the shared dataset_role label.
    print("Preparing metrology")
    site_ids = eligible_sites["site_id"].to_list()
    train_raw = sapn_funcs.prepare_evm_metrology(
        evm_parquets,
        evm_pv_circuits,
        train_start,
        train_end,
    )
    validation_raw = sapn_funcs.prepare_sapn_metrology(
        sapn_validation_data_path,
        sapn_pv_circuits,
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
    site_metrology = (
        pl.concat([train_site_metrology, validation_site_metrology], how="vertical")
        .lazy()
        .with_columns([
            adelaide_datetime_expr("t_stamp").dt.date().alias("actual_day"),
            time_of_day_5min_expr("t_stamp").alias("actual_tod"),
        ])
        .filter(~pl.col("actual_day").is_in(sorted(EXCLUDED_LOCAL_DAYS)))
    )

    print("Resolving capacity")
    capacity = resolve_capacity(eligible_sites, site_metrology)
    site_metrology = add_capacity_and_grid(site_metrology, capacity, bom_mapping)

    # Attach BOM irradiance and clear-sky references. This is the highest-risk
    # coverage stage because it uses exact timestamp joins to BOM 5-minute rows.
    print("Preparing BOM and clear-sky features")
    bom10min = prepare_bom10min(bom_files, bom_mapping)
    structured = attach_clear_sky_features(site_metrology, bom10min)

    structured = (
        structured
        .select(STRUCTURED_COLUMNS)
        .sort(["dataset_role", "site_id", "t_stamp"])
    )

    output_path = output_dir / "structured_data.parquet"
    print(f"Writing {output_path}")
    # Stream the lazy query to parquet so the full structured data does not
    # need to be materialised in memory before writing.
    structured.sink_parquet(output_path, compression="zstd")

    if WRITE_DIAGNOSTICS:
        print("Writing diagnostics")
        structured_df = pl.scan_parquet(output_path).collect()
        clear_sky_day_diagnostics = build_clear_sky_day_diagnostics(
            structured_df,
            site_metrology,
            bom10min,
        )
        write_diagnostic_files(
            output_dir,
            bom_mapping,
            structured_df,
            clear_sky_day_diagnostics=clear_sky_day_diagnostics,
        )
    print("Done")

def main():
    """Run the full local structured-data build."""
    return run(
        output_dir=OUTPUT_DIR,
        sapn_validation_data_path=SAPN_CLEANED_DATA_PATH,
        # site_id=SITE_ID,
        # limit_sites=LIMIT_SITES,
        write_diagnostics=WRITE_DIAGNOSTICS,
    )

if __name__ == "__main__":
    main()
