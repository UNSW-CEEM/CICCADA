"""Shared builder utilities used by both structured-data builders.

This module intentionally contains only the generic builder-level helpers and
constants shared by both `build_structured_high_resolution.py` and
`build_structured_5m.py`. Run-specific paths, date windows, and builder-specific
workflow functions stay in the build scripts themselves.
"""

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ADELAIDE_TZ = "Australia/Adelaide"

# Capacity is normally taken from metadata. The observed-peak fallback is only
# for sites where metadata is missing or clearly lower than observed generation.
CAPACITY_TOLERANCE = 1.10
FALLBACK_CAPACITY_KW = 5.0

# These four constants are intentionally shared by both structured-data
# builders so they apply the same clear-sky candidate ranking, row filtering,
# and reactive-power placeholder behavior.
# Number of ranked BOM clear-sky candidate days retained for each site/day.
BOM_CLEAR_SKY_CANDIDATES = 3
# Minimum clear-sky normalised power required before a candidate profile is
# treated as valid rather than effectively empty/noisy.
MIN_VALID_CLEAR_SKY_P_KW_NORM = 0.2
# Local Adelaide dates excluded from both builders because those validation
# days are intentionally omitted from the structured outputs.
EXCLUDED_LOCAL_DAYS = {
    date(2022, 11, 12),
    date(2022, 11, 18),
}

# The local SAPN/EVM files inspected for this workflow do not include reactive
# power. The original Solar Analytics filter uses S_norm to remove apparent
# power limited rows; setting it to 1.0 preserves all rows for that filter while
# making the bypass explicit in the structured data.
S_NORM_REACTIVE_POWER_UNAVAILABLE = 1.0


def parse_date(value):
    """Accept either an ISO date string or an already-parsed date object."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def date_range(start_date, end_date):
    """Yield every calendar day from start_date through end_date inclusive."""
    current = parse_date(start_date)
    end_date = parse_date(end_date)
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def evm_training_parquets(evm_training_dir, start_date, end_date):
    """Return the pre-converted EVM training parquets in the folder.

    The parquet files should preserve the original EVM CSV row grain and
    columns. This script intentionally does not own the CSV-to-parquet
    conversion step.

    File discovery is deliberately folder-based rather than filename-date based.
    The actual training date window is applied later after parsing utc_tstamp,
    which is safer than assuming every parquet filename follows a date pattern.
    """
    files = sorted(Path(evm_training_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No EVM training parquet files found in {evm_training_dir}"
        )
    return files


def bom_daily_parquets(bom_root, start_date, end_date):
    """Return BOM daily parquet files in the requested window."""
    bom_root = Path(bom_root)
    files = []
    for day in date_range(start_date, end_date):
        daily_path = (
            bom_root
            / f"{day.year}"
            / f"{day.month}"
            / f"{day.day}"
            / f"{day:%Y%m%d}.parquet"
        )
        if daily_path.exists():
            files.append(daily_path)
    if files:
        return files

    # If the expected year/month/day layout is not present, fall back to all
    # parquet files under the BOM root. This keeps the script usable with a
    # flatter local extract.
    return sorted(bom_root.rglob("*.parquet"))


def adelaide_datetime_expr(column_name):
    """Convert UTC timestamps into Adelaide local timestamps for solar day features."""
    return (
        pl.col(column_name)
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(ADELAIDE_TZ)
    )


def end_labeled_5min_utc_expr(column_name):
    bucket_floor = pl.col(column_name).dt.truncate("5m")
    return (
        pl.when(pl.col(column_name) == bucket_floor)
        .then(bucket_floor)
        .otherwise(bucket_floor + pl.duration(minutes=5))
    )


def time_of_day_5min_expr(column_name):
    """Use SAPN end-labeled 5-minute local time-of-day bins."""
    local_ts = adelaide_datetime_expr(column_name)
    bucket_floor = local_ts.dt.truncate("5m")
    return (
        pl.when(local_ts == bucket_floor)
        .then(bucket_floor)
        .otherwise(bucket_floor + pl.duration(minutes=5))
        .dt.time()
    )


def read_sapn_site_details(path):
    """Read SAPN site metadata, including AC capacity in W."""
    return pl.read_csv(path).with_columns(
        [
            pl.col("site_id").cast(pl.Int64),
            pl.col("ac_cap_w").cast(pl.Float64, strict=False),
        ]
    )


def read_sapn_circuit_details(path):
    """Read SAPN circuit metadata and standardise ID/polarity types."""
    return pl.read_csv(path).with_columns(
        [
            pl.col("site_id").cast(pl.Int64),
            pl.col("c_id").cast(pl.Int64),
            pl.col("polarity").cast(pl.Float64, strict=False),
        ]
    )


def read_evm_site_metadata(path):
    """Read EVM site metadata and rename fields to avoid SAPN/EVM ambiguity."""
    return (
        pl.read_csv(path)
        .rename(
            {
                "postcode": "evm_postcode",
                "latitude": "site_latitude",
                "longitude": "site_longitude",
                "ac_capacity_kw": "evm_ac_capacity_kw",
            }
        )
        .with_columns(
            [
                pl.col("site_id").cast(pl.Int64),
                pl.col("site_latitude").cast(pl.Float64, strict=False),
                pl.col("site_longitude").cast(pl.Float64, strict=False),
                pl.col("evm_ac_capacity_kw").cast(pl.Float64, strict=False),
            ]
        )
    )


def read_evm_circuit_metadata(path):
    """Read EVM circuit metadata using the same names as SAPN circuit metadata."""
    return (
        pl.read_csv(path)
        .rename(
            {
                "circuit_id": "c_id",
                "circuit_type": "con_type",
                "circuit_polarity": "polarity",
            }
        )
        .with_columns(
            [
                pl.col("site_id").cast(pl.Int64),
                pl.col("c_id").cast(pl.Int64),
                pl.col("polarity").cast(pl.Float64, strict=False),
            ]
        )
    )


def read_site_cohort(path):
    """Read an optional final-analysis site cohort from a CSV with site_id."""
    cohort = (
        pl.read_csv(path)
        .select("site_id")
        .with_columns(pl.col("site_id").cast(pl.Int64))
        .unique()
    )
    if cohort.is_empty():
        raise ValueError(f"Site cohort is empty: {path}")
    return cohort


def build_eligible_sites(
    site_details, circuit_details, evm_sites, site_cohort=None, limit_sites=None
):
    """Validate which sites can be used in the structured-data build.

    Returns:
    - eligible: only the sites that pass every requirement and will be processed.
    - candidates: every site that was checked, with eligibility_status explaining
      whether it passed or why it was excluded. The main workflow ignores this,
      but it can be useful while manually debugging site eligibility.

    A site is eligible only if all of these are true:
    - SAPN site metadata has exactly one row for the site_id.
    - SAPN circuit metadata has at least one pv_site_net circuit for the site.
    - SAPN circuit metadata has no more than three pv_site_net circuits.
    - EVM site metadata has latitude and longitude for the same site_id.
    - If a cohort CSV is supplied, the site_id appears in that cohort.
    """
    # Count SAPN site metadata rows. A duplicate/missing site metadata row makes
    # capacity and site-level joins ambiguous.
    site_row_counts = (
        site_details.group_by("site_id").len().rename({"len": "site_metadata_rows"})
    )
    # Count SAPN PV site-net circuits. These are the circuits that will be
    # summed into site-level PV generation.
    pv_counts = (
        circuit_details.filter(pl.col("con_type") == "pv_site_net")
        .group_by("site_id")
        .len()
        .rename({"len": "pv_site_net_count"})
    )

    # candidates keeps all sites and annotates each with an eligibility_status.
    # We keep excluded rows here so diagnostics can explain what happened.
    candidates = (
        site_row_counts.join(pv_counts, on="site_id", how="left")
        .with_columns(pl.col("pv_site_net_count").fill_null(0))
        .join(
            site_details.select(["site_id", "ac_cap_w"]).unique(
                subset=["site_id"], keep="first"
            ),
            on="site_id",
            how="left",
        )
        .join(
            evm_sites.select(
                [
                    "site_id",
                    "site_latitude",
                    "site_longitude",
                    "evm_postcode",
                    "evm_ac_capacity_kw",
                ]
            ),
            on="site_id",
            how="left",
        )
        .with_columns(
            # First failing condition wins. If none fail, the site is eligible.
            pl.when(pl.col("site_metadata_rows") != 1)
            .then(pl.lit("site_metadata_rows_not_1"))
            .when(pl.col("pv_site_net_count") == 0)
            .then(pl.lit("no_pv_site_net"))
            .when(pl.col("pv_site_net_count") > 3)
            .then(pl.lit("more_than_3_pv_site_net"))
            .when(
                pl.col("site_latitude").is_null() | pl.col("site_longitude").is_null()
            )
            .then(pl.lit("missing_evm_lat_lon"))
            .otherwise(pl.lit("eligible"))
            .alias("eligibility_status")
        )
    )

    if site_cohort is not None:
        # The default workflow intentionally restricts modelling to the final
        # confidence-tier SAPN cohort. Sites outside this cohort are not errors;
        # they are marked as outside_site_cohort for diagnostics.
        candidates = (
            candidates.join(
                site_cohort.with_columns(pl.lit(True).alias("in_site_cohort")),
                on="site_id",
                how="left",
            )
            .with_columns(pl.col("in_site_cohort").fill_null(False))
            .with_columns(
                pl.when(~pl.col("in_site_cohort"))
                .then(pl.lit("outside_site_cohort"))
                .otherwise(pl.col("eligibility_status"))
                .alias("eligibility_status")
            )
        )

    # eligible is the subset that actually flows into PV circuit selection,
    # BOM grid mapping, metrology reads, and the final structured dataset.
    eligible = candidates.filter(pl.col("eligibility_status") == "eligible")
    if limit_sites is not None:
        eligible = eligible.sort("site_id").head(limit_sites)

    return eligible, candidates


def pv_circuits_for_sites(circuit_details, eligible_sites):
    """Keep only PV circuits for selected sites."""
    site_ids = eligible_sites["site_id"].to_list()
    return (
        circuit_details.filter(
            (pl.col("site_id").is_in(site_ids)) & (pl.col("con_type") == "pv_site_net")
        )
        .select(["site_id", "c_id", "con_type", "polarity"])
        .unique()
    )


def map_sites_to_bom_grid(eligible_sites, bom_points_csv):
    """Map each site to the nearest available BOM grid point."""
    # All candidate BOM grid points. Postcode is not used here; we choose the
    # closest grid point from the available latitude/longitude list.
    bom_points = (
        pl.read_csv(bom_points_csv)
        .select(["latitude", "longitude"])
        .with_columns(
            [
                pl.col("latitude").cast(pl.Float64, strict=False),
                pl.col("longitude").cast(pl.Float64, strict=False),
            ]
        )
        .drop_nulls()
        .unique()
    )

    grid_lat = bom_points["latitude"].to_numpy()
    grid_lon = bom_points["longitude"].to_numpy()
    rows = []

    for row in eligible_sites.iter_rows(named=True):
        # Original Trino metadata already had n_lat/n_long. Locally SAPN
        # metadata does not, so derive it from the EVM site lat/lon.
        site_lat = row["site_latitude"]
        site_lon = row["site_longitude"]

        # Euclidean distance in latitude/longitude degrees is enough here
        # because we only need the nearest point in the local BOM grid.
        distance_sq = np.square(grid_lat - site_lat) + np.square(grid_lon - site_lon)
        nearest_idx = int(np.argmin(distance_sq))
        rows.append(
            {
                "site_id": row["site_id"],
                "site_latitude": site_lat,
                "site_longitude": site_lon,
                # n_lat/n_long are the BOM coordinates used in later joins.
                "n_lat": float(grid_lat[nearest_idx]),
                "n_long": float(grid_lon[nearest_idx]),
                "bom_grid_distance_degrees": float(math.sqrt(distance_sq[nearest_idx])),
            }
        )

    return pl.DataFrame(rows)


def round_up_to_half_kw_expr(column_name):
    return (pl.col(column_name) * 2.0).ceil() / 2.0


def resolve_capacity(eligible_sites, site_metrology):
    """Resolve the capacity denominator used for P_kw_norm and final scoring.

    This follows the SAPN ratedCapacityOfPV idea: use metadata capacity when it
    is plausible, otherwise use a robust observed peak, with 5 kW only as the
    final fallback.
    """
    metadata = (
        eligible_sites.lazy()
        .select(["site_id", "ac_cap_w", "evm_ac_capacity_kw"])
        .with_columns(
            [
                (pl.col("ac_cap_w").cast(pl.Float64, strict=False) / 1000.0).alias(
                    "sapn_ac_capacity_kw"
                ),
                pl.col("evm_ac_capacity_kw").cast(pl.Float64, strict=False),
            ]
        )
        .with_columns(
            pl.when(pl.col("sapn_ac_capacity_kw") > 0)
            .then(pl.col("sapn_ac_capacity_kw"))
            .when(pl.col("evm_ac_capacity_kw") > 0)
            .then(pl.col("evm_ac_capacity_kw"))
            .otherwise(None)
            .alias("metadata_capacity_kw")
        )
    )

    positive_power = (
        # Robust observed peak is based on positive PV generation only. It is
        # not the normal path when metadata capacity is plausible.
        site_metrology.filter(pl.col("P_kw") > 0).select(["site_id", "P_kw"])
    )
    counts = positive_power.group_by("site_id").len().rename({"len": "sample_count"})
    ranked = (
        positive_power.join(counts, on="site_id", how="left")
        .sort(["site_id", "P_kw"], descending=[False, True])
        .with_columns(
            [
                pl.col("P_kw").cum_count().over("site_id").alias("rank_desc"),
                pl.max_horizontal(
                    pl.lit(20),
                    (pl.col("sample_count").cast(pl.Float64) * 0.01)
                    .ceil()
                    .cast(pl.Int64),
                ).alias("top_n"),
            ]
        )
        .filter(pl.col("rank_desc") <= pl.col("top_n"))
    )
    robust_peak = ranked.group_by("site_id").agg(
        [
            pl.col("P_kw").median().alias("robust_peak_kw"),
            pl.col("P_kw").max().alias("raw_peak_kw"),
            pl.col("sample_count").max().alias("positive_power_samples"),
        ]
    )

    return (
        metadata.join(robust_peak, on="site_id", how="left")
        .with_columns(
            pl.when(
                pl.col("metadata_capacity_kw").is_not_null()
                & (
                    pl.col("robust_peak_kw").is_null()
                    | (
                        pl.col("robust_peak_kw")
                        <= pl.col("metadata_capacity_kw") * CAPACITY_TOLERANCE
                    )
                )
            )
            .then(pl.col("metadata_capacity_kw"))
            .when(
                pl.col("metadata_capacity_kw").is_not_null()
                & (
                    pl.col("robust_peak_kw")
                    > pl.col("metadata_capacity_kw") * CAPACITY_TOLERANCE
                )
            )
            .then(round_up_to_half_kw_expr("robust_peak_kw"))
            .when(pl.col("robust_peak_kw").is_not_null())
            .then(round_up_to_half_kw_expr("robust_peak_kw"))
            .otherwise(pl.lit(FALLBACK_CAPACITY_KW))
            .alias("ac_capacity_kw")
        )
        .with_columns(
            pl.when(
                pl.col("metadata_capacity_kw").is_not_null()
                & (
                    pl.col("robust_peak_kw").is_null()
                    | (
                        pl.col("robust_peak_kw")
                        <= pl.col("metadata_capacity_kw") * CAPACITY_TOLERANCE
                    )
                )
            )
            .then(pl.lit("metadata"))
            .when(
                pl.col("metadata_capacity_kw").is_not_null()
                & (
                    pl.col("robust_peak_kw")
                    > pl.col("metadata_capacity_kw") * CAPACITY_TOLERANCE
                )
            )
            .then(pl.lit("observed_peak_metadata_exceeded"))
            .when(pl.col("robust_peak_kw").is_not_null())
            .then(pl.lit("observed_peak"))
            .otherwise(pl.lit("fallback_5kw"))
            .alias("capacity_source")
        )
        .with_columns(pl.col("ac_capacity_kw").alias("S_99"))
    )


def as_lazy(df):
    if isinstance(df, pl.LazyFrame):
        return df
    return df.lazy()


def add_capacity_and_grid(site_metrology, capacity, bom_mapping):
    """Attach capacity, n_lat/n_long, and normalised P/Q/S columns."""
    return (
        site_metrology.join(
            as_lazy(capacity).select(
                ["site_id", "ac_capacity_kw", "S_99", "capacity_source"]
            ),
            on="site_id",
            how="inner",
        )
        .join(
            bom_mapping.lazy().select(["site_id", "n_lat", "n_long"]),
            on="site_id",
            how="inner",
        )
        .with_columns(
            [
                (pl.col("P_kw") / pl.col("S_99")).alias("P_kw_norm"),
                # Q is unavailable in these local extracts. S_norm=1.0 makes the
                # original apparent-power filter pass without pretending Q exists.
                pl.lit(None).cast(pl.Float64).alias("Q_kvar_norm"),
                pl.lit(S_NORM_REACTIVE_POWER_UNAVAILABLE)
                .cast(pl.Float64)
                .alias("S_norm"),
            ]
        )
    )


def prepare_bom10min(bom_files, bom_mapping):
    """Read BOM rows only for the selected nearest grid points."""
    grid_points = bom_mapping.select(
        [
            pl.col("n_lat").alias("latitude"),
            pl.col("n_long").alias("longitude"),
        ]
    ).unique()

    return (
        # The BOM scan is restricted to selected n_lat/n_long points before
        # downstream joins so sample runs do not carry every BOM grid location.
        pl.scan_parquet([str(path) for path in bom_files])
        .with_columns(
            [
                pl.col("latitude").cast(pl.Float64, strict=False),
                pl.col("longitude").cast(pl.Float64, strict=False),
                pl.col("time").cast(pl.Datetime).alias("time"),
                pl.col("surface_global_irradiance")
                .cast(pl.Float64, strict=False)
                .alias("GHI"),
                pl.col("cloud_type").cast(pl.Float64, strict=False).alias("cloud_type"),
            ]
        )
        .join(grid_points.lazy(), on=["latitude", "longitude"], how="inner")
        .select(["time", "latitude", "longitude", "GHI", "cloud_type"])
        .unique()
    )


def bom10_to_5min(bom10min):
    """Replicate the original BOM 10-minute to 5-minute expansion."""
    # Original workflow treats each 10-minute BOM irradiance value as applying
    # to its timestamp and the +5 minute timestamp.
    original = bom10min.select(
        [
            pl.col("time").alias("time_5min"),
            "latitude",
            "longitude",
            "GHI",
            "cloud_type",
        ]
    )
    plus_five = bom10min.select(
        [
            (pl.col("time") + pl.duration(minutes=5)).alias("time_5min"),
            "latitude",
            "longitude",
            "GHI",
            "cloud_type",
        ]
    )
    return pl.concat([original, plus_five], how="vertical")


def clear_sky_days(bom10min, start_day=None, end_day=None):
    """Find low-cloud/high-GHI BOM days by grid point and local month."""
    daily_cloud = (
        bom10min.with_columns(
            [
                adelaide_datetime_expr("time").dt.date().alias("day"),
                adelaide_datetime_expr("time").dt.month().alias("month"),
            ]
        )
        .filter(~pl.col("day").is_in(sorted(EXCLUDED_LOCAL_DAYS)))
        .group_by(["latitude", "longitude", "day", "month"])
        .agg(
            [
                pl.col("cloud_type").sum().alias("cloud_sum"),
                pl.col("GHI").max().alias("max_GHI"),
            ]
        )
    )

    if start_day is not None and end_day is not None:
        daily_cloud = daily_cloud.filter(
            pl.col("day").is_between(start_day, end_day, closed="both")
        )

    return (
        # Select up to the first three low-cloud, high-irradiance days per
        # month/grid point. These become candidates for clear-sky references.
        daily_cloud.sort(["month", "latitude", "longitude", "cloud_sum", "day"])
        .with_columns(
            pl.col("day")
            .cum_count()
            .over(["month", "latitude", "longitude"])
            .alias("rn")
        )
        .filter(
            (pl.col("rn") <= BOM_CLEAR_SKY_CANDIDATES)
            & (pl.col("cloud_sum") <= 60)
            & (pl.col("max_GHI") > 200)
        )
        .select(
            [
                pl.col("day").alias("clear_sky_day"),
                "latitude",
                "longitude",
                "cloud_sum",
                "max_GHI",
                "rn",
            ]
        )
    )
