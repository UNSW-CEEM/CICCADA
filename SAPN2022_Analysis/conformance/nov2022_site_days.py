import polars as pl

from checkData import checkDupes
from checkPVBehaviour import CheckPVBehaviour
from funcs import mapCircuitDataToSite


DAY_COVERAGE_THRESHOLD = 0.80


def build_nov2022_site_day_long(all_data, circuit_details, site_number, start_day, end_day):
    circuit_details_ldf = circuit_details if isinstance(circuit_details, pl.LazyFrame) else circuit_details.lazy()
    pv_circuit_ids = (
        circuit_details_ldf
        .filter(
            (pl.col("site_id") == site_number)
            & (pl.col("con_type") == "pv_site_net")
        )
        .select("c_id")
        .unique()
        .collect()["c_id"]
        .to_list()
    )
    if not pv_circuit_ids:
        return pl.DataFrame(
            schema={
                "c_id": pl.Int64,
                "local_tstamp": pl.Datetime(time_zone="Australia/Adelaide"),
                "utc_tstamp": pl.Utf8,
                "duration": pl.Float64,
                "power": pl.Float64,
                "voltage_valid": pl.Float64,
            }
        )

    all_data_ldf = all_data if isinstance(all_data, pl.LazyFrame) else all_data.lazy()
    site_day_df = (
        all_data_ldf
        .filter(pl.col("c_id").is_in(pv_circuit_ids))
        .select([
            "c_id",
            "local_tstamp",
            "utc_tstamp",
            "duration",
            "power",
            "voltage_valid",
        ])
        .filter(pl.col("local_tstamp").is_between(start_day, end_day, closed="both"))
        .collect()
    )
    if site_day_df.is_empty():
        return site_day_df

    return site_day_df.group_by("c_id", maintain_order=True).map_groups(checkDupes)


def summarize_nov2022_day_eligibility(
    site_day_long,
    prepared_day_df,
    coverage_threshold=DAY_COVERAGE_THRESHOLD,
    window_seconds=12 * 60 * 60,
):
    required_columns = {"local_tstamp", "utc_tstamp", "duration", "power"}
    if site_day_long.is_empty() or not required_columns.issubset(set(site_day_long.columns)):
        return {
            "total_rows": 0,
            "rows_with_power": 0,
            "rows_with_v10m": 0,
            "rows_common_power_v10m": 0,
            "covered_seconds": 0.0,
            "window_seconds": float(window_seconds),
            "common_power_v10m_coverage_pct": 0.0,
            "eligible": False,
            "reason": "insufficient_raw_columns",
        }

    v10m_lookup = (
        prepared_day_df
        .select(["local_tstamp", "v10m_avg"])
        .unique(subset=["local_tstamp"], keep="first")
    )
    df = (
        site_day_long
        .group_by(["local_tstamp", "utc_tstamp", "duration"])
        .agg(
            pl.col("power").is_not_null().any().alias("_has_power")
        )
        .join(
            v10m_lookup,
            on="local_tstamp",
            how="left",
        )
        .with_columns([
            pl.col("v10m_avg").is_not_null().alias("_has_v10m"),
            pl.col("duration").cast(pl.Float64, strict=False).fill_null(0.0).alias("_duration_s"),
        ])
        .with_columns(
            (pl.col("_has_power") & pl.col("_has_v10m")).alias("_has_common_power_v10m")
        )
    )

    total_rows = int(df.height)
    rows_with_power = int(df.filter(pl.col("_has_power")).height)
    rows_with_v10m = int(df.filter(pl.col("_has_v10m")).height)
    rows_common_power_v10m = int(df.filter(pl.col("_has_common_power_v10m")).height)
    covered_seconds = float(
        df.select(
            pl.when(pl.col("_has_common_power_v10m"))
            .then(pl.col("_duration_s"))
            .otherwise(0)
            .sum()
        ).item()
    )
    coverage_pct = 0.0 if window_seconds == 0 else (covered_seconds / window_seconds) * 100.0

    return {
        "total_rows": total_rows,
        "rows_with_power": rows_with_power,
        "rows_with_v10m": rows_with_v10m,
        "rows_common_power_v10m": rows_common_power_v10m,
        "covered_seconds": covered_seconds,
        "window_seconds": float(window_seconds),
        "common_power_v10m_coverage_pct": coverage_pct,
        "eligible": coverage_pct >= (coverage_threshold * 100.0),
        "reason": None if coverage_pct >= (coverage_threshold * 100.0) else "common_power_v10m_coverage_below_threshold",
    }


def collect_site_days(site_number, circuit_details, all_data, days_to_check):
    eligible_day_behaviours = []
    excluded_day_rows = []
    mapped_day_count = 0

    for day in days_to_check:
        start_day = pl.datetime(2022, 11, day, 6, 0, 0, time_zone="Australia/Adelaide")
        end_day = pl.datetime(2022, 11, day, 18, 0, 0, time_zone="Australia/Adelaide")

        site_day_long = build_nov2022_site_day_long(
            all_data,
            circuit_details,
            site_number,
            start_day,
            end_day,
        )
        if site_day_long.is_empty():
            continue

        mapped_day_count += 1
        wide = mapCircuitDataToSite(site_day_long, site_number)
        behaviour = CheckPVBehaviour(wide, volCol="voltage_valid")
        prepared_day_df = behaviour.prepare_site_day_frame()
        eligibility = summarize_nov2022_day_eligibility(
            site_day_long,
            prepared_day_df,
            coverage_threshold=DAY_COVERAGE_THRESHOLD,
        )
        if eligibility["eligible"]:
            eligible_day_behaviours.append({
                "day": day,
                "behaviour": behaviour,
            })
            continue

        excluded_day_rows.append({
            "site_id": site_number,
            "day": day,
            "reason": eligibility["reason"],
            "common_power_v10m_coverage_pct": eligibility["common_power_v10m_coverage_pct"],
            "rows_common_power_v10m": eligibility["rows_common_power_v10m"],
            "rows_with_power": eligibility["rows_with_power"],
            "rows_with_v10m": eligibility["rows_with_v10m"],
            "covered_seconds": eligibility["covered_seconds"],
            "window_seconds": eligibility["window_seconds"],
            "total_rows": eligibility["total_rows"],
            "coverage_threshold_pct": DAY_COVERAGE_THRESHOLD * 100.0,
        })

    return {
        "eligible_day_behaviours": eligible_day_behaviours,
        "excluded_day_rows": excluded_day_rows,
        "mapped_day_count": mapped_day_count,
    }
