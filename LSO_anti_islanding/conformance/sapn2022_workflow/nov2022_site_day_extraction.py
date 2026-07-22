"""Extract and deduplicate one SAPN November 2022 site-day."""

import polars as pl


def _dedupe_circuit_rows(df, highest=False):
    dupes = (
        df.group_by("local_tstamp")
        .agg([
            pl.count().alias("n_rows"),
            pl.col("power").n_unique().alias("power_n_unique"),
        ])
        .filter(pl.col("n_rows") > 1)
    )
    same_power_count = dupes.filter(pl.col("power_n_unique") == 1).height
    different_power_count = dupes.filter(pl.col("power_n_unique") > 1).height

    if different_power_count > 0:
        if highest is True:
            df = df.group_by("local_tstamp").agg([
                pl.col("power").max().alias("power"),
                pl.all().exclude("power").first(),
            ])
        else:
            bad_timestamps = (
                dupes.filter(pl.col("power_n_unique") > 1).select("local_tstamp")
            )
            df = df.join(bad_timestamps, on="local_tstamp", how="anti")

    if same_power_count > 0:
        df = df.unique(subset=["local_tstamp"], keep="first")
    return df.sort("local_tstamp")


def extract_nov2022_site_day(
    all_data,
    circuit_details,
    site_number,
    start_day,
    end_day,
):
    circuit_details_ldf = (
        circuit_details if isinstance(circuit_details, pl.LazyFrame) else circuit_details.lazy()
    )
    pv_circuit_ids = (
        circuit_details_ldf.filter(
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
        all_data_ldf.filter(pl.col("c_id").is_in(pv_circuit_ids))
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
    return site_day_df.group_by("c_id", maintain_order=True).map_groups(
        _dedupe_circuit_rows
    )
