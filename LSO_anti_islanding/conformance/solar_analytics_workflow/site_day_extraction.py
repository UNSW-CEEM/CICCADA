"""Extract one pre-cleaned Solar Analytics site-day."""

import polars as pl

from core.data_cleaning import deduplicateMeasurements


def extract_solar_analytics_site_day(
    all_data,
    site_number,
    start_day,
    end_day,
):
    """Return the site's PV circuits within an inclusive local-time window."""
    all_data_ldf = all_data if isinstance(all_data, pl.LazyFrame) else all_data.lazy()
    site_day_df = (
        all_data_ldf.filter(
            (pl.col("site_id") == site_number)
            & (pl.col("con_type") == "pv_site_net")
        )
        .select([
            "c_id",
            "local_tstamp",
            "utc_tstamp",
            "power",
            "voltage_valid",
        ])
        .filter(pl.col("local_tstamp").is_between(start_day, end_day, closed="both"))
        .collect()
    )
    return deduplicateMeasurements(site_day_df).sort(["local_tstamp", "c_id"])
