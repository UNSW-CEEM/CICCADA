"""Extract one pre-cleaned SAPN November 2022 site-day."""

import polars as pl

from core.data_cleaning import deduplicateMeasurements


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
                "local_tstamp": pl.Datetime,
                "utc_tstamp": pl.Datetime(time_zone="UTC"),
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
    # The shared preprocessing step guarantees uniqueness for newly cleaned
    # data. Keep this idempotent call so older cleaned parquets remain usable.
    return deduplicateMeasurements(site_day_df).sort("local_tstamp")
