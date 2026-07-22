"""Shared measurement-cleaning functions used by data adapters.

The historical mixed-purpose helpers have moved to ``legacy/funcs_legacy.py``.
The implementations remain numerically unchanged from the verified workflow.
"""

import polars as pl

from config import LOCAL_TIMEZONE, VALID_VOLTAGE_MAX, VALID_VOLTAGE_MIN


def convertPowerToKw(allData, convert=False):
    allData = allData.with_columns((pl.col("power") / (100 * 1000)).alias("power"))
    print("Converting power to KW")
    return allData


def addLocalTStamp(ldf, add=False):
    if add is True:
        ldf = ldf.with_columns(
            pl.col("utc_tstamp")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(LOCAL_TIMEZONE)
            .alias("local_tstamp")
        )
    return ldf


def addValidVoltage(df, Vmin=VALID_VOLTAGE_MIN, Vmax=VALID_VOLTAGE_MAX):
    return (
        df.with_columns(
            pl.col("voltage").cast(pl.Float64, strict=False).alias("voltage_f")
        )
        .with_columns(
            pl.when(
                pl.col("voltage_f").is_not_null()
                & pl.col("voltage_f").is_between(Vmin, Vmax)
            )
            .then(pl.col("voltage_f"))
            .when(
                pl.col("vmean").is_not_null()
                & pl.col("vmean").is_between(Vmin, Vmax)
            )
            .then(pl.col("vmean"))
            .otherwise(None)
            .alias("voltage_valid")
        )
    )


def dedupeCircuitPolarity(circuitDetails):
    polarity_lookup = (
        circuitDetails.select(["c_id", "polarity"])
        .group_by("c_id")
        .agg([
            pl.len().alias("_metadata_rows"),
            pl.col("polarity").n_unique().alias("_polarity_n_unique"),
            pl.col("polarity").first().alias("polarity"),
        ])
    )
    conflicts = polarity_lookup.filter(pl.col("_polarity_n_unique") > 1)
    if not conflicts.is_empty():
        bad_cids = conflicts["c_id"].head(10).to_list()
        raise ValueError(
            "Conflicting polarity values found for duplicated c_id rows in "
            f"circuit details: {bad_cids}"
        )
    return polarity_lookup.select(["c_id", "polarity"])


def addPolarityToPower(df, circuitDetails):
    polarity_lookup = dedupeCircuitPolarity(circuitDetails)
    return (
        df.join(polarity_lookup.lazy(), on="c_id", how="left")
        .with_columns((pl.col("power") * pl.col("polarity")).alias("power"))
        .drop("polarity")
    )
