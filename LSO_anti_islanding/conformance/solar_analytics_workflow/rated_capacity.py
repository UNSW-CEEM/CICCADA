"""Solar Analytics rated-capacity policy."""

import polars as pl


def add_s_rated_capacity(site_details):
    """Add S_rated from valid SolA AC-capacity and S_99 metadata."""
    required_columns = {"ac_capacity_kw", "s_99"}
    missing_columns = required_columns.difference(site_details.columns)
    if missing_columns:
        raise ValueError(
            "Solar Analytics site metadata is missing capacity columns: "
            f"{sorted(missing_columns)}"
        )

    site_details = site_details.with_columns(
        [
            pl.col("ac_capacity_kw").cast(pl.Float64, strict=False),
            pl.col("s_99").cast(pl.Float64, strict=False),
        ]
    )
    valid_ac_capacity = (
        pl.col("ac_capacity_kw").is_not_null()
        & pl.col("ac_capacity_kw").is_finite()
        & (pl.col("ac_capacity_kw") > 0)
    )
    valid_s_99 = (
        pl.col("s_99").is_not_null() & pl.col("s_99").is_finite() & (pl.col("s_99") > 0)
    )
    return site_details.with_columns(
        pl.when(valid_ac_capacity)
        .then(
            pl.when(valid_s_99 & (pl.col("s_99") > pl.col("ac_capacity_kw")))
            .then(pl.col("s_99"))
            .otherwise(pl.col("ac_capacity_kw"))
        )
        .otherwise(
            pl.when(valid_s_99)
            .then(pl.col("s_99"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
        )
        .alias("s_rated")
    )
