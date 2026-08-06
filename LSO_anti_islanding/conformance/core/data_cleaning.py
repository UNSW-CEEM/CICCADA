"""Shared measurement-cleaning functions used by data adapters."""

import polars as pl

from config import (
    DEDUPLICATION_ABSOLUTE_TOLERANCE_KW,
    DEDUPLICATION_RELATIVE_TOLERANCE,
    LOCAL_TIMEZONE,
    VALID_VOLTAGE_MAX,
    VALID_VOLTAGE_MIN,
)


SUPPORTED_LOCAL_TIMEZONES = (
    "Australia/Adelaide",
    "Australia/Brisbane",
    "Australia/Darwin",
    "Australia/Hobart",
    "Australia/Melbourne",
    "Australia/Perth",
    "Australia/Sydney",
)


def convertWToKw(data):
    """Convert watts to kilowatts."""
    return data.with_columns(
        (pl.col("power").cast(pl.Float64, strict=False) / 1_000.0).alias("power")
    )


def convertcWToKw(data):
    """Convert centiwatts to kilowatts."""
    return data.with_columns(
        (pl.col("power").cast(pl.Float64, strict=False) / 100_000.0).alias(
            "power"
        )
    )


def clipNegativePower(data):
    """Clip measured negative power to zero without replacing missing values.
       This is applied after implementing polarity
    """
    return data.with_columns(
        pl.when(pl.col("power") < 0)
        .then(pl.lit(0.0))
        .otherwise(pl.col("power"))
        .alias("power")
    )


def deduplicateMeasurements(data, timestamp_column="utc_tstamp"):
    """Apply the configured duplicate policy to long-form kW measurements.

    Duplicate circuit-timestamps within the configured absolute or relative
    power tolerance are represented by their mean power. Every copy is removed
    when the full power spread exceeds that tolerance.
    """
    key_columns = ["c_id", timestamp_column]
    data_columns = data.collect_schema().names()
    other_columns = [
        column
        for column in data_columns
        if column not in {*key_columns, "power"}
    ]

    grouped = data.group_by(
        key_columns,
        maintain_order=True,
    ).agg([
        pl.len().alias("_row_count"),
        pl.col("power").null_count().alias("_power_null_count"),
        pl.col("power").min().alias("_power_min"),
        pl.col("power").max().alias("_power_max"),
        pl.col("power").mean().alias("_power_mean"),
        *[pl.col(column).first().alias(column) for column in other_columns],
    ])

    power_spread = pl.col("_power_max") - pl.col("_power_min")
    maximum_absolute_power = pl.max_horizontal(
        pl.col("_power_min").abs(),
        pl.col("_power_max").abs(),
    )
    allowed_spread = pl.max_horizontal(
        pl.lit(DEDUPLICATION_ABSOLUTE_TOLERANCE_KW),
        maximum_absolute_power * DEDUPLICATION_RELATIVE_TOLERANCE,
    )
    single_row = pl.col("_row_count") == 1
    all_power_null = pl.col("_power_null_count") == pl.col("_row_count")
    numeric_values_within_tolerance = (
        (pl.col("_power_null_count") == 0)
        # Allow for representation error when a converted value lies exactly
        # on the configured tolerance boundary.
        & (power_spread <= allowed_spread + 1e-12)
    )

    retained_power = pl.col("_power_mean").alias("power")
    output_columns = [
        retained_power if column == "power" else pl.col(column)
        for column in data_columns
    ]
    return grouped.filter(
        single_row | all_power_null | numeric_values_within_tolerance
    ).select(output_columns)


def _utc_datetime_expression(data):
    utc_dtype = data.collect_schema()["utc_tstamp"]
    if utc_dtype == pl.String:
        return pl.col("utc_tstamp").str.to_datetime(
            time_zone="UTC",
            strict=False,
        )
    if isinstance(utc_dtype, pl.Datetime):
        utc_expression = pl.col("utc_tstamp")
        if utc_dtype.time_zone is None:
            return utc_expression.dt.replace_time_zone("UTC")
        return utc_expression.dt.convert_time_zone("UTC")
    raise TypeError(
        "utc_tstamp must be a string or datetime column, "
        f"not {utc_dtype!r}."
    )


def addLocalTStamp(data, timezone_column="timezone"):
    """Add canonical UTC and local wall-clock timestamps.

    ``timezone_column`` contains an IANA timezone for every measurement. Local
    timestamps are timezone-naive because a Polars column cannot mix multiple
    timezone-aware datetime types. The timezone column is retained alongside it.
    """
    if timezone_column not in data.collect_schema():
        raise ValueError(
            f"Missing required timezone column {timezone_column!r}."
        )

    utc_expression = _utc_datetime_expression(data)
    local_expression = None
    for timezone_name in SUPPORTED_LOCAL_TIMEZONES:
        converted = (
            utc_expression.dt.convert_time_zone(timezone_name)
            .dt.replace_time_zone(None)
        )
        condition = pl.col(timezone_column) == timezone_name
        if local_expression is None:
            local_expression = pl.when(condition).then(converted)
        else:
            local_expression = local_expression.when(condition).then(converted)

    local_expression = local_expression.otherwise(
        utc_expression.dt.convert_time_zone(LOCAL_TIMEZONE).dt.replace_time_zone(None)
    )
    return data.with_columns([
        utc_expression.alias("utc_tstamp"),
        local_expression.alias("local_tstamp"),
    ])


def addValidVoltage(
    df,
    Vmin=VALID_VOLTAGE_MIN,
    Vmax=VALID_VOLTAGE_MAX,
    *,
    fallback_col=None,
):
    """Add an in-range voltage using an explicitly configured fallback."""
    schema = df.collect_schema()
    if "voltage" not in schema:
        raise ValueError("Missing required voltage column 'voltage'.")
    if fallback_col is not None and fallback_col not in schema:
        raise ValueError(
            f"Missing configured fallback voltage column {fallback_col!r}."
        )

    df = df.with_columns(
        pl.col("voltage").cast(pl.Float64, strict=False).alias("voltage_f")
    )
    valid_voltage = (
        pl.when(
            pl.col("voltage_f").is_not_null()
            & pl.col("voltage_f").is_between(Vmin, Vmax)
        )
        .then(pl.col("voltage_f"))
    )
    if fallback_col is not None:
        fallback_voltage = pl.col(fallback_col).cast(pl.Float64, strict=False)
        valid_voltage = valid_voltage.when(
            fallback_voltage.is_not_null()
            & fallback_voltage.is_between(Vmin, Vmax)
        ).then(fallback_voltage)

    return df.with_columns(
        valid_voltage.otherwise(None).alias("voltage_valid")
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
