"""Shared measurement-cleaning functions used by data adapters."""

import polars as pl

from config import LOCAL_TIMEZONE, VALID_VOLTAGE_MAX, VALID_VOLTAGE_MIN


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


def deduplicateMeasurements(data, timestamp_column="utc_tstamp"):
    """Apply the verified SAPN duplicate policy to long-form measurements.

    One copy is retained when duplicate circuit-timestamps have identical power.
    Every copy is removed when their power values conflict.
    """
    key_columns = ["c_id", timestamp_column]
    conflicting_keys = (
        data.group_by(key_columns)
        .agg(pl.col("power").n_unique().alias("_power_n_unique"))
        .filter(pl.col("_power_n_unique") > 1)
        .select(key_columns)
    )
    return data.join(conflicting_keys, on=key_columns, how="anti").unique(
        subset=key_columns,
        keep="first",
        maintain_order=True,
    )


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
