from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl


ADELAIDE_TZ = "Australia/Adelaide"
POWER_W_TO_KW = 1000
POWER_CW_TO_KW = 100 * POWER_W_TO_KW


def checkDupes(df, highest=False):
    dupes = (
        df.group_by("local_tstamp")
        .agg([
            pl.count().alias("n_rows"),
            pl.col("power").n_unique().alias("power_n_unique"),
        ])
        .filter(pl.col("n_rows") > 1)
    )

    nDupWithSamePower = dupes.filter(pl.col("power_n_unique") == 1).height
    nDupWithDiffPower = dupes.filter(pl.col("power_n_unique") > 1).height

    if nDupWithDiffPower > 0:
        if highest is True:
            df = (
                df.group_by("local_tstamp")
                .agg([
                    pl.col("power").max().alias("power"),
                    pl.all().exclude("power").first(),
                ])
            )
        else:
            bad_ts = (
                dupes
                .filter(pl.col("power_n_unique") > 1)
                .select("local_tstamp")
            )
            df = df.join(bad_ts, on="local_tstamp", how="anti")

    if nDupWithSamePower > 0:
        df = df.unique(subset=["local_tstamp"], keep="first")

    return df.sort("local_tstamp")


def convertPowerToKw(allData, convert=False):
    allData = allData.with_columns((pl.col("power") / POWER_CW_TO_KW).alias("power"))
    print("Converting power to KW")
    return allData


def addLocalTStamp(ldf, add=False):
    if add is True:
        ldf = ldf.with_columns(
            pl.col("utc_tstamp")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(ADELAIDE_TZ)
            .alias("local_tstamp")
        )
    return ldf


def addValidVoltage(df, Vmin=80, Vmax=300):
    df = (
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
    return df


def dedupeCircuitPolarity(circuitDetails):
    polarity_lookup = (
        circuitDetails
        .select(["c_id", "polarity"])
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
    df = (
        df.join(polarity_lookup.lazy(), on="c_id", how="left")
        .with_columns((pl.col("power") * pl.col("polarity")).alias("power"))
        .drop("polarity")
    )
    return df


def mapCircuitDataToSite(allData, circuitDetails, siteNumber, startDay, endDay):
    allCircuitsOnSite = circuitDetails.filter(pl.col("site_id") == siteNumber).unique()
    pvCircuitsDataOnSite = allCircuitsOnSite.filter(pl.col("con_type") == "pv_site_net")
    pvCircuitNosOnSite = list(pvCircuitsDataOnSite["c_id"])
    site_df = (
        allData
        .filter(pl.col("c_id").is_in(pvCircuitNosOnSite))
        .select([
            "c_id",
            "local_tstamp",
            "utc_tstamp",
            "duration",
            "power",
            "voltage_valid",
        ])
    )

    if startDay is not None and endDay is not None:
        site_df = site_df.filter(
            pl.col("local_tstamp").is_between(startDay, endDay, closed="both")
        ).collect()

    if site_df.is_empty():
        return 0, site_df, pvCircuitNosOnSite

    deDuped = site_df.group_by("c_id", maintain_order=True).map_groups(checkDupes)
    wide = (
        deDuped.pivot(
            values=["power", "voltage_valid"],
            index=["local_tstamp", "utc_tstamp", "duration"],
            on="c_id",
        )
    )
    wide = wide.sort("local_tstamp").with_row_index("row_id").with_columns(
        pl.lit(siteNumber).alias("site_id")
    )
    return 1, wide, pvCircuitNosOnSite


def empty_site_metrology_frame():
    return pl.DataFrame(
        schema={
            "dataset_role": pl.Utf8,
            "site_id": pl.Int64,
            "t_stamp": pl.Datetime,
            "P_kw": pl.Float64,
            "V": pl.Float64,
        }
    )


def _prepare_cleaned_metrology(raw, circuitDetails, power_divisor, start_date, end_date):
    cleaned = (
        raw.with_columns([
            pl.col("c_id").cast(pl.Int64),
            pl.col("utc_tstamp").cast(pl.Utf8),
            pl.col("power").cast(pl.Float64, strict=False),
            pl.col("voltage").cast(pl.Float64, strict=False),
            pl.col("vmean").cast(pl.Float64, strict=False),
            pl.col("duration").cast(pl.Int64, strict=False),
        ])
        .with_columns((pl.col("power") / power_divisor).alias("power"))
    )
    cleaned = addLocalTStamp(cleaned, add=True)
    cleaned = addValidVoltage(cleaned)
    cleaned = addPolarityToPower(cleaned, circuitDetails)

    if start_date is not None and end_date is not None:
        cleaned = cleaned.filter(
            pl.col("local_tstamp").dt.date().is_between(start_date, end_date, closed="both")
        )

    return cleaned.select([
        "c_id",
        "utc_tstamp",
        "duration",
        "power",
        "voltage_valid",
        "local_tstamp",
    ])


def _prepare_precleaned_metrology(cleaned, start_date, end_date):
    prepared = cleaned.with_columns([
        pl.col("c_id").cast(pl.Int64),
        pl.col("utc_tstamp").cast(pl.Utf8),
        pl.col("duration").cast(pl.Int64, strict=False),
        pl.col("power").cast(pl.Float64, strict=False),
        pl.col("voltage_valid").cast(pl.Float64, strict=False),
        pl.col("local_tstamp").cast(pl.Datetime(time_zone=ADELAIDE_TZ)),
    ])

    if start_date is not None and end_date is not None:
        prepared = prepared.filter(
            pl.col("local_tstamp").dt.date().is_between(start_date, end_date, closed="both")
        )

    return prepared.select([
        "c_id",
        "utc_tstamp",
        "duration",
        "power",
        "voltage_valid",
        "local_tstamp",
    ])


def prepare_sapn_metrology(sapn_data_path, sapn_circuits, start_date, end_date):
    # this could be processed data or not depends on the file you are passing
    sapn_data_path = Path(sapn_data_path)
    raw = pl.scan_parquet(sapn_data_path)

    if sapn_data_path.name.endswith("_data_cleaned_sa.parquet"):
        return _prepare_precleaned_metrology(raw, start_date, end_date)

    return _prepare_cleaned_metrology(
        raw,
        sapn_circuits,
        POWER_CW_TO_KW,
        start_date,
        end_date,
    )


def prepare_evm_metrology(evm_parquets, evm_circuits, start_date, end_date):
    if not evm_parquets:
        raise FileNotFoundError("No EVM training parquet files found for the selected date range")

    raw = pl.scan_parquet([str(path) for path in evm_parquets])
    return _prepare_cleaned_metrology(
        raw,
        evm_circuits,
        POWER_W_TO_KW,
        start_date,
        end_date,
    )


def _window_for_date_range(start_date, end_date):
    local_tz = ZoneInfo(ADELAIDE_TZ)
    start_day = datetime.combine(start_date, time.min, tzinfo=local_tz)
    end_day = datetime.combine(end_date, time.max, tzinfo=local_tz)
    return start_day, end_day


def _build_structured_site_rows(wide, dataset_role):
    power_cols = [
        col for col in wide.columns
        if col.startswith("power")
        and not col.endswith("_next")
        and not col.endswith("_logic")
    ]
    voltage_cols = [col for col in wide.columns if col.startswith("voltage_valid")]

    if not power_cols or not voltage_cols:
        return empty_site_metrology_frame()

    df = wide.clone()
    for col in voltage_cols:
        rolled_name = f"vmean_rolling_10m{col.replace('voltage_valid', '', 1)}"
        rolled = (
            df.filter(pl.col(col).is_not_null())
            .with_columns(
                pl.col(col).rolling_mean_by(
                    by="local_tstamp",
                    window_size="10m",
                ).alias(rolled_name)
            )
            .select(["local_tstamp", rolled_name])
        )
        df = df.join(rolled, on="local_tstamp", how="left")

    vmean_cols = [col for col in df.columns if col.startswith("vmean_rolling_10m")]
    if not vmean_cols:
        return empty_site_metrology_frame()

    df = df.with_columns([
        pl.mean_horizontal([pl.col(col) for col in voltage_cols]).alias("V"),
        pl.mean_horizontal([pl.col(col) for col in vmean_cols]).alias("v10m_avg"),
        pl.sum_horizontal([
            pl.when(pl.col(col).cast(pl.Float64, strict=False).fill_null(0) < 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col(col).cast(pl.Float64, strict=False).fill_null(0))
            for col in power_cols
        ]).alias("site_power_kw"),
        pl.any_horizontal([pl.col(col).is_not_null() for col in power_cols]).alias("_has_power"),
    ])

    return (
        df.filter(pl.col("_has_power") & pl.col("v10m_avg").is_not_null())
        .select([
            pl.lit(dataset_role).alias("dataset_role"),
            pl.col("site_id").cast(pl.Int64),
            pl.col("utc_tstamp")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f", strict=False)
            .alias("t_stamp"),
            pl.col("site_power_kw").cast(pl.Float64).alias("P_kw"),
            pl.col("V").cast(pl.Float64),
        ])
        .filter(pl.col("t_stamp").is_not_null())
        .sort("t_stamp")
    )


def aggregate_site_metrology(allData, circuitDetails, site_ids, start_date, end_date, dataset_role):
    start_day, end_day = _window_for_date_range(start_date, end_date)
    site_frames = []

    for site_id in sorted(set(site_ids)):
        has_data, wide, _ = mapCircuitDataToSite(
            allData,
            circuitDetails,
            site_id,
            start_day,
            end_day,
        )
        if not has_data:
            continue

        site_frame = _build_structured_site_rows(wide, dataset_role)
        if not site_frame.is_empty():
            site_frames.append(site_frame)

    if not site_frames:
        return empty_site_metrology_frame()

    return pl.concat(site_frames, how="vertical")
