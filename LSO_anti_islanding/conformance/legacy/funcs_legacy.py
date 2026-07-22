"""Legacy combined data helpers retained for historical reference."""

import math
from pathlib import Path

import polars as pl

''' only add functions that reads from dataset and make small additons to the dataset
    do not add scripts to inspect data
'''

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Nov2022"

RAW_SITE_DATA_PATH = str(DATA_DIR / "ebm_1_20221112_20221119_data_processed_sa")
CLEANED_SITE_DATA_PATH = str(DATA_DIR / "ebm_1_20221112_20221119_data_cleaned_sa")
CIRCUIT_DETAILS_PATH = str(DATA_DIR / "ebm_1_20221112_20221119_circuit_details.csv")

def convertPowerToKw(allData, convert=False):
    allData = allData.with_columns((pl.col("power") / (100*1000)).alias("power"))
    print("Converting power to KW")
    return allData

def addLocalTStamp(ldf, add = False):
    # adds local tstamp for easy filtering
    # Add SA local time column without touching the original
    if add == True:
        ldf = ldf.with_columns(
                pl.col("utc_tstamp")
                .str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S%.f') # parse strings → Datetime
                .dt.replace_time_zone("UTC")                            # mark as UTC
                .dt.convert_time_zone("Australia/Adelaide")             # convert to SA local (DST-aware)
                .alias("local_tstamp")
            )
    return ldf
           
def addValidVoltage(df, Vmin = 80, Vmax = 300):
    "add a valid voltage column based on different voltage values"
    df = (df.with_columns(
            pl.col("voltage").cast(pl.Float64, strict=False)
            .alias("voltage_f") # convert str voltage to float
        )
        .with_columns(
            pl.when(
                pl.col("voltage_f").is_not_null() &
                pl.col("voltage_f").is_between(Vmin, Vmax)
            )
            .then(pl.col("voltage_f"))
            .when(
                pl.col("vmean").is_not_null() &
                pl.col("vmean").is_between(Vmin, Vmax)
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
        df.join(polarity_lookup.lazy(),on="c_id",how="left"
        )
        .with_columns((pl.col("power") * pl.col("polarity")).alias("power")
        ).drop("polarity")
        )
    return df


def buildCleanedSiteData(rawLocalPath=RAW_SITE_DATA_PATH,
                         circuitDetailsPath=CIRCUIT_DETAILS_PATH,
                         cleanedLocalPath=CLEANED_SITE_DATA_PATH):
    raw_parquet_path = Path(rawLocalPath + ".parquet")
    if not raw_parquet_path.exists():
        raise FileNotFoundError(
            f"Missing raw processed site data at {raw_parquet_path}."
        )

    circuit_details_path = Path(circuitDetailsPath)
    if not circuit_details_path.exists():
        raise FileNotFoundError(
            f"Missing circuit details at {circuit_details_path}."
        )

    circuitDetails = pl.read_csv(circuit_details_path)
    allData = loadSiteData(rawLocalPath, None, csv=False, convert=False)
    allData = convertPowerToKw(allData, convert=True)
    allData = addLocalTStamp(allData, add=True)
    allData = addValidVoltage(allData)
    allData = addPolarityToPower(allData, circuitDetails)

    cleaned_parquet_path = Path(cleanedLocalPath + ".parquet")
    cleaned_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    allData.sink_parquet(cleaned_parquet_path, compression="zstd")
    return cleaned_parquet_path


def loadCleanedSiteData(cleanedLocalPath=CLEANED_SITE_DATA_PATH):
    cleaned_parquet_path = Path(cleanedLocalPath + ".parquet")
    if not cleaned_parquet_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned site data at {cleaned_parquet_path}. "
            "Run data_processing.py first."
        )
    return pl.scan_parquet(cleaned_parquet_path)

def _round_up_to_half_kw(value_kw):
    return math.ceil(value_kw * 2.0) / 2.0


def _metadata_capacity_kw(siteDetails, siteNumber):
    site_row = siteDetails.filter(pl.col("site_id") == siteNumber).select("ac_cap_w")
    if site_row.is_empty():
        return None

    ac_cap_w = site_row["ac_cap_w"][0]
    if ac_cap_w is None:
        return None

    try:
        ac_cap_kw = float(ac_cap_w) / 1000.0
    except (TypeError, ValueError):
        return None

    if ac_cap_kw <= 0:
        return None
    return ac_cap_kw


def _robust_observed_peak_kw(day_behaviours):
    site_power_frames = []
    for day_info in day_behaviours or []:
        behaviour = day_info.get("behaviour")
        if behaviour is None:
            continue

        df = behaviour.circuitData
        power_cols = [
            c for c in df.columns
            if c.startswith("power")
            and not c.endswith("_next")
            and not c.endswith("_logic")
        ]
        if not power_cols:
            continue

        site_power_frames.append(
            df.select(
                pl.sum_horizontal(
                    [pl.col(c).cast(pl.Float64, strict=False).fill_null(0).clip(lower_bound=0) for c in power_cols]
                ).alias("site_power_kw")
            )
        )

    if not site_power_frames:
        return None, None

    site_power = pl.concat(site_power_frames, how="vertical").filter(pl.col("site_power_kw") > 0)
    if site_power.is_empty():
        return None, None

    sample_count = site_power.height
    top_n = min(sample_count, max(20, math.ceil(sample_count * 0.01)))
    top_slice = site_power.sort("site_power_kw", descending=True).head(top_n)

    robust_peak_kw = top_slice.select(pl.col("site_power_kw").median()).item()
    raw_max_kw = site_power.select(pl.col("site_power_kw").max()).item()
    return robust_peak_kw, raw_max_kw


def ratedCapacityOfPV(siteDetails, siteNumber, day_behaviours=None,
                      metadata_tolerance=1.10, fallback_kw=5.0):
    metadata_kw = _metadata_capacity_kw(siteDetails, siteNumber)
    robust_peak_kw, _ = _robust_observed_peak_kw(day_behaviours)

    if metadata_kw is not None:
        if robust_peak_kw is None or robust_peak_kw <= (metadata_kw * metadata_tolerance):
            return metadata_kw
        return _round_up_to_half_kw(robust_peak_kw)

    if robust_peak_kw is not None:
        return _round_up_to_half_kw(robust_peak_kw)

    return fallback_kw

def getPVDataFromCircuit(circuitNumber, df, 
                                       start=None, end=None):
    ''' 
    assumes some functions on df has already been ran, see main
    return circuit data filtered on circuitID
    '''

    circuitData = df.filter(pl.col('c_id') == circuitNumber)
    # Sort by timestamp
    circuitData = circuitData.sort('local_tstamp').with_row_index("row_id")

    # filter for specific day
    if start is not None and end is not None:
        # print('Filtering data for site number: {} and day: {} to {}'.format(circuitNumber, start, end))
        filtered = circuitData.filter((pl.col("local_tstamp") >= start) &(pl.col("local_tstamp") <= end))
        filtered = filtered.collect() # load the filtered data in the memory
        return filtered

    return circuitData.collect()

def getPVCircuitDataUsingSite(siteNumber, allSites, allSites2, df):
    ''' 
        assumes some functions on df has already been ran, see main
        this could be improved to accomodate parquet 
        probs also dont need it in the future
        just gotta check this function if you start using it again
        it returns data for a PV circuit using site number as the identifier
        with its rated capacity
    '''

    circuitIDs = allSites2.filter(pl.col('site_id') == siteNumber).select('c_id', 'con_type')
    # get data for the filtered circuit level data and any battery or solar installed

    # filter by pv data
    try:
        circuitID   = circuitIDs.filter(pl.col('con_type')=='pv_site_net')['c_id'][0] # take the first one if multiple are available
    except IndexError:
        print("NO PV SYSTEM")
        return None, None
    circuitData = df.filter(pl.col('c_id')==circuitID) # just pick the first id
    PRated      = allSites.filter(pl.col("site_id")==siteNumber)["ac_cap_w"][0]/1000
    del df # delete data to create space

    # Sort by timestamp
    circuitData = circuitData.sort('local_tstamp').with_row_index("row_id")
    return circuitData, PRated

def mapCircuitDataToSite(siteDayLong, siteNumber):
    index_columns = ["local_tstamp", "utc_tstamp"]
    select_columns = [
        "c_id",
        "local_tstamp",
        "utc_tstamp",
        "power",
        "voltage_valid",
    ]
    if "duration" in siteDayLong.columns:
        index_columns.append("duration")
        select_columns.insert(3, "duration")

    analysis_long = siteDayLong.select(select_columns)
    if analysis_long.is_empty():
        return (
            analysis_long
            .select([c for c in index_columns if c != "duration"])
            .with_row_index("row_id")
            .with_columns(pl.lit(siteNumber).alias("site_id"))
        )

    wide = (
        analysis_long
        .pivot(
            values=["power", "voltage_valid"],
            index=index_columns,
            on="c_id",
        )
        .sort("local_tstamp")
        .drop("duration", strict=False)
        .with_row_index("row_id")
        .with_columns(pl.lit(siteNumber).alias("site_id"))
    )
    return wide

def loadSiteData(localPath, S3Path, csv = False, convert = False):
    ''' return/ save/ read the processed data preferably as parquet 
        localPath: local path where it stored
        S3Path   : Server path on S3
        csv.     : param if you want .csv format
        convert  : param if you want to convert it to parquet
    '''
    # Try local Parquet first
    if Path(localPath+".parquet").exists():
        fileName = localPath+'.parquet'
        # Use lazy scan locally too to avoid full-memory load if it’s big
        siteData = pl.scan_parquet(localPath+'.parquet') #.collect()
        return siteData
    # try csv then
    if Path(localPath+".csv").exists():
        fileName = localPath+'.csv'
        siteData = pl.read_csv(fileName)
        if csv == True:
            return siteData
        else: # convert to parquet and return that
            if convert == True: # if it needs to be converted
                siteData.write_parquet(localPath+'.parquet', compression="zstd")
                print('Converting to parquet')
                siteData = pl.scan_parquet(localPath+'.parquet').collect() # return parquet
                return siteData # is this parquet, confirm

    # Fall back to S3
    if S3Path is None:
        raise FileNotFoundError(
            f"Missing local site data at {localPath}.parquet or {localPath}.csv."
        )
    try:
        # Prefer lazy scan for large files to keep memory low
        return pl.scan_parquet(S3Path).collect()
    except FileNotFoundError as e:
        # The object/key truly doesn't exist
        raise FileNotFoundError(f"S3 object not found: {S3Path}") from e
