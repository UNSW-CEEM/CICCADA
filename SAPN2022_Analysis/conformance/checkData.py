import polars as pl

def checkDupes(df, highest = False):
    # Find duplicated timestamps (count > 1)
    dupes = (
        df.group_by("local_tstamp")
        .agg([
            pl.count().alias("n_rows"),
            pl.col("power").n_unique().alias("power_n_unique"),
            # pl.col("power").alias("power_values")  # optional: inspect raw values
        ])
        .filter(pl.col("n_rows") > 1)
    )

    # Quick summaries:
    nDupTimestamps    = dupes.height
    nDupWithSamePower = dupes.filter(pl.col("power_n_unique") == 1).height
    nDupWithDiffPower = dupes.filter(pl.col("power_n_unique") > 1).height

    if nDupWithDiffPower>0:
        if highest==True: # keep the highest value
            # print ("Conflicting power values at identical timestamps")        
            # print(dupes.filter(pl.col("power_n_unique") > 1))
            # POLICY: resolve conflicts by taking max(power)
            df = (
                df.group_by("local_tstamp")
                .agg([
                    pl.col("power").max().alias("power"),
                    pl.all().exclude("power").first(),
                ])
            )
        else: # remove that data
            bad_ts = (dupes
                .filter(pl.col("power_n_unique") > 1)
                .select("local_tstamp"))
            # Drop ALL rows with those timestamps
            df = df.join(bad_ts, on="local_tstamp", how="anti")
            # print(f"Dropping {nDupWithDiffPower} timestamps with conflicting power values")

     # filter the first if they are the same
    if nDupWithSamePower>0:
        df = df.unique(subset=["local_tstamp"], keep="first")
        # print(f"Deduplicated {nDupWithSamePower} timestamps with identical power values")
    
    if nDupWithDiffPower==0 and nDupWithSamePower == 0:
        pass
        # print("No duplciates!")
    df = df.sort('local_tstamp')
    return df
