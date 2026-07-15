"""
Serverless access to the CICCADA data lake.
------------
A drop-in replacement for trino_config.py / duckdb_config.py for everyday querying. 
Reads the same AWS Glue tables, but through Amazon Athena (serverless SQL) instead of a Trino cluster 

REQUIREMENTS
------------
1. Logged in:           aws sso login --profile ciccada
2. Terminal knows the profile:
       PowerShell:  $env:AWS_PROFILE = "ciccada"
       (or set AWS_PROFILE in your VSCode launch / .env)
3. Packages installed in the `ciccada` env:
       awswrangler  boto3  pandas  pyarrow

USAGE
-----
    from aws_config import aq, tables, databases

    databases()                       # list Glue databases
    tables("solar_analytics")         # list tables in a database
    df = aq("SELECT * FROM circuits LIMIT 5")          # query -> DataFrame
    df = aq("SELECT * FROM sola_ts4 LIMIT 5",
            database="solar_analytics_iceberg")
"""

import os
import boto3
import pandas as pd
import awswrangler as wr

# AWS session
# We name the SSO profile explicitly so this works inside a VSCode/Jupyter
# kernel, which does NOT inherit the AWS_PROFILE from a terminal window.
# If you set AWS_PROFILE system-wide it is honoured. 
# Otherwise, default to the profile you created with `aws configure sso` (named "ciccada").

# If you get ProfileNotFound, run `aws configure list-profiles` in a terminal
# and put the real name below. If you get a token/expired error, just re-run
# `aws sso login --profile ciccada`.
REGION = "ap-southeast-2"
PROFILE = os.environ.get("AWS_PROFILE", "ciccada")
session = boto3.Session(profile_name=PROFILE, region_name=REGION)

# Athena settings
# Athena must stage its query results somewhere in S3. 
# Any writable prefix in the project bucket works.
# This folder is created automatically on first use.
BUCKET = "project-ciccada"
ATHENA_OUTPUT = f"s3://{BUCKET}/athena-results/"   # where Athena stages results
DEFAULT_DB = "solar_analytics"                     # default Glue database

# ===========================================================================
# CATALOG + ATHENA (SQL over registered tables)
# ===========================================================================
def databases() -> pd.DataFrame:
    """List all Glue databases in the account."""
    return wr.catalog.databases(boto3_session=session)
 
 
def tables(database: str = DEFAULT_DB) -> pd.DataFrame:
    """List the tables in a Glue database."""
    return wr.catalog.tables(database=database, boto3_session=session)
 
 
def aq(sql: str, database: str = DEFAULT_DB) -> pd.DataFrame:
    """Run an Athena SQL query and return a pandas DataFrame ('Athena query').
 
    Cost note: Athena bills by DATA SCANNED. 
    On the big telemetry table always filter on the partition columns (year, month, is_pv) 
    and name the columns you need instead of SELECT *. Dimension tables (circuits, sites) are tiny.
    """
    return wr.athena.read_sql_query(
        sql, database=database, s3_output=ATHENA_OUTPUT,
        boto3_session=session, ctas_approach=False,
    )
 
 
def describe(table: str, database: str = DEFAULT_DB) -> pd.DataFrame:
    """Show a table's columns and types. Scans no data (metadata only).
 
    Works for Iceberg tables too, where the Glue column list can be empty.
    """
    return aq(f"DESCRIBE {table}", database=database)
 
 
# ===========================================================================
# STORAGE (files in S3)
# ===========================================================================
def s3_ls(prefix: str = "") -> pd.DataFrame:
    """List folders and files directly under an S3 prefix in the project bucket.
 
    This is the 'ground truth' view -- it works even for data that nobody has
    registered in Glue yet. Use it to discover the layout of any new source.
 
        s3_ls()                       # top level of the bucket
        s3_ls("spark-warehouse/")     # inside a folder
    """
    s3 = session.client("s3")
    rows = []
    token = None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for p in resp.get("CommonPrefixes", []):
            rows.append({"type": "folder", "name": p["Prefix"], "size_mb": None})
        for o in resp.get("Contents", []):
            if o["Key"] == prefix:      # skip the folder placeholder object
                continue
            rows.append({"type": "file", "name": o["Key"],
                         "size_mb": round(o["Size"] / 1e6, 3)})
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return pd.DataFrame(rows)
 
 
# ===========================================================================
# DUCKDB (read a Parquet file directly -- no Glue, no Athena, runs locally)
# ===========================================================================
_duck = None
 
def _duck_con():
    """Create a DuckDB connection wired to S3 with your current creds."""
    global _duck
    import duckdb
    if _duck is None:
        _duck = duckdb.connect()
        _duck.sql("INSTALL httpfs; LOAD httpfs;")
    # Refresh credentials on every call -- SSO creds expire after ~1 hour.
    c = session.get_credentials().get_frozen_credentials()
    _duck.sql(f"""
        SET s3_region='{REGION}';
        SET s3_access_key_id='{c.access_key}';
        SET s3_secret_access_key='{c.secret_key}';
        SET s3_session_token='{c.token}';
    """)
    return _duck
 
 
def dread(s3_path: str, limit: int | None = 5) -> pd.DataFrame:
    """Read a Parquet file (or glob) straight from S3 into a DataFrame.
 
    Reads the file's OWN schema, so it sidesteps wrong/missing Glue definitions.
    Point it at a file or a wildcard:
 
        dread("s3://project/.../compliance_voltvar.parquet/*.parquet")
 
    Pass limit=None to read everything (careful with big tables).
    """
    q = f"SELECT * FROM read_parquet('{s3_path}')"
    if limit is not None:
        q += f" LIMIT {limit}"
    return _duck_con().sql(q).df()


# ---------------------------------------------------------------------------
# COST SAFETY 
# ---------------------------------------------------------------------------
# Athena charges by *data scanned*, not by rows returned (~AUD $8 per TB in Sydney). 
# The telemetry fact table is billions of rows. Two rules:
#
#   1. ALWAYS filter on the partition columns (year, month, and often is_pv)
#      so Athena only reads the relevant files. The colleague did exactly this
#      in every query: `WHERE is_pv = True AND year = 2025 AND month = 1`.
#
#   2. NEVER run `SELECT *` on the fact table without those filters, and avoid
#      it entirely when you only need a few columns. `SELECT circuit_id,
#      t_stamp, voltage, power` scans far less than `SELECT *`.
#
# Dimension tables (circuits, sites, partition_lookup) are tiny — query them
# freely.
#
# For very large pulls (e.g. a whole month of telemetry into pandas), switch
# the helper to `ctas_approach=True`, which writes results as Parquet and is
# much faster/cheaper to read back. Ask Claude when you reach that point.
