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
ATHENA_OUTPUT = "s3://project-ciccada/athena-results/"

# The Glue database most everyday queries will hit:
# Change if needed
DEFAULT_DB = "solar_analytics"


def aq(sql: str, database: str = DEFAULT_DB) -> pd.DataFrame:
    """Run an Athena SQL query and return a pandas DataFrame.

    `aq` = "Athena query". Example:
        aq("SELECT count(*) FROM circuits")
    """
    return wr.athena.read_sql_query(
        sql,
        database=database,
        s3_output=ATHENA_OUTPUT,
        boto3_session=session,
        ctas_approach=False,  
    )


def tables(database: str = DEFAULT_DB) -> pd.DataFrame:
    """List the tables in a Glue database (with their columns)."""
    return wr.catalog.tables(database=database, boto3_session=session)


def databases() -> pd.DataFrame:
    """List all Glue databases in the account."""
    return wr.catalog.databases(boto3_session=session)


def columns(table: str, database: str = DEFAULT_DB) -> pd.DataFrame:
    """Show the column names and types of a single table."""
    return wr.catalog.table(database=database, table=table,
                            boto3_session=session)


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
