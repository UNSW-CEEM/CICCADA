"""Local connection to the private CICCADA Trino service."""

from contextlib import contextmanager
import json
import os
import signal
import socket
import subprocess
import time
from uuid import uuid4

import polars as pl

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


AWS_PROFILE = "ciccada"
AWS_REGION = "ap-southeast-2"
SSM_INSTANCE_ID = "i-0f5bc0dd90f8a58d1"
TRINO_HOST = "trino2.ciccada"
TRINO_PORT = 8080
LOCAL_PORT = 18080


@contextmanager
def local_trino_engine(
    catalog: str = "iceberg",
    schema: str | None = None,
):
    """Connect this laptop to Trino through a temporary SSM tunnel."""

    # 1. Check the cached AWS identity. Open the SSO login flow only when the
    #    cached session has expired.
    identity = subprocess.run(
        [
            "aws",
            "sts",
            "get-caller-identity",
            "--profile",
            AWS_PROFILE,
            "--region",
            AWS_REGION,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if identity.returncode != 0:
        subprocess.run(
            ["aws", "sso", "login", "--profile", AWS_PROFILE],
            check=True,
        )

    # 2. Build and start the same SSM port-forwarding command that would
    #    otherwise run in a terminal. Popen keeps it running in the background
    #    while this Python process uses Trino.
    parameters = json.dumps(
        {
            "host": [TRINO_HOST],
            "portNumber": [str(TRINO_PORT)],
            "localPortNumber": [str(LOCAL_PORT)],
        }
    )
    command = [
        "aws",
        "ssm",
        "start-session",
        "--target",
        SSM_INSTANCE_ID,
        "--document-name",
        "AWS-StartPortForwardingSessionToRemoteHost",
        "--parameters",
        parameters,
        "--profile",
        AWS_PROFILE,
        "--region",
        AWS_REGION,
    ]

    # Give the AWS CLI and its Session Manager child process their own process
    # group so both can be stopped together during cleanup.
    tunnel = subprocess.Popen(command, start_new_session=True)
    engine = None
    try:
        # 3. Wait until the local end of the tunnel accepts connections. This
        #    is more reliable than sleeping for an assumed number of seconds.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if tunnel.poll() is not None:
                raise RuntimeError(
                    f"SSM tunnel exited with code {tunnel.returncode}; "
                    "see the AWS output above for details."
                )

            try:
                with socket.create_connection(
                    ("127.0.0.1", LOCAL_PORT), timeout=1
                ):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise TimeoutError(f"SSM tunnel did not open on port {LOCAL_PORT}")

        # 4. Point SQLAlchemy at the local end of the tunnel. SELECT 1 is a
        #    harmless check that the tunnel reaches a working Trino server.
        engine = create_engine(
            f"trino://ubuntu@127.0.0.1:{LOCAL_PORT}/{catalog}/{schema}",
            pool_pre_ping=True,
        )
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        # 5. Make the verified engine available inside the caller's with block.
        yield engine

    finally:
        # 6. Release the SQLAlchemy connection pool and stop the background
        #    tunnel, including when setup or a query raises an exception.
        if engine is not None:
            engine.dispose()

        try:
            os.killpg(tunnel.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        if tunnel.poll() is None:
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(tunnel.pid, signal.SIGKILL)
                tunnel.wait()


def read_query_via_parquet(
    trino_engine: Engine,
    query: str,
) -> pl.DataFrame:
    """Transfer one Trino query result through short-lived Iceberg Parquet.
    transfers compressed columnar data, making large queries substantially faster
    
    The query is written to a uniquely named managed table in the staging
    schema. Polars reads its compressed data files directly from S3, and the
    staging table is always dropped before the fully collected DataFrame is
    returned to the caller.
    """
    query = query.strip().rstrip(";")
    if not query:
        raise ValueError("Query cannot be empty")

    table_name = f"evm_batch_{uuid4().hex[:12]}"
    table = f"iceberg.test_db.{table_name}"
    files_table = f'iceberg.test_db."{table_name}$files"'

    try:
        # Create a temporary table containing the filtered Parquet data.
        with trino_engine.connect() as connection:
            result = connection.execute(text(f"""
                CREATE TABLE {table}
                WITH (format = 'PARQUET')
                AS
                {query}
            """))
            if result.returns_rows:
                result.fetchall()

        # Retrieve the generated Parquet paths and expected row count.
        files = pl.read_database(
            query=(
                "SELECT file_path, record_count "
                f"FROM {files_table}"
            ),
            connection=trino_engine,
        )

        # Read Parquet directly from S3, or preserve the empty-table schema.
        if files.is_empty():
            expected_rows = 0
            data = pl.read_database(
                query=f"SELECT * FROM {table} WHERE FALSE",
                connection=trino_engine,
            )
        else:
            expected_rows = int(files.get_column("record_count").sum())
            data = (
                pl.scan_parquet(
                    files.get_column("file_path").to_list(),
                    credential_provider=pl.CredentialProviderAWS(
                        profile_name=AWS_PROFILE,
                        region_name=AWS_REGION,
                    ),
                    cache=False,
                )
                .collect(engine="streaming")
            )

        # Validate that every staged row was collected.
        if data.height != expected_rows:
            raise RuntimeError(
                "Staged Parquet row-count mismatch: "
                f"expected {expected_rows}, collected {data.height}"
            )
    finally:
        # Always remove this exact temporary table before returning.
        with trino_engine.connect() as connection:
            result = connection.execute(
                text(f"DROP TABLE IF EXISTS {table}")
            )
            if result.returns_rows:
                result.fetchall()

    return data


# some useful query functions
# defs helful to test individual queries
# but for quertying continiously better to use local_trino_engine()
def trino_sql(
    query: str,
    *,
    catalog: str, schema: str,
):
    with local_trino_engine(catalog=catalog,schema=schema,) as engine:
        return pl.read_database(query=query, connection=engine)

    # example: df = trino_sql(
    # "SELECT * FROM sites LIMIT 10", catalog="hive", schema="solar_analytics",)

# hive query
def hive_sql(query: str, schema = "solar_analytics"):
    return trino_sql(query, catalog="hive", schema=schema,)

# iceberg query
def iceberg_sql(query: str, schema = "solar_analytics_iceberg"):
    return trino_sql(query, catalog="iceberg", schema=schema)
