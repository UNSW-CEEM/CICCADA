"""Direct connection to the CICCADA Trino service from EC2."""

import polars as pl
from sqlalchemy import create_engine, text

from .trino_config import TRINO_EC2_ICEBERG_URL


engine = create_engine(
    TRINO_EC2_ICEBERG_URL,
    pool_pre_ping=True,
)


def iceberg_sql(query: str) -> pl.DataFrame:
    return pl.read_database(query=query, connection=engine)


def iceberg_exec(query: str) -> None:
    with engine.connect() as connection:
        result = connection.execute(text(query))
        if result.returns_rows:
            result.fetchall()

    print("Executed")
