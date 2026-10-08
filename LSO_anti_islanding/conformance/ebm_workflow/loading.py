"""EBM input loading."""

from pathlib import Path

import polars as pl

def load_ebm_site_details(site_details_path):
    """Load SAPN sites having exactly one source metadata row."""
    return (
        pl.read_csv(site_details_path)
        .filter(pl.col("inverter_count") == 1) # equivalent to 1 inverter
        .with_columns(
            (pl.col("ac_capacity_kw").cast(pl.Float64, strict=False)).alias(
                "capacity_kw"
            )
        )
    )

def load_ebm_circuit_details(circuit_details_path):
    return pl.read_csv(circuit_details_path).rename(
        {
            "circuit_id": "c_id",
            "circuit_type": "con_type",
            "circuit_polarity": "polarity",
        }
    )


def load_ebm_cleaned_data(cleaned_path):
    return pl.scan_parquet(Path(cleaned_path))
