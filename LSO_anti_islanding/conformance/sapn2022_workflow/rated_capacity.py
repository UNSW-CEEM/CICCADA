"""SAPN 2022 rated-capacity policy."""

import math

import polars as pl
from sapn2022_workflow.config import MAX_PV_SITE_NET_CIRCUITS
from sapn2022_workflow.site_preparation import (
    map_circuit_data_to_site,
    select_site_pv_data,
)


def rated_capacity_of_pv_sapn2022(
    site_details,
    site_number,
    aligned_site_data=None,
):
    """Return the metadata, calculated, and chosen SAPN capacities for one site."""
    # Read the site's valid AC capacity from metadata.
    metadata_kw = None
    site_row = site_details.filter(pl.col("site_id") == site_number).select(
        "capacity_kw"
    )
    if not site_row.is_empty() and site_row["capacity_kw"][0] is not None:
        try:
            capacity_kw = float(site_row["capacity_kw"][0])
            if capacity_kw > 0:
                metadata_kw = capacity_kw
        except (TypeError, ValueError):
            pass

    observed_kw = None
    if aligned_site_data is not None and not aligned_site_data.is_empty():
        power_cols = [
            column
            for column in aligned_site_data.columns
            if column.startswith("power")
            and not column.endswith("_next")
            and not column.endswith("_logic")
        ]
        if power_cols:
            # Sum complete, aligned multiphase readings from the cleaned data.
            complete_power = pl.all_horizontal(
                [pl.col(column).is_not_null() for column in power_cols]
            )
            site_power = (
                aligned_site_data.filter(complete_power)
                .select(
                    pl.sum_horizontal(
                        [
                            pl.col(column)
                            .cast(pl.Float64, strict=False)
                            .clip(lower_bound=0)
                            for column in power_cols
                        ]
                    ).alias("site_power_kw")
                )
                .filter(pl.col("site_power_kw") > 0)
            )
            if not site_power.is_empty():
                sample_count = site_power.height
                top_n = min(sample_count, max(20, math.ceil(sample_count * 0.01)))
                robust_peak_kw = (
                    site_power.sort("site_power_kw", descending=True)
                    .head(top_n)
                    .select(pl.col("site_power_kw").median())
                    .item()
                )
                observed_kw = math.ceil(robust_peak_kw * 10.0) / 10.0

    # Use the higher available estimate; do not invent a default capacity.
    if metadata_kw is None:
        chosen_kw = observed_kw
    elif observed_kw is None:
        chosen_kw = metadata_kw
    else:
        chosen_kw = max(metadata_kw, observed_kw)

    return {
        "site_id": site_number,
        "metadata_ac_capacity_kw": metadata_kw,
        "calculated_ac_capacity_kw": observed_kw,
        "chosen_ac_capacity_kw": chosen_kw,
    }


def generate_rated_capacity(
    site_details,
    circuit_details,
    all_data,
    candidate_site_ids,
    pv_site_net_counts,
    output_path,
):
    """Calculate and write the SAPN rated-capacity CSV."""
    candidate_site_id_set = set(candidate_site_ids)
    capacity_rows = []
    for site_id in site_details["site_id"]:
        aligned_site_data = None
        pv_site_net_count = pv_site_net_counts.get(site_id, 0)
        if (
            site_id in candidate_site_id_set
            and 0 < pv_site_net_count <= MAX_PV_SITE_NET_CIRCUITS
        ):
            site_data = select_site_pv_data(
                all_data,
                circuit_details,
                site_id,
            )
            if not site_data.is_empty():
                # Capacity uses every cleaned PV timestamp, without conformance prep.
                aligned_site_data = map_circuit_data_to_site(site_data, site_id)
        capacity_rows.append(
            rated_capacity_of_pv_sapn2022(
                site_details,
                site_id,
                aligned_site_data=aligned_site_data,
            )
        )

    pl.DataFrame(
        capacity_rows,
        schema={
            "site_id": pl.Int64,
            "metadata_ac_capacity_kw": pl.Float64,
            "calculated_ac_capacity_kw": pl.Float64,
            "chosen_ac_capacity_kw": pl.Float64,
        },
    ).write_csv(output_path)
