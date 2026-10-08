"""EBM rated-capacity policy."""

import polars as pl
from ebm_workflow.config import MAX_PV_SITE_NET_CIRCUITS
from ebm_workflow.site_preparation import (
    map_circuit_data_to_site,
    select_site_pv_data,
)


def rated_capacity_of_pv_ebm(
    site_details,
    site_number,
    aligned_site_data=None,
):
    """Return the metadata, calculated, and chosen EBM capacities for one site."""
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
                observed_kw = site_power.select(
                    pl.col("site_power_kw").quantile(
                        0.99,
                        interpolation="linear",
                    )
                ).item()

    if metadata_kw is None:
        chosen_kw = observed_kw
    elif observed_kw is None:
        chosen_kw = metadata_kw
    elif observed_kw <= metadata_kw * 1.04:
        chosen_kw = metadata_kw
    else:
        chosen_kw = observed_kw

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
    pv_site_counts,
    output_path,
):
    """Calculate and write the EBM rated-capacity CSV."""
    candidate_site_id_set = set(candidate_site_ids)
    capacity_rows = []
    for site_id in site_details["site_id"]:
        aligned_site_data = None
        pv_site_count = pv_site_counts.get(site_id, 0)
        if (
            site_id in candidate_site_id_set
            and 0 < pv_site_count <= MAX_PV_SITE_NET_CIRCUITS
        ):
            site_data = select_site_pv_data(
                all_data,
                circuit_details,
                site_id,
            )
            if not site_data.is_empty():
                aligned_site_data = map_circuit_data_to_site(site_data, site_id)
        capacity_rows.append(
            rated_capacity_of_pv_ebm(
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
