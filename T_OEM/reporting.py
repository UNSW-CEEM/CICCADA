"""Site and cohort reporting for coordinated conformance results."""

import polars as pl


def summarise_site_conformance(
    site_conformance_results: pl.DataFrame,
) -> pl.DataFrame:
    """Return one row of conformance performance for one site."""
    return (
        site_conformance_results.group_by(["pseudonym", "solar_coupling"])
        .agg(
            pl.col("passive_anti_islanding_conformant")
            .is_not_null()
            .sum()
            .alias("passive_anti_islanding_num_eligible_timestamps"),
            pl.col("passive_anti_islanding_conformant")
            .fill_null(False)
            .sum()
            .alias("passive_anti_islanding_num_conformant_timestamps"),
            pl.col("sustained_operation_conformant")
            .is_not_null()
            .sum()
            .alias("sustained_operation_num_eligible_timestamps"),
            pl.col("sustained_operation_conformant")
            .fill_null(False)
            .sum()
            .alias("sustained_operation_num_conformant_timestamps"),
            pl.col("volt_watt_conformant")
            .is_not_null()
            .sum()
            .alias("volt_watt_num_eligible_timestamps"),
            pl.col("volt_watt_conformant")
            .fill_null(False)
            .sum()
            .alias("volt_watt_num_conformant_timestamps"),
            pl.col("csip_conformant")
            .is_not_null()
            .sum()
            .alias("csip_num_eligible_timestamps"),
            pl.col("csip_conformant")
            .fill_null(False)
            .sum()
            .alias("csip_num_conformant_timestamps"),
            pl.col("as_4777_conformant")
            .is_not_null()
            .sum()
            .alias("as_4777_num_eligible_timestamps"),
            pl.col("as_4777_conformant")
            .fill_null(False)
            .sum()
            .alias("as_4777_num_conformant_timestamps"),
        )
        .with_columns(
            pl.when(pl.col("passive_anti_islanding_num_eligible_timestamps") > 0)
            .then(
                100
                * pl.col("passive_anti_islanding_num_conformant_timestamps")
                / pl.col("passive_anti_islanding_num_eligible_timestamps")
            )
            .otherwise(None)
            .alias("passive_anti_islanding_conformant_percentage"),
            pl.when(pl.col("sustained_operation_num_eligible_timestamps") > 0)
            .then(
                100
                * pl.col("sustained_operation_num_conformant_timestamps")
                / pl.col("sustained_operation_num_eligible_timestamps")
            )
            .otherwise(None)
            .alias("sustained_operation_conformant_percentage"),
            pl.when(pl.col("volt_watt_num_eligible_timestamps") > 0)
            .then(
                100
                * pl.col("volt_watt_num_conformant_timestamps")
                / pl.col("volt_watt_num_eligible_timestamps")
            )
            .otherwise(None)
            .alias("volt_watt_conformant_percentage"),
            pl.when(pl.col("csip_num_eligible_timestamps") > 0)
            .then(
                100
                * pl.col("csip_num_conformant_timestamps")
                / pl.col("csip_num_eligible_timestamps")
            )
            .otherwise(None)
            .alias("csip_conformant_percentage"),
            pl.when(pl.col("as_4777_num_eligible_timestamps") > 0)
            .then(
                100
                * pl.col("as_4777_num_conformant_timestamps")
                / pl.col("as_4777_num_eligible_timestamps")
            )
            .otherwise(None)
            .alias("as_4777_conformant_percentage"),
        )
        .with_columns(
            pl.when(pl.col("as_4777_num_eligible_timestamps") >= 20)
            .then(pl.col("as_4777_conformant_percentage") >= 90)
            .otherwise(None)
            .alias("as_4777_site_conformant"),
            pl.when(pl.col("csip_num_eligible_timestamps") >= 20)
            .then(pl.col("csip_conformant_percentage") >= 90)
            .otherwise(None)
            .alias("csip_site_conformant"),
        )
        .select(
            "pseudonym",
            "solar_coupling",
            "passive_anti_islanding_num_eligible_timestamps",
            "passive_anti_islanding_num_conformant_timestamps",
            "passive_anti_islanding_conformant_percentage",
            "sustained_operation_num_eligible_timestamps",
            "sustained_operation_num_conformant_timestamps",
            "sustained_operation_conformant_percentage",
            "volt_watt_num_eligible_timestamps",
            "volt_watt_num_conformant_timestamps",
            "volt_watt_conformant_percentage",
            "csip_num_eligible_timestamps",
            "csip_num_conformant_timestamps",
            "csip_conformant_percentage",
            "as_4777_num_eligible_timestamps",
            "as_4777_conformant_percentage",
            "as_4777_site_conformant",
            "csip_site_conformant",
        )
    )


def summarise_cohort_conformance(
    site_conformance_performance: pl.DataFrame,
) -> pl.DataFrame:
    """Return mechanism-level conformance performance for a site cohort."""
    total_sites = site_conformance_performance.height

    return pl.DataFrame(
        {
            "assessment": [
                "Passive anti-islanding",
                "Limit for sustained operation",
                "Volt-Watt",
                "AS/NZS 4777 overall",
                "CSIP",
            ],
            "total_sites": [total_sites] * 5,
            "sites_with_sufficient_eligible_evidence": [
                site_conformance_performance.filter(
                    pl.col("passive_anti_islanding_num_eligible_timestamps") >= 20
                ).height,
                site_conformance_performance.filter(
                    pl.col("sustained_operation_num_eligible_timestamps") >= 20
                ).height,
                site_conformance_performance.filter(
                    pl.col("volt_watt_num_eligible_timestamps") >= 20
                ).height,
                site_conformance_performance.filter(
                    pl.col("as_4777_num_eligible_timestamps") >= 20
                ).height,
                site_conformance_performance.filter(
                    pl.col("csip_num_eligible_timestamps") >= 20
                ).height,
            ],
            "conformant_sites": [
                site_conformance_performance.filter(
                    (pl.col("passive_anti_islanding_num_eligible_timestamps") >= 20)
                    & (pl.col("passive_anti_islanding_conformant_percentage") >= 90)
                ).height,
                site_conformance_performance.filter(
                    (pl.col("sustained_operation_num_eligible_timestamps") >= 20)
                    & (pl.col("sustained_operation_conformant_percentage") >= 90)
                ).height,
                site_conformance_performance.filter(
                    (pl.col("volt_watt_num_eligible_timestamps") >= 20)
                    & (pl.col("volt_watt_conformant_percentage") >= 90)
                ).height,
                site_conformance_performance.filter(
                    (pl.col("as_4777_num_eligible_timestamps") >= 20)
                    & (pl.col("as_4777_conformant_percentage") >= 90)
                ).height,
                site_conformance_performance.filter(
                    (pl.col("csip_num_eligible_timestamps") >= 20)
                    & (pl.col("csip_conformant_percentage") >= 90)
                ).height,
            ],
        }
    ).with_columns(
        pl.when(pl.col("sites_with_sufficient_eligible_evidence") > 0)
        .then(
            100
            * pl.col("conformant_sites")
            / pl.col("sites_with_sufficient_eligible_evidence")
        )
        .otherwise(None)
        .alias("conformant_sites_percentage")
    )
