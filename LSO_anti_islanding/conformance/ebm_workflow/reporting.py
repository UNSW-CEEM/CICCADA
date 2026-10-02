"""EBM-owned result-table construction and threshold reporting plots."""

from pathlib import Path

import polars as pl
from ebm_workflow.plotting import (
    plot_site_threshold_distribution,
    plot_site_threshold_distribution_extremes,
)

SITE_CONFORMANCE_NAME = "site_conformance.csv"
SITE_CONFORMANCE_FINAL_TABLE_NAME = "site_conformance_final_table.csv"
SITE_CONFORMANCE_TIME_DISTRIBUTION_NAME = "site_conformance_time_distribution.csv"
SITE_CONFORMANCE_TOD_DISTRIBUTION_NAME = "site_conformance_tod_distribution.csv"
SITE_LEVEL_VARIOUS_VOLTAGES_NAME = "site_level_various_voltages.csv"
CONFORMANCE_EXCLUSIONS_NAME = "conformance_exclusions.csv"

SITE_CONFORMANCE_SCHEMA = {
    "site_id": pl.Int64,
    "threshold_method": pl.Utf8,
    "los_calculated_threshold_used": pl.Float64,
    "ov1_calculated_threshold_used": pl.Float64,
    "los_lowest_disconnect_voltage": pl.Float64,
    "ov1_lowest_disconnect_voltage": pl.Float64,
    "los_lowest_disconnect_threshold_used": pl.Float64,
    "ov1_lowest_disconnect_threshold_used": pl.Float64,
    "los_calculated_responsible_count": pl.Int64,
    "los_calculated_conformant_count": pl.Int64,
    "los_calculated_conformance_pct": pl.Float64,
    "los_calculated_pass": pl.Boolean,
    "ov1_calculated_responsible_count": pl.Int64,
    "ov1_calculated_conformant_count": pl.Int64,
    "ov1_calculated_conformance_pct": pl.Float64,
    "ov1_calculated_pass": pl.Boolean,
    "overall_calculated_responsible_count": pl.Int64,
    "overall_calculated_conformant_count": pl.Int64,
    "overall_calculated_conformance_pct": pl.Float64,
    "overall_calculated_pass": pl.Boolean,
    "los_disconnect_support_added_count": pl.Int64,
    "ov1_disconnect_support_added_count": pl.Int64,
    "los_disconnect_supported_responsible_count": pl.Int64,
    "los_disconnect_supported_conformant_count": pl.Int64,
    "los_disconnect_supported_conformance_pct": pl.Float64,
    "los_disconnect_supported_pass": pl.Boolean,
    "ov1_disconnect_supported_responsible_count": pl.Int64,
    "ov1_disconnect_supported_conformant_count": pl.Int64,
    "ov1_disconnect_supported_conformance_pct": pl.Float64,
    "ov1_disconnect_supported_pass": pl.Boolean,
    "overall_disconnect_supported_responsible_count": pl.Int64,
    "overall_disconnect_supported_conformant_count": pl.Int64,
    "overall_disconnect_supported_conformance_pct": pl.Float64,
    "overall_disconnect_supported_pass": pl.Boolean,
    "overall_conformance_category": pl.Utf8,
    "los_lowest_disconnect_responsible_count": pl.Int64,
    "los_lowest_disconnect_conformant_count": pl.Int64,
    "los_lowest_disconnect_conformance_pct": pl.Float64,
    "los_lowest_disconnect_pass": pl.Boolean,
    "ov1_lowest_disconnect_responsible_count": pl.Int64,
    "ov1_lowest_disconnect_conformant_count": pl.Int64,
    "ov1_lowest_disconnect_conformance_pct": pl.Float64,
    "ov1_lowest_disconnect_pass": pl.Boolean,
    "overall_lowest_disconnect_responsible_count": pl.Int64,
    "overall_lowest_disconnect_conformant_count": pl.Int64,
    "overall_lowest_disconnect_conformance_pct": pl.Float64,
    "overall_lowest_disconnect_pass": pl.Boolean,
}

SITE_CONFORMANCE_TIME_DISTRIBUTION_SCHEMA = {
    "site_id": pl.Int64,
    "threshold_method": pl.Utf8,
    "case": pl.Utf8,
    "eligible_timestamp_count": pl.Int64,
    "conformant_timestamp_count": pl.Int64,
    "disconnect_support_timestamp_count": pl.Int64,
    "non_conformant_timestamp_count": pl.Int64,
    "conformant_pct": pl.Float64,
    "non_conformant_pct": pl.Float64,
    "disconnected_below_threshold_count": pl.Int64,
    "disconnected_unknown_voltage_count": pl.Int64,
}

SITE_CONFORMANCE_TOD_DISTRIBUTION_SCHEMA = {
    "site_id": pl.Int64,
    "time_of_day_bin": pl.Utf8,
    "eligible_timestamp_count": pl.Int64,
    "eligible_threshold_timestamp_count": pl.Int64,
    "disconnect_support_timestamp_count": pl.Int64,
    "conformant_timestamp_count": pl.Int64,
    "non_conformant_timestamp_count": pl.Int64,
    "disconnected_below_threshold_count": pl.Int64,
    "disconnected_unknown_voltage_count": pl.Int64,
    "large_negative_power_timestamp_count": pl.Int64,
    "within_tolerance_negative_power_timestamp_count": pl.Int64,
    "positive_site_power_timestamp_count": pl.Int64,
    "zero_site_power_timestamp_count": pl.Int64,
}

CONFORMANCE_EXCLUSIONS_SCHEMA = {
    "site_id": pl.Int64,
    "exclusion_scope": pl.Utf8,
    "day": pl.Date,
    "reason": pl.Utf8,
    "common_power_v10m_coverage_pct": pl.Float64,
    "rows_common_power_v10m": pl.Int64,
    "rows_with_power": pl.Int64,
    "rows_with_v10m": pl.Int64,
    "covered_seconds": pl.Float64,
    "window_seconds": pl.Float64,
    "total_rows": pl.Int64,
    "coverage_threshold_pct": pl.Float64,
}

FINAL_TABLE_SCHEMA = {
    "Method Used": pl.Utf8,
    "Case": pl.Utf8,
    "Eligible Sites After Filtering": pl.UInt32,
    "Sites Assessed": pl.UInt32,
    "Unassessed Sites": pl.UInt32,
    "Conformant Sites": pl.UInt32,
    "Conformant (Erratic) Sites": pl.UInt32,
    "Non-Conformant Sites": pl.UInt32,
    "Total Conformant Sites": pl.UInt32,
    "Conformant Percentage (% of Assessed)": pl.Float64,
    "Conformant (Erratic) Percentage (% of Assessed)": pl.Float64,
    "Non-Conformant Percentage (% of Assessed)": pl.Float64,
    "Total Conformance Percentage (% of Assessed)": pl.Float64,
}


def build_site_conformance_table(
    phase_b_calculated,
    phase_b_disconnect_supported,
    phase_b_lowest_disconnect,
):
    """Combine the three Phase B cases into the EBM site-conformance schema."""
    calculated_source = phase_b_calculated["site_conformance"]
    disconnect_source = phase_b_disconnect_supported["site_conformance"]
    lowest_source = phase_b_lowest_disconnect["site_conformance"]
    if (
        calculated_source.is_empty()
        and disconnect_source.is_empty()
        and lowest_source.is_empty()
    ):
        return pl.DataFrame(schema=SITE_CONFORMANCE_SCHEMA)
    if (
        calculated_source.is_empty()
        or disconnect_source.is_empty()
        or lowest_source.is_empty()
    ):
        raise ValueError("All three EBM Phase B cases are required for reporting.")

    calculated = calculated_source.select(
        [
            "site_id",
            "threshold_method",
            pl.col("los_threshold_used").alias("los_calculated_threshold_used"),
            pl.col("ov1_threshold_used").alias("ov1_calculated_threshold_used"),
            "los_lowest_disconnect_voltage",
            "ov1_lowest_disconnect_voltage",
            pl.col("los_responsible_count").alias("los_calculated_responsible_count"),
            pl.col("los_conformant_count").alias("los_calculated_conformant_count"),
            pl.col("los_conformance_pct").alias("los_calculated_conformance_pct"),
            pl.col("los_pass").alias("los_calculated_pass"),
            pl.col("ov1_responsible_count").alias("ov1_calculated_responsible_count"),
            pl.col("ov1_conformant_count").alias("ov1_calculated_conformant_count"),
            pl.col("ov1_conformance_pct").alias("ov1_calculated_conformance_pct"),
            pl.col("ov1_pass").alias("ov1_calculated_pass"),
            pl.col("overall_responsible_count").alias(
                "overall_calculated_responsible_count"
            ),
            pl.col("overall_conformant_count").alias(
                "overall_calculated_conformant_count"
            ),
            pl.col("overall_conformance_pct").alias("overall_calculated_conformance_pct"),
            pl.col("overall_pass").alias("overall_calculated_pass"),
        ]
    )
    disconnect_supported = disconnect_source.select(
        [
            "site_id",
            "threshold_method",
            "los_disconnect_support_added_count",
            "ov1_disconnect_support_added_count",
            "los_disconnect_supported_responsible_count",
            "los_disconnect_supported_conformant_count",
            "los_disconnect_supported_conformance_pct",
            "los_disconnect_supported_pass",
            "ov1_disconnect_supported_responsible_count",
            "ov1_disconnect_supported_conformant_count",
            "ov1_disconnect_supported_conformance_pct",
            "ov1_disconnect_supported_pass",
            "overall_disconnect_supported_responsible_count",
            "overall_disconnect_supported_conformant_count",
            "overall_disconnect_supported_conformance_pct",
            "overall_disconnect_supported_pass",
        ]
    )
    lowest_disconnect = lowest_source.select(
        [
            "site_id",
            "threshold_method",
            pl.col("los_threshold_used").alias("los_lowest_disconnect_threshold_used"),
            pl.col("ov1_threshold_used").alias("ov1_lowest_disconnect_threshold_used"),
            pl.col("los_responsible_count").alias(
                "los_lowest_disconnect_responsible_count"
            ),
            pl.col("los_conformant_count").alias(
                "los_lowest_disconnect_conformant_count"
            ),
            pl.col("los_conformance_pct").alias("los_lowest_disconnect_conformance_pct"),
            pl.col("los_pass").alias("los_lowest_disconnect_pass"),
            pl.col("ov1_responsible_count").alias(
                "ov1_lowest_disconnect_responsible_count"
            ),
            pl.col("ov1_conformant_count").alias(
                "ov1_lowest_disconnect_conformant_count"
            ),
            pl.col("ov1_conformance_pct").alias("ov1_lowest_disconnect_conformance_pct"),
            pl.col("ov1_pass").alias("ov1_lowest_disconnect_pass"),
            pl.col("overall_responsible_count").alias(
                "overall_lowest_disconnect_responsible_count"
            ),
            pl.col("overall_conformant_count").alias(
                "overall_lowest_disconnect_conformant_count"
            ),
            pl.col("overall_conformance_pct").alias(
                "overall_lowest_disconnect_conformance_pct"
            ),
            pl.col("overall_pass").alias("overall_lowest_disconnect_pass"),
        ]
    )

    return (
        calculated.join(
            disconnect_supported,
            on=["site_id", "threshold_method"],
            how="inner",
            validate="1:1",
        )
        .join(
            lowest_disconnect,
            on=["site_id", "threshold_method"],
            how="inner",
            validate="1:1",
        )
        .with_columns(
            pl.when(
                pl.col("overall_disconnect_supported_pass").eq(True)
                & pl.col("overall_calculated_pass").eq(True)
            )
            .then(pl.lit("conformant"))
            .when(pl.col("overall_disconnect_supported_pass").eq(True))
            .then(pl.lit("conformant_erratic"))
            .when(pl.col("overall_disconnect_supported_pass").eq(False))
            .then(pl.lit("non_conformant"))
            .otherwise(pl.lit("unassessed"))
            .alias("overall_conformance_category")
        )
        .select(list(SITE_CONFORMANCE_SCHEMA))
        .cast(SITE_CONFORMANCE_SCHEMA, strict=False)
        .sort("site_id")
    )


def build_site_conformance_time_distribution(
    phase_b_calculated,
    phase_b_disconnect_supported,
    phase_b_lowest_disconnect,
):
    """Build the three-case per-site timestamp summary used by EBM reports."""
    calculated_source = phase_b_calculated["site_conformance"]
    disconnect_source = phase_b_disconnect_supported["site_conformance"]
    lowest_source = phase_b_lowest_disconnect["site_conformance"]
    if (
        calculated_source.is_empty()
        and disconnect_source.is_empty()
        and lowest_source.is_empty()
    ):
        return pl.DataFrame(schema=SITE_CONFORMANCE_TIME_DISTRIBUTION_SCHEMA)
    if (
        calculated_source.is_empty()
        or disconnect_source.is_empty()
        or lowest_source.is_empty()
    ):
        raise ValueError("All three EBM Phase B cases are required for reporting.")

    calculated = calculated_source.select(
        [
            "site_id",
            "threshold_method",
            pl.lit("calculated").alias("case"),
            pl.col("overall_responsible_count").alias("eligible_timestamp_count"),
            pl.col("overall_conformant_count").alias("conformant_timestamp_count"),
            pl.lit(0, dtype=pl.Int64).alias("disconnect_support_timestamp_count"),
            (
                pl.col("overall_responsible_count") - pl.col("overall_conformant_count")
            ).alias("non_conformant_timestamp_count"),
            pl.col("overall_conformance_pct").alias("conformant_pct"),
            (100.0 - pl.col("overall_conformance_pct")).alias("non_conformant_pct"),
            "disconnected_below_threshold_count",
            "disconnected_unknown_voltage_count",
        ]
    )
    disconnect_supported = disconnect_source.select(
        [
            "site_id",
            "threshold_method",
            pl.lit("disconnect_supported").alias("case"),
            pl.col("overall_disconnect_supported_responsible_count").alias(
                "eligible_timestamp_count"
            ),
            pl.col("overall_disconnect_supported_conformant_count").alias(
                "conformant_timestamp_count"
            ),
            (
                pl.col("los_disconnect_support_added_count")
                + pl.col("ov1_disconnect_support_added_count")
            ).alias("disconnect_support_timestamp_count"),
            (
                pl.col("overall_disconnect_supported_responsible_count")
                - pl.col("overall_disconnect_supported_conformant_count")
            ).alias("non_conformant_timestamp_count"),
            pl.col("overall_disconnect_supported_conformance_pct").alias(
                "conformant_pct"
            ),
            (100.0 - pl.col("overall_disconnect_supported_conformance_pct")).alias(
                "non_conformant_pct"
            ),
            "disconnected_below_threshold_count",
            "disconnected_unknown_voltage_count",
        ]
    )
    lowest_disconnect = lowest_source.select(
        [
            "site_id",
            "threshold_method",
            pl.lit("lowest_disconnect").alias("case"),
            pl.col("overall_responsible_count").alias("eligible_timestamp_count"),
            pl.col("overall_conformant_count").alias("conformant_timestamp_count"),
            pl.lit(0, dtype=pl.Int64).alias("disconnect_support_timestamp_count"),
            (
                pl.col("overall_responsible_count") - pl.col("overall_conformant_count")
            ).alias("non_conformant_timestamp_count"),
            pl.col("overall_conformance_pct").alias("conformant_pct"),
            (100.0 - pl.col("overall_conformance_pct")).alias("non_conformant_pct"),
            "disconnected_below_threshold_count",
            "disconnected_unknown_voltage_count",
        ]
    )
    return (
        pl.concat([calculated, disconnect_supported, lowest_disconnect], how="vertical")
        .select(list(SITE_CONFORMANCE_TIME_DISTRIBUTION_SCHEMA))
        .cast(SITE_CONFORMANCE_TIME_DISTRIBUTION_SCHEMA, strict=False)
        .sort(["site_id", "case"])
    )


def build_ebm_site_conformance(results):
    """Validate and normalize the combined EBM site-conformance table."""
    site_conformance = results["site_conformance"]
    site_thresholds = results["site_thresholds"]
    if site_conformance.is_empty() and site_thresholds.is_empty():
        return pl.DataFrame(schema=SITE_CONFORMANCE_SCHEMA)
    if site_conformance.is_empty() or site_thresholds.is_empty():
        raise ValueError(
            "EBM conformance and threshold tables must contain the same sites."
        )

    combined_site_ids = site_conformance.select("site_id").join(
        site_thresholds.select("site_id"),
        on="site_id",
        how="inner",
        validate="1:1",
    )
    if (
        combined_site_ids.height != site_conformance.height
        or combined_site_ids.height != site_thresholds.height
    ):
        raise ValueError("EBM conformance and threshold tables have different site IDs.")
    return (
        site_conformance.select(list(SITE_CONFORMANCE_SCHEMA))
        .cast(SITE_CONFORMANCE_SCHEMA, strict=False)
        .sort("site_id")
    )


def build_site_conformance_tod_distribution(timestamp_detail):
    """Aggregate one site's detail into right-closed five-minute TOD bins."""
    if timestamp_detail.is_empty():
        return pl.DataFrame(schema=SITE_CONFORMANCE_TOD_DISTRIBUTION_SCHEMA)
    return (
        timestamp_detail.with_columns(
            [
                (
                    (pl.col("local_tstamp") - pl.duration(microseconds=1)).dt.truncate(
                        "5m"
                    )
                    + pl.duration(minutes=5)
                )
                .dt.strftime("%H:%M")
                .alias("time_of_day_bin"),
                (pl.col("los_responsible") | pl.col("ov1_responsible")).alias(
                    "_eligible_threshold"
                ),
                (
                    pl.col("los_disconnect_support_added")
                    | pl.col("ov1_disconnect_support_added")
                ).alias("_disconnect_support"),
                (pl.col("los_conformant") | pl.col("ov1_conformant")).alias(
                    "_base_conformant"
                ),
            ]
        )
        .group_by(["site_id", "time_of_day_bin"])
        .agg(
            [
                (
                    pl.col("_eligible_threshold").sum()
                    + pl.col("_disconnect_support").sum()
                )
                .cast(pl.Int64)
                .alias("eligible_timestamp_count"),
                pl.col("_eligible_threshold")
                .sum()
                .cast(pl.Int64)
                .alias("eligible_threshold_timestamp_count"),
                pl.col("_disconnect_support")
                .sum()
                .cast(pl.Int64)
                .alias("disconnect_support_timestamp_count"),
                (pl.col("_base_conformant").sum() + pl.col("_disconnect_support").sum())
                .cast(pl.Int64)
                .alias("conformant_timestamp_count"),
                (pl.col("_eligible_threshold").sum() - pl.col("_base_conformant").sum())
                .cast(pl.Int64)
                .alias("non_conformant_timestamp_count"),
                pl.col("disconnected_below_threshold")
                .sum()
                .cast(pl.Int64)
                .alias("disconnected_below_threshold_count"),
                pl.col("disconnected_unknown_voltage")
                .sum()
                .cast(pl.Int64)
                .alias("disconnected_unknown_voltage_count"),
                pl.col("large_negative_power_timestamp")
                .sum()
                .cast(pl.Int64)
                .alias("large_negative_power_timestamp_count"),
                pl.col("within_tolerance_negative_power_timestamp")
                .sum()
                .cast(pl.Int64)
                .alias("within_tolerance_negative_power_timestamp_count"),
                pl.col("positive_site_power_timestamp")
                .sum()
                .cast(pl.Int64)
                .alias("positive_site_power_timestamp_count"),
                pl.col("zero_site_power_timestamp")
                .sum()
                .cast(pl.Int64)
                .alias("zero_site_power_timestamp_count"),
            ]
        )
        .select(list(SITE_CONFORMANCE_TOD_DISTRIBUTION_SCHEMA))
        .cast(SITE_CONFORMANCE_TOD_DISTRIBUTION_SCHEMA, strict=False)
        .sort(["site_id", "time_of_day_bin"])
    )


def add_disconnect_voltage_lists(site_summary, phase_a_records):
    """Add serialized LOS and OV1 disconnect-voltage lists by site."""
    list_columns = ["all_los_disconnect_voltages", "all_ov1_disconnect_voltages"]
    if phase_a_records.is_empty():
        return site_summary.with_columns(
            [pl.lit("[]").alias(column) for column in list_columns]
        )
    voltage_lists = (
        phase_a_records.group_by("site_id")
        .agg(
            pl.col("v10m_disc")
            .filter(pl.col("mechanism") == "LOS")
            .drop_nulls()
            .sort()
            .alias(list_columns[0]),
            pl.col("vinst_disc")
            .filter(pl.col("mechanism") == "OV1")
            .drop_nulls()
            .sort()
            .alias(list_columns[1]),
        )
        .with_columns(
            [
                pl.format(
                    "[{}]",
                    pl.col(column)
                    .list.eval(pl.element().cast(pl.Utf8))
                    .list.join(", "),
                ).alias(column)
                for column in list_columns
            ]
        )
    )
    return site_summary.join(voltage_lists, on="site_id", how="left").with_columns(
        pl.col(list_columns).fill_null("[]")
    )


def build_method_conformance_final_table(site_conformance):
    """Summarize assessed and conformant EBM sites for all three cases."""
    if site_conformance.is_empty():
        return pl.DataFrame(schema=FINAL_TABLE_SCHEMA)

    def case_summary(pass_column, case_name, *, split_erratic=False):
        expressions = [
            pl.col("site_id").n_unique().alias("Eligible Sites After Filtering"),
            pl.col(pass_column).is_not_null().sum().alias("Sites Assessed"),
            pl.col(pass_column).is_null().sum().alias("Unassessed Sites"),
        ]
        if split_erratic:
            expressions.extend(
                [
                    pl.col("overall_conformance_category")
                    .eq("conformant")
                    .sum()
                    .alias("Conformant Sites"),
                    pl.col("overall_conformance_category")
                    .eq("conformant_erratic")
                    .sum()
                    .alias("Conformant (Erratic) Sites"),
                ]
            )
        else:
            expressions.extend(
                [
                    pl.col(pass_column)
                    .eq(True)
                    .fill_null(False)
                    .sum()
                    .alias("Conformant Sites"),
                    pl.lit(0, dtype=pl.UInt32).alias("Conformant (Erratic) Sites"),
                ]
            )
        expressions.append(
            pl.col(pass_column)
            .eq(False)
            .fill_null(False)
            .sum()
            .alias("Non-Conformant Sites")
        )
        return (
            site_conformance.group_by("threshold_method", maintain_order=True)
            .agg(expressions)
            .with_columns(pl.lit(case_name).alias("Case"))
        )

    final_table = (
        pl.concat(
            [
                case_summary("overall_calculated_pass", "calculated"),
                case_summary(
                    "overall_disconnect_supported_pass",
                    "disconnect_supported",
                    split_erratic=True,
                ),
                case_summary("overall_lowest_disconnect_pass", "lowest_disconnect"),
            ]
        )
        .with_columns(
            (pl.col("Conformant Sites") + pl.col("Conformant (Erratic) Sites")).alias(
                "Total Conformant Sites"
            )
        )
        .with_columns(
            [
                (pl.col("Conformant Sites") / pl.col("Sites Assessed") * 100.0)
                .round(2)
                .alias("Conformant Percentage (% of Assessed)"),
                (
                    pl.col("Conformant (Erratic) Sites")
                    / pl.col("Sites Assessed")
                    * 100.0
                )
                .round(2)
                .alias("Conformant (Erratic) Percentage (% of Assessed)"),
                (pl.col("Non-Conformant Sites") / pl.col("Sites Assessed") * 100.0)
                .round(2)
                .alias("Non-Conformant Percentage (% of Assessed)"),
                (pl.col("Total Conformant Sites") / pl.col("Sites Assessed") * 100.0)
                .round(2)
                .alias("Total Conformance Percentage (% of Assessed)"),
            ]
        )
        .rename({"threshold_method": "Method Used"})
        .select(list(FINAL_TABLE_SCHEMA))
        .cast(FINAL_TABLE_SCHEMA, strict=False)
    )
    return final_table


def build_ebm_conformance_exclusions(results):
    """Build site-only EBM exclusions; EBM does not filter individual days."""
    rows = [
        {
            "site_id": site_id,
            "exclusion_scope": "site",
            "day": None,
            "reason": reason,
        }
        for reason, site_ids in results["skipped_sites"].items()
        for site_id in site_ids
    ]
    if not rows:
        return pl.DataFrame(schema=CONFORMANCE_EXCLUSIONS_SCHEMA)
    return pl.DataFrame(
        [
            {column: row.get(column) for column in CONFORMANCE_EXCLUSIONS_SCHEMA}
            for row in rows
        ],
        schema=CONFORMANCE_EXCLUSIONS_SCHEMA,
        strict=False,
    ).sort(["site_id", "exclusion_scope", "day"], nulls_last=True)


def write_ebm_threshold_distribution_plots(phase_a, output_dir):
    """Write the EBM LOS and OV1 threshold-distribution plots."""
    if phase_a.is_empty():
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mechanism, voltage_column, prefix in (
        ("LOS", "v10m_disc", "los"),
        ("OV1", "vinst_disc", "ov1"),
    ):
        stats = (
            phase_a.filter(pl.col("mechanism") == mechanism)
            .group_by("site_id")
            .agg(
                [
                    pl.col(voltage_column).min().alias("min_v"),
                    pl.col(voltage_column).median().alias("median_v"),
                    pl.col(voltage_column).max().alias("max_v"),
                    pl.col(voltage_column).std().alias("std_v"),
                    pl.len().alias("n_events"),
                ]
            )
        )
        plot_site_threshold_distribution(
            stats,
            title=(
                f"{mechanism} Thresholds Across Assessed Sites — "
                "Min / Median / Max (Std on right)"
            ),
            save_path=output_dir / f"{prefix}_threshold_distribution.png",
        )
        plot_site_threshold_distribution_extremes(
            stats,
            title=f"{mechanism} Thresholds — Lowest 20 Std Sites (n_events >= 3)",
            save_path=output_dir / f"{prefix}_threshold_lowest20_std.png",
            highest_std=False,
            min_events=3,
            n_sites=20,
        )
        plot_site_threshold_distribution_extremes(
            stats,
            title=f"{mechanism} Thresholds — Highest 20 Std Sites (n_events >= 3)",
            save_path=output_dir / f"{prefix}_threshold_highest20_std.png",
            highest_std=True,
            min_events=3,
            n_sites=20,
        )
