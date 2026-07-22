"""Helpers used only by archived summary and plotting code."""

import polars as pl

# Helper: ensure a list column becomes [] instead of null so explode works safely
def ensure_list(col: str) -> pl.Expr:
    return pl.when(pl.col(col).is_null()).then(pl.lit([])).otherwise(pl.col(col))

# Prepare a base frame with event_date (local tz respected)
def add_event_date(df_events: pl.DataFrame) -> pl.DataFrame:
    return df_events.with_columns(
        pl.col("t_last_event").dt.date().alias("event_date")
    )

def split_nonmixed_groups(
    voltage_stats_site_day: pl.DataFrame,  # site_day out (must have pct_time_disconnected_during_events)
    site_summary_mixed: pl.DataFrame,      # contains 'site_id' of mixed sites
    T: float = 90.0
) -> pl.DataFrame:
    """
    Returns: DataFrame with columns -> site_id, nonmixed_status ∈ {"Always Compliant","Always Non-compliant"}.
    Excludes Mixed sites and 'No KPI' days.
    """
    kpi_day = (
        voltage_stats_site_day
        .with_columns([
            pl.when(pl.col("pct_time_disconnected_during_events").is_null())
              .then(pl.lit("No KPI"))
              .when(pl.col("pct_time_disconnected_during_events") >= T)
              .then(pl.lit("Compliant"))
              .otherwise(pl.lit("Non-compliant"))
              .alias("LABEL_DAY")
        ])
    )
    kpi_day_valid = kpi_day.filter(pl.col("LABEL_DAY") != "No KPI")

    mixed_ids = site_summary_mixed.select("site_id")
    nonmixed_ids = (
        kpi_day_valid.select("site_id").unique()
        .join(mixed_ids, on="site_id", how="anti")
    )

    nonmixed_status = (
        kpi_day_valid.join(nonmixed_ids, on="site_id", how="inner")
        .group_by("site_id")
        .agg([
            (pl.col("LABEL_DAY") == "Compliant").sum().alias("n_comp"),
            (pl.col("LABEL_DAY") == "Non-compliant").sum().alias("n_noncomp"),
        ])
        .with_columns([
            pl.when((pl.col("n_comp") > 0) & (pl.col("n_noncomp") == 0))
              .then(pl.lit("Always Compliant"))
              .when((pl.col("n_noncomp") > 0) & (pl.col("n_comp") == 0))
              .then(pl.lit("Always Non-compliant"))
              .otherwise(pl.lit(None))
              .alias("nonmixed_status")
        ])
        .filter(pl.col("nonmixed_status").is_not_null())
        .select(["site_id", "nonmixed_status"])
    )
    return nonmixed_status
