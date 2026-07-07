import polars as pl
from helperFuncs import ensure_list, add_event_date

def summarize_multi_method_site_outputs(
    phase_b_summary_by_method_df: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    if phase_b_summary_by_method_df is None or phase_b_summary_by_method_df.is_empty():
        empty = pl.DataFrame()
        return {"site_comparison": empty}

    comparison = (
        phase_b_summary_by_method_df
        .select("site_id")
        .unique()
        .sort("site_id")
    )
    pivot_specs = [
        "overall_pass",
        "los_pass",
        "ov1_pass",
        "los_eligible",
        "ov1_eligible",
        "los_compliance_pct",
        "ov1_compliance_pct",
        "los_threshold_used",
        "pass_basis",
    ]
    for value_col in pivot_specs:
        pivot = (
            phase_b_summary_by_method_df
            .select(["site_id", "method_key", value_col])
            .pivot(
                values=value_col,
                index="site_id",
                on="method_key",
                aggregate_function="first",
            )
        )
        rename_map = {
            col: f"{value_col}__{col}"
            for col in pivot.columns
            if col != "site_id"
        }
        comparison = comparison.join(pivot.rename(rename_map), on="site_id", how="left")

    overall_cols = sorted([c for c in comparison.columns if c.startswith("overall_pass__")])
    assessed_cols = overall_cols
    comparison = comparison.with_columns([
        pl.struct(overall_cols)
        .map_elements(
            lambda row: len(set(row.values())) > 1,
            return_dtype=pl.Boolean,
        )
        .alias("any_disagreement"),
        pl.struct(assessed_cols)
        .map_elements(
            lambda row: len({v for v in row.values() if v is not None}) > 1,
            return_dtype=pl.Boolean,
        )
        .alias("assessed_outcome_disagreement"),
        pl.struct(assessed_cols)
        .map_elements(
            lambda row: any(v is not None for v in row.values()),
            return_dtype=pl.Boolean,
        )
        .alias("any_method_assessed"),
        pl.struct(assessed_cols)
        .map_elements(
            lambda row: all(v is None for v in row.values()),
            return_dtype=pl.Boolean,
        )
        .alias("all_methods_unassessed"),
    ])
    return {"site_comparison": comparison}


def statsLos(complianceLOSResultAll, sitesWithOverVoltage, uniqueLosSites):
    # this percentage has repeat sites over multiple days
    # not returning it
    sitesCompliancePerc = complianceLOSResultAll.filter(pl.col("compliance_pct")>=90) \
            .height/sitesWithOverVoltage * 100 # can also use len(limitOfSustainencesites) as denominator
    sitesNonCompliancePerc = complianceLOSResultAll.filter(pl.col("compliance_pct")<90) \
            .height/sitesWithOverVoltage * 100 # or 100 - sitesCompliancePerc

    # individual site results
    resultIndividualSites = (
    complianceLOSResultAll.group_by("site_id").agg(pl.col("compliant").sum().alias("total_compliant"),
                                                pl.col("non_compliant").sum().alias("total_non_compliant"),
                                                pl.col("indeterminate").sum().alias("total_indeterminate"),
                                                pl.col("late_response").sum().alias("total_late_response"))
                            .with_columns(pl.sum_horizontal("total_compliant","total_non_compliant", 
                                                            "total_indeterminate", "total_late_response")
                                                            .alias("total"))
                                .with_columns([
                                (pl.col("total_compliant") / pl.col("total") * 100).alias("pct_compliant"),
                                (pl.col("total_non_compliant") / pl.col("total") * 100).alias("pct_non_compliant"),
                            ])
            )

    # individual sites result aggregated
    compliantSitesIndividualPct    = resultIndividualSites.filter(pl.col("pct_compliant")>=90
                                                                ).height/uniqueLosSites *100
    nonCompliantSitesIndividualPct = resultIndividualSites.filter(pl.col("pct_compliant")<90
                                                                ).height/uniqueLosSites *100

    # repeat entries of sites in the data
    repeatSites    = (complianceLOSResultAll.group_by("site_id").len().filter(pl.col("len") > 1))
    repeatSitesIds = repeatSites["site_id"].to_list()  # avoid deprecation
    numRepeatSites = repeatSites.height

    repeat_summary = (
        complianceLOSResultAll
        .filter(pl.col("site_id").is_in(repeatSitesIds))
        .group_by("site_id")
        .agg([
            pl.len().alias("n_entries"),
            (pl.col("compliance_pct") >= 90).sum().alias("n_compliant"),
            (pl.col("compliance_pct") < 90).sum().alias("n_non_compliant"),
        ])
        .with_columns([
            (pl.col("n_non_compliant") == 0).alias("all_compliant"),
            (pl.col("n_compliant") == 0).alias("all_non_compliant"),
            ((pl.col("n_compliant") > 0) & (pl.col("n_non_compliant") > 0)).alias("mixed")
        ])
    )

    summary_counts = repeat_summary.select([
        pl.col("all_compliant").sum().alias("sites_all_compliant"),
        pl.col("all_non_compliant").sum().alias("sites_all_non_compliant"),
        pl.col("mixed").sum().alias("sites_mixed"),
    ])
    return compliantSitesIndividualPct, nonCompliantSitesIndividualPct, summary_counts

def reconnectionStats(reconnectionLOSResultAll):
    # ============================================================
    # RECONNECTION ANALYSIS (Aggregated Across All LOS Runs)
    # ============================================================
    # I am calculating a bunch of them see which ones do you want
    # ------------------------------------------------------------
    # 1) Overall behavior distribution
    #    Shows how events ended:
    #    - Stayed Connected During Event
    #    - Reconnected During Event
    #    - Reconnected After Event
    #    - Never Reconnected
    # ------------------------------------------------------------
    behavior_summary = (
        reconnectionLOSResultAll
        .group_by("behavior_tag")
        .len()
        .with_columns(
            # Convert counts to ratios for easier interpretation
            (pl.col("len") / pl.col("len").sum()).alias("ratio")
        )
    )

    # ------------------------------------------------------------
    # 2) Focus only on events that reconnected AFTER the event
    #    These are the only ones with a meaningful reconnection_time
    # ------------------------------------------------------------
    reconn = (
        reconnectionLOSResultAll
        .filter(pl.col("behavior_tag") == "Reconnected After Event")
        .filter(pl.col("reconnection_time").is_not_null())
    )

    # ------------------------------------------------------------
    # 3) Core reconnection time statistics
    #    Median + p90 + p95 are more robust than the mean
    # ------------------------------------------------------------
    reconnection_stats = reconn.select([
        pl.len().alias("n_events"),
        pl.median("reconnection_time").alias("median_sec"),
        pl.quantile("reconnection_time", 0.9).alias("p90_sec"),
        pl.quantile("reconnection_time", 0.95).alias("p95_sec"),
        pl.mean("reconnection_time").alias("mean_sec"),
        pl.max("reconnection_time").alias("max_sec"),
    ])

    # Define thresholds in seconds
    range_stats = reconn.select([
        pl.len().alias("n_events"),
        # Very fast reconnection: 0–5s
        ((pl.col("reconnection_time") <= 5)).mean().alias("pct_0_5s"),
        # Quick reconnection: 5–60s
        ((pl.col("reconnection_time") > 5) &
        (pl.col("reconnection_time") <= 60)).mean().alias("pct_5_60s"),
        # Within 1–5 minutes: 60–300s
        ((pl.col("reconnection_time") > 60) &
        (pl.col("reconnection_time") <= 300)).mean().alias("pct_1_5min"),
        # Within 5–15 minutes: 300–900s
        ((pl.col("reconnection_time") > 300) &
        (pl.col("reconnection_time") <= 900)).mean().alias("pct_5_15min"),
        # Within 15–30 minutes: 900–1800s
        ((pl.col("reconnection_time") > 900) &
        (pl.col("reconnection_time") <= 1800)).mean().alias("pct_15_30min"),
        # Slow recovery: >30 minutes
        (pl.col("reconnection_time") > 1800).mean().alias("pct_gt_30min"),
    ])

    # ------------------------------------------------------------
    # 4) more than 60s check
    # ------------------------------------------------------------
    tail_check = reconn.select([
        (pl.col("reconnection_time") > 60).mean().alias("over_60s_ratio"),
        (pl.col("reconnection_time") > 300).mean().alias("over_5min_ratio"),
        (pl.col("reconnection_time") > 1800).mean().alias("over_30min_ratio"),
    ])

    # ------------------------------------------------------------
    # 5) CDF (Cumulative Distribution Function)
    #    Shows: What % reconnect within X seconds?
    #    This is usually the most informative visualization.
    # ------------------------------------------------------------
    cdf = (
        reconn
        .sort("reconnection_time")
        .with_row_index("rank")
        .with_columns(
            # Normalize rank to [0,1] to get cumulative probability
            (pl.col("rank") / pl.len()).alias("cdf")
        )
    )

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(8, 5))
    # plt.plot(cdf["reconnection_time"], cdf["cdf"])
    # plt.xlabel("Reconnection Time (seconds)")
    # plt.ylabel("Cumulative Probability")
    # plt.title("CDF of Reconnection Time")
    # plt.grid(True)
    # # plt.show()
    # plt.close()

    return behavior_summary, range_stats, tail_check

def summarize_never_reconnected_outputs(rr: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Summarize the outputs produced by analyze_never_reconnected_events for the NR cohort.
    Returns a dict of tidy Polars DataFrames you can print/export.

    Expected columns (best-effort if some are missing):
      behavior_tag,
      D0_after_coverage, D1_coverage, nightfall_flag,
      no_OV_longest_s, reconnect_feasible_screen, context_tag,
      dwell_found, dwell_first_hysteresis,
      V_trip_obs_list (List[Float]), V_rec_obs_list (optional)
    """

    # ---------- 0) Defensive: add missing columns with defaults ----------
    def ensure_col(df: pl.DataFrame, name: str, dtype, default):
        return df if name in df.columns else df.with_columns(pl.lit(default).cast(dtype).alias(name))

    rr = rr.clone()

    # Ensure core columns exist
    rr = ensure_col(rr, "behavior_tag", pl.Utf8, None)
    rr = ensure_col(rr, "D0_after_coverage", pl.Float64, None)
    rr = ensure_col(rr, "D1_coverage", pl.Float64, None)
    rr = ensure_col(rr, "nightfall_flag", pl.Boolean, False)

    rr = ensure_col(rr, "no_OV_longest_s", pl.Float64, None)
    rr = ensure_col(rr, "reconnect_feasible_screen", pl.Boolean, None)
    rr = ensure_col(rr, "context_tag", pl.Utf8, None)

    rr = ensure_col(rr, "dwell_found", pl.Boolean, None)
    rr = ensure_col(rr, "dwell_first_hysteresis", pl.Float64, None)

    rr = ensure_col(rr, "V_trip_obs_list", pl.List(pl.Float64), [])
    # Optional: if V_rec_obs_list missing, create empty lists
    rr = ensure_col(rr, "V_rec_obs_list", pl.List(pl.Float64), [])

    # Derived: trip_n, V_trip_min
    rr = rr.with_columns([
        pl.when(pl.col("V_trip_obs_list").is_not_null())
          .then(pl.col("V_trip_obs_list").list.len())
          .otherwise(pl.lit(0))
          .alias("trip_n"),
        pl.when((pl.col("V_trip_obs_list").is_not_null()) & (pl.col("V_trip_obs_list").list.len() > 0))
          .then(pl.col("V_trip_obs_list").list.min())
          .otherwise(pl.lit(None))
          .alias("V_trip_min"),
    ])

    # Helper for safe denominators
    total_events = rr.height
    if total_events == 0:
        # Return empty frames for all keys
        empty = pl.DataFrame()
        return {
            "outcome_by_tag": empty,
            "observability_stats": empty,
            "screen_stats": empty,
            "stability_stats": empty,
            "below_min_trip_stats": empty,
        }

    # ---------- 1) Outcome distribution ----------
    outcome_counts = (
        rr.group_by("behavior_tag")
          .agg(pl.len().alias("count"))
          .sort("count", descending=True)
    )
    outcome_by_tag = (
        outcome_counts
        .with_columns([
            (pl.col("count") / pl.lit(total_events)).alias("share")
        ])
    )

    # ---------- 2) Observability (Step 1) ----------
    observability_stats = (
        rr.select([
            pl.col("D0_after_coverage").cast(pl.Float64).alias("D0_after_cov"),
            pl.col("D1_coverage").cast(pl.Float64).alias("D1_cov"),
            pl.col("nightfall_flag").cast(pl.Boolean).alias("nightfall_flag"),
        ])
        .with_columns([
            pl.col("D0_after_cov").fill_null(0.0),
            pl.col("D1_cov").fill_null(0.0),
            pl.col("nightfall_flag").fill_null(False),
        ])
        .select([
            pl.col("D0_after_cov").mean().alias("D0_after_cov_mean"),
            pl.col("D1_cov").mean().alias("D1_cov_mean"),
            pl.col("D0_after_cov").median().alias("D0_after_cov_median"),
            pl.col("D1_cov").median().alias("D1_cov_median"),
            (pl.when(pl.col("nightfall_flag")).then(1).otherwise(0)).sum().alias("nightfall_count"),
            (pl.col("nightfall_flag").cast(pl.Int8).mean()).alias("nightfall_rate"),
        ])
    )

    # ---------- 3) Screen context (Step 3) ----------
    screen_stats = (
        rr.select([
            pl.col("no_OV_longest_s").cast(pl.Float64),
            pl.col("reconnect_feasible_screen").cast(pl.Boolean),
            pl.col("context_tag").cast(pl.Utf8),
        ])
        .with_columns([
            pl.col("no_OV_longest_s").fill_null(0.0),
            pl.col("reconnect_feasible_screen").fill_null(False),
        ])
        .select([
            pl.col("no_OV_longest_s").mean().alias("no_OV_longest_s_mean"),
            pl.col("no_OV_longest_s").median().alias("no_OV_longest_s_median"),
            pl.col("reconnect_feasible_screen").cast(pl.Int8).mean().alias("reconnect_feasible_screen_rate"),
        ])
    )

    # ---------- 4) Stability (Step 4) ----------
    stability_stats = (
        rr.select([
            pl.col("dwell_found").cast(pl.Boolean).fill_null(False).alias("dwell_found_flag"),
            pl.col("dwell_first_hysteresis").cast(pl.Float64),
        ])
        .with_columns([
            pl.col("dwell_found_flag").cast(pl.Int8).alias("dwell_found_int")
        ])
        .select([
            pl.col("dwell_found_int").sum().alias("dwell_found_count"),
            (pl.col("dwell_found_int").mean()).alias("dwell_found_rate"),
        ])
    )

    # ---------- 5) Below-min-trip outcomes (Step 5) ----------
    # Decompose final tags:
    bm_reconn = (rr["behavior_tag"] == "NORMAL_RECONNECT_BEHAVIOUR").sum()
    bm_no_reconn = (rr["behavior_tag"] == "LOCKOUT/PROTECTION_HOLD").sum()
    no_stable = (rr["behavior_tag"] == "NO_STABLE_CONDITIONS").sum()

    # Trip edge coverage
    with_trip_edges = (rr["trip_n"] > 0).sum()

    # V_trip_min distribution (only where available)
    trip_min_stats = (
        rr.filter(pl.col("trip_n") > 0)
          .select([
              pl.col("V_trip_min").min().alias("V_trip_min_min"),
              pl.col("V_trip_min").median().alias("V_trip_min_median"),
              pl.col("V_trip_min").quantile(0.25, "nearest").alias("V_trip_min_p25"),
              pl.col("V_trip_min").quantile(0.75, "nearest").alias("V_trip_min_p75"),
              pl.len().alias("V_trip_min_count")
          ])
    )

    below_min_trip_stats = pl.DataFrame({
        "metric": [
            "below_min_then_reconnected_count",
            "below_min_then_no_reconnect_count",
            "no_stable_conditions_count",
            "total_events",
            "with_trip_edges_count",
            "rate_below_min_then_reconnected",
            "rate_below_min_then_no_reconnect",
            "rate_no_stable_conditions",
        ],
        "value": pl.Series(
            "value",
            [
                float(bm_reconn),
                float(bm_no_reconn),
                float(no_stable),
                float(total_events),
                float(with_trip_edges),
                (bm_reconn / total_events) if total_events else 0.0,
                (bm_no_reconn / total_events) if total_events else 0.0,
                (no_stable / total_events) if total_events else 0.0,
            ],
            dtype=pl.Float64,  # << force a single dtype
        ),
    })

    results = {
        "outcome_by_tag": outcome_by_tag,
        "observability_stats": observability_stats,
        "screen_stats": screen_stats,
        "stability_stats": stability_stats,
        "below_min_trip_stats": below_min_trip_stats,
    }

    # Attach V_trip_min stats if we had any trip edges
    if trip_min_stats.height > 0:
        results["trip_min_distribution"] = trip_min_stats

    return results

def site_day_label_mix(df_events: pl.DataFrame) -> pl.DataFrame:
    """
    Site‑day behavior mix (event-based).
    For each (site_id, event_date = date(t_last_event)), compute:
      - num_events
      - counts & percentages for each behavior_tag:
          Stayed Connected During Event
          Reconnected During Event (no switching)
          Reconnected During Event (Intermittent Switching)
          Reconnected After Event
          Never Reconnected
    """
    # Derive local event_date from t_last_event (no tz cast)
    df = df_events.with_columns(
        pl.col("t_last_event").dt.date().alias("event_date")
    )

    # Boolean tests for each label
    stayed            = (pl.col("behavior_tag") == "Stayed Connected During Event")
    during_no_switch  = (pl.col("behavior_tag") == "Reconnected During Event (no switching)")
    during_switch     = (pl.col("behavior_tag") == "Reconnected During Event (Intermittent Switching)")
    after             = (pl.col("behavior_tag") == "Reconnected After Event")
    never             = (pl.col("behavior_tag") == "Never Reconnected")

    # Aggregate per (site_id, event_date)
    mix = (
        df
        .group_by(["site_id", "event_date"])
        .agg([
            pl.len().alias("num_events"),
            stayed.sum().alias("n_stayed_connected"),
            during_no_switch.sum().alias("n_reconnected_during_no_switch"),
            during_switch.sum().alias("n_reconnected_during_switch"),
            after.sum().alias("n_reconnected_after"),
            never.sum().alias("n_never_reconnected"),
        ])
        .with_columns([
            (pl.col("n_stayed_connected")             / pl.col("num_events")).alias("pct_stayed_connected"),
            (pl.col("n_reconnected_during_no_switch") / pl.col("num_events")).alias("pct_reconnected_during_no_switch"),
            (pl.col("n_reconnected_during_switch")    / pl.col("num_events")).alias("pct_reconnected_during_switch"),
            (pl.col("n_reconnected_after")            / pl.col("num_events")).alias("pct_reconnected_after"),
            (pl.col("n_never_reconnected")            / pl.col("num_events")).alias("pct_never_reconnected"),
        ])
        .sort(["site_id", "event_date"])
    )
    return mix

def site_day_voltage_stats(df_events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Returns:
      out               -> wide/flat table (per site_id, event_date) with min/max/median/std and counts per bucket
                           + pct_time_disconnected_during_events
      out_stats_struct  -> compact table with 4 struct columns whose headers document field order:
                           - 'disc_first_stats (min,max,median,std,counts)'
                           - 'disc_all_stats (min,max,median,std,counts)'
                           - 'rec_during_stats (min,max,median,std,counts)'
                           - 'rec_after_stats (min,max,median,std,counts)'
                         Each struct has fields: min, max, median, std, counts
    """
    df = add_event_date(df_events)

    # ---------- Disconnection: first per event ----------
    disc_first = (
        df
        .filter(pl.col("v_disc_first").is_not_null())
        .group_by(["site_id", "event_date"])
        .agg([
            pl.col("v_disc_first").min().alias("disc_first_min_v"),
            pl.col("v_disc_first").max().alias("disc_first_max_v"),
            pl.col("v_disc_first").median().alias("disc_first_median_v"),
            pl.col("v_disc_first").std().alias("disc_first_std_v"),
            pl.col("v_disc_first").count().alias("disc_first_count_events"),
        ])
    )

    # ---------- Disconnection: all transitions (flatten v_disc_during) ----------
    disc_all = (
        df
        .with_columns(ensure_list("v_disc_during").alias("v_disc_during"))
        .select(["site_id", "event_date", "v_disc_during"])
        .explode("v_disc_during")
        .filter(pl.col("v_disc_during").is_not_null())
        .group_by(["site_id", "event_date"])
        .agg([
            pl.col("v_disc_during").min().alias("disc_all_min_v"),
            pl.col("v_disc_during").max().alias("disc_all_max_v"),
            pl.col("v_disc_during").median().alias("disc_all_median_v"),
            pl.col("v_disc_during").std().alias("disc_all_std_v"),
            pl.col("v_disc_during").count().alias("disc_all_count_transitions"),
        ])
    )

    # ---------- Reconnection: all during-event transitions (flatten v_rec_during) ----------
    rec_during = (
        df
        .with_columns(ensure_list("v_rec_during").alias("v_rec_during"))
        .select(["site_id", "event_date", "v_rec_during"])
        .explode("v_rec_during")
        .filter(pl.col("v_rec_during").is_not_null())
        .group_by(["site_id", "event_date"])
        .agg([
            pl.col("v_rec_during").min().alias("rec_during_min_v"),
            pl.col("v_rec_during").max().alias("rec_during_max_v"),
            pl.col("v_rec_during").median().alias("rec_during_median_v"),
            pl.col("v_rec_during").std().alias("rec_during_std_v"),
            pl.col("v_rec_during").count().alias("rec_during_count_transitions"),
        ])
    )

    # ---------- Reconnection: first after-event per event ----------
    rec_after = (
        df
        .filter(pl.col("v_rec_after").is_not_null())
        .group_by(["site_id", "event_date"])
        .agg([
            pl.col("v_rec_after").min().alias("rec_after_min_v"),
            pl.col("v_rec_after").max().alias("rec_after_max_v"),
            pl.col("v_rec_after").median().alias("rec_after_median_v"),
            pl.col("v_rec_after").std().alias("rec_after_std_v"),
            pl.col("v_rec_after").count().alias("rec_after_count_events"),
        ])
    )

    # ---------- Time-based KPI: Σdisc_time / Σevent_time (per site_id, event_date) ----------
    time_kpi = (
        df
        .filter(pl.col("event_duration_sec").is_not_null() & pl.col("disc_time_in_event_sec").is_not_null())
        .group_by(["site_id", "event_date"])
        .agg([
            pl.col("event_duration_sec").sum().alias("sum_event_time_s"),
            pl.col("disc_time_in_event_sec").sum().alias("sum_disc_time_s"),
        ])
        .with_columns([
            pl.when(pl.col("sum_event_time_s") > 0)
              .then(100.0 * pl.col("sum_disc_time_s") / pl.col("sum_event_time_s"))
              .otherwise(pl.lit(None))
              .alias("pct_time_disconnected_during_events")
        ])
        .select(["site_id", "event_date", "pct_time_disconnected_during_events"])
    )

    # ---------- Join all blocks on (site_id, event_date) ----------
    base = df.select(["site_id", "event_date"]).unique()

    out = (
        base
        .join(disc_first, on=["site_id", "event_date"], how="left")
        .join(disc_all,   on=["site_id", "event_date"], how="left")
        .join(rec_during, on=["site_id", "event_date"], how="left")
        .join(rec_after,  on=["site_id", "event_date"], how="left")
        .join(time_kpi,   on=["site_id", "event_date"], how="left")  # <- NEW KPI
        .sort(["site_id", "event_date"])
    )

    # Fill only counts to 0; leave stats/KPI as null when not defined
    count_cols = [
        "disc_first_count_events",
        "disc_all_count_transitions",
        "rec_during_count_transitions",
        "rec_after_count_events",
    ]
    out = out.with_columns([pl.col(c).fill_null(0) for c in count_cols])

    # ---------- Separate compact struct table with ALL categories ----------
    out_stats_struct = (
        out
        .with_columns([
            pl.struct([
                pl.col("disc_first_min_v").alias("min"),
                pl.col("disc_first_max_v").alias("max"),
                pl.col("disc_first_median_v").alias("median"),
                pl.col("disc_first_std_v").alias("std"),
                pl.col("disc_first_count_events").alias("counts"),
            ]).alias("disc_first_stats (min,max,median,std,counts)"),

            pl.struct([
                pl.col("disc_all_min_v").alias("min"),
                pl.col("disc_all_max_v").alias("max"),
                pl.col("disc_all_median_v").alias("median"),
                pl.col("disc_all_std_v").alias("std"),
                pl.col("disc_all_count_transitions").alias("counts"),
            ]).alias("disc_all_stats (min,max,median,std,counts)"),

            pl.struct([
                pl.col("rec_during_min_v").alias("min"),
                pl.col("rec_during_max_v").alias("max"),
                pl.col("rec_during_median_v").alias("median"),
                pl.col("rec_during_std_v").alias("std"),
                pl.col("rec_during_count_transitions").alias("counts"),
            ]).alias("rec_during_stats (min,max,median,std,counts)"),

            pl.struct([
                pl.col("rec_after_min_v").alias("min"),
                pl.col("rec_after_max_v").alias("max"),
                pl.col("rec_after_median_v").alias("median"),
                pl.col("rec_after_std_v").alias("std"),
                pl.col("rec_after_count_events").alias("counts"),
            ]).alias("rec_after_stats (min,max,median,std,counts)"),
        ])
        .select([
            "site_id", "event_date",
            "disc_first_stats (min,max,median,std,counts)",
            "disc_all_stats (min,max,median,std,counts)",
            "rec_during_stats (min,max,median,std,counts)",
            "rec_after_stats (min,max,median,std,counts)",
        ])
        .sort(["site_id", "event_date"])
    )

    return out, out_stats_struct

def site_label_mix(df_events: pl.DataFrame) -> pl.DataFrame:
    """
    Site-level behavior mix:
      - num_events
      - counts & percentages for each behavior_tag category.
    """
    stayed             = (pl.col("behavior_tag") == "Stayed Connected During Event")
    during_no_switch   = (pl.col("behavior_tag") == "Reconnected During Event (no switching)")
    during_switch      = (pl.col("behavior_tag") == "Reconnected During Event (Intermittent Switching)")
    after              = (pl.col("behavior_tag") == "Reconnected After Event")
    never              = (pl.col("behavior_tag") == "Never Reconnected")

    mix = (
        df_events
        .group_by("site_id")
        .agg([
            pl.len().alias("num_events"),
            stayed.sum().alias("n_stayed_connected"),
            during_no_switch.sum().alias("n_reconnected_during_no_switch"),
            during_switch.sum().alias("n_reconnected_during_switch"),
            after.sum().alias("n_reconnected_after"),
            never.sum().alias("n_never_reconnected"),
        ])
        .with_columns([
            (pl.col("n_stayed_connected")              / pl.col("num_events")).alias("pct_stayed_connected"),
            (pl.col("n_reconnected_during_no_switch")  / pl.col("num_events")).alias("pct_reconnected_during_no_switch"),
            (pl.col("n_reconnected_during_switch")     / pl.col("num_events")).alias("pct_reconnected_during_switch"),
            (pl.col("n_reconnected_after")             / pl.col("num_events")).alias("pct_reconnected_after"),
            (pl.col("n_never_reconnected")             / pl.col("num_events")).alias("pct_never_reconnected"),
        ])
        .sort("site_id")
    )
    return mix

def site_voltage_stats(df_events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Site-level voltage distributions:
      out: wide flat table per site_id (min/max/median/std + counts for each bucket)
           + pct_time_disconnected_during_events
           + event_dates (list of distinct event_date covered by the site)
      out_stats_struct: compact struct view with 4 columns:
        'disc_first_stats (min,max,median,std,counts)'
        'disc_all_stats (min,max,median,std,counts)'
        'rec_during_stats (min,max,median,std,counts)'
        'rec_after_stats (min,max,median,std,counts)'
      Each struct contains fields: min, max, median, std, counts
    """
    # ---------- Disconnection: first per event ----------
    disc_first = (
        df_events
        .filter(pl.col("v_disc_first").is_not_null())
        .group_by("site_id")
        .agg([
            pl.col("v_disc_first").min().alias("disc_first_min_v"),
            pl.col("v_disc_first").max().alias("disc_first_max_v"),
            pl.col("v_disc_first").median().alias("disc_first_median_v"),
            pl.col("v_disc_first").std().alias("disc_first_std_v"),
            pl.col("v_disc_first").count().alias("disc_first_count_events"),
        ])
    )

    # ---------- Disconnection: all transitions (flatten v_disc_during) ----------
    disc_all = (
        df_events
        .with_columns(ensure_list("v_disc_during").alias("v_disc_during"))
        .select(["site_id", "v_disc_during"])
        .explode("v_disc_during")
        .filter(pl.col("v_disc_during").is_not_null())
        .group_by("site_id")
        .agg([
            pl.col("v_disc_during").min().alias("disc_all_min_v"),
            pl.col("v_disc_during").max().alias("disc_all_max_v"),
            pl.col("v_disc_during").median().alias("disc_all_median_v"),
            pl.col("v_disc_during").std().alias("disc_all_std_v"),
            pl.col("v_disc_during").count().alias("disc_all_count_transitions"),
        ])
    )

    # ---------- Reconnection: all during-event transitions (flatten v_rec_during) ----------
    rec_during = (
        df_events
        .with_columns(ensure_list("v_rec_during").alias("v_rec_during"))
        .select(["site_id", "v_rec_during"])
        .explode("v_rec_during")
        .filter(pl.col("v_rec_during").is_not_null())
        .group_by("site_id")
        .agg([
            pl.col("v_rec_during").min().alias("rec_during_min_v"),
            pl.col("v_rec_during").max().alias("rec_during_max_v"),
            pl.col("v_rec_during").median().alias("rec_during_median_v"),
            pl.col("v_rec_during").std().alias("rec_during_std_v"),
            pl.col("v_rec_during").count().alias("rec_during_count_transitions"),
        ])
    )

    # ---------- Reconnection: first after-event per event ----------
    rec_after = (
        df_events
        .filter(pl.col("v_rec_after").is_not_null())
        .group_by("site_id")
        .agg([
            pl.col("v_rec_after").min().alias("rec_after_min_v"),
            pl.col("v_rec_after").max().alias("rec_after_max_v"),
            pl.col("v_rec_after").median().alias("rec_after_median_v"),
            pl.col("v_rec_after").std().alias("rec_after_std_v"),
            pl.col("v_rec_after").count().alias("rec_after_count_events"),
        ])
    )

    # ---------- Time-based KPI at site level ----------
    time_kpi = (
        df_events
        .filter(pl.col("event_duration_sec").is_not_null() & pl.col("disc_time_in_event_sec").is_not_null())
        .group_by("site_id")
        .agg([
            pl.col("event_duration_sec").sum().alias("sum_event_time_s"),
            pl.col("disc_time_in_event_sec").sum().alias("sum_disc_time_s"),
        ])
        .with_columns([
            pl.when(pl.col("sum_event_time_s") > 0)
              .then(100.0 * pl.col("sum_disc_time_s") / pl.col("sum_event_time_s"))
              .otherwise(pl.lit(None))
              .alias("pct_time_disconnected_during_events")
        ])
        .select(["site_id", "pct_time_disconnected_during_events"])
    )

    # ---------- event_dates (distinct list of event days per site) ----------
    event_dates = (
        df_events
        .with_columns(pl.col("t_last_event").dt.date().alias("event_date"))
        .group_by("site_id")
        .agg(pl.col("event_date").unique().sort().alias("event_dates"))
    )

    # ---------- Join all blocks on site_id ----------
    base = df_events.select("site_id").unique()

    out = (
        base
        .join(disc_first, on="site_id", how="left")
        .join(disc_all,   on="site_id", how="left")
        .join(rec_during, on="site_id", how="left")
        .join(rec_after,  on="site_id", how="left")
        .join(time_kpi,   on="site_id", how="left")   # <- NEW KPI
        .join(event_dates,on="site_id", how="left")   # <- NEW event_dates
        .sort("site_id")
    )

    # Fill only counts to 0; leave stats/KPI/event_dates as null when not defined
    count_cols = [
        "disc_first_count_events",
        "disc_all_count_transitions",
        "rec_during_count_transitions",
        "rec_after_count_events",
    ]
    out = out.with_columns([pl.col(c).fill_null(0) for c in count_cols])

    # ---------- Separate compact struct table with ALL categories ----------
    out_stats_struct = (
        out
        .with_columns([
            pl.struct([
                pl.col("disc_first_min_v").alias("min"),
                pl.col("disc_first_max_v").alias("max"),
                pl.col("disc_first_median_v").alias("median"),
                pl.col("disc_first_std_v").alias("std"),
                pl.col("disc_first_count_events").alias("counts"),
            ]).alias("disc_first_stats (min,max,median,std,counts)"),

            pl.struct([
                pl.col("disc_all_min_v").alias("min"),
                pl.col("disc_all_max_v").alias("max"),
                pl.col("disc_all_median_v").alias("median"),
                pl.col("disc_all_std_v").alias("std"),
                pl.col("disc_all_count_transitions").alias("counts"),
            ]).alias("disc_all_stats (min,max,median,std,counts)"),

            pl.struct([
                pl.col("rec_during_min_v").alias("min"),
                pl.col("rec_during_max_v").alias("max"),
                pl.col("rec_during_median_v").alias("median"),
                pl.col("rec_during_std_v").alias("std"),
                pl.col("rec_during_count_transitions").alias("counts"),
            ]).alias("rec_during_stats (min,max,median,std,counts)"),

            pl.struct([
                pl.col("rec_after_min_v").alias("min"),
                pl.col("rec_after_max_v").alias("max"),
                pl.col("rec_after_median_v").alias("median"),
                pl.col("rec_after_std_v").alias("std"),
                pl.col("rec_after_count_events").alias("counts"),
            ]).alias("rec_after_stats (min,max,median,std,counts)"),
        ])
        .select([
            "site_id",
            "disc_first_stats (min,max,median,std,counts)",
            "disc_all_stats (min,max,median,std,counts)",
            "rec_during_stats (min,max,median,std,counts)",
            "rec_after_stats (min,max,median,std,counts)",
        ])
        .sort("site_id")
    )

    return out, out_stats_struct

def filter_sites_by_time_kpi_leq(
    site_level_out: pl.DataFrame,
    pct_threshold: float = 90.0,  # "disconnected 90% or less"
) -> pl.DataFrame:
    """
    Keep sites whose pct_time_disconnected_during_events <= pct_threshold.
    Expects columns: 'site_id', 'pct_time_disconnected_during_events', 'event_dates'.
    Returns: site_id | pct_time_disconnected_during_events | event_dates
    """
    required = {"site_id", "pct_time_disconnected_during_events", "event_dates"}
    missing = required - set(site_level_out.columns)
    if missing:
        raise ValueError(f"Missing required columns in site_level_out: {sorted(missing)}")

    return (
        site_level_out
        .filter(pl.col("pct_time_disconnected_during_events") <= pct_threshold)
        .select(["site_id", "pct_time_disconnected_during_events", "event_dates"])
        .sort("site_id")
    )

def build_mixed_compliance_views(
    reconnection_result: pl.DataFrame,
    site_day_out: pl.DataFrame,   # from site_day_voltage_stats(...).out (must have site_id, event_date, pct_time_disconnected_during_events)
    T: float = 90.0,              # compliance threshold (%)
):
    """
    Returns three Polars DataFrames (in this order):
      1) mixed_sites_summary   (site-level, Mixed Compliance sites only)
      2) mixed_sites_day_detail (site-day, Mixed sites only; excludes 'No KPI' days)
      3) site_day_time_pct     (separate day-level time % for Mixed sites only)

    Expects:
      reconnection_result columns: ['site_id','t_last_event','event_duration_sec','disc_time_in_event_sec', ...]
      site_day_out columns:        ['site_id','event_date','pct_time_disconnected_during_events', ...]
    """

    # --- Validate inputs ---
    req_day = {"site_id", "event_date", "pct_time_disconnected_during_events"}
    if not req_day.issubset(site_day_out.columns):
        missing = sorted(req_day - set(site_day_out.columns))
        raise ValueError(f"site_day_out missing columns: {missing}")

    req_evt = {"site_id", "t_last_event", "event_duration_sec", "disc_time_in_event_sec"}
    if not req_evt.issubset(reconnection_result.columns):
        missing = sorted(req_evt - set(reconnection_result.columns))
        raise ValueError(f"reconnection_result missing columns: {missing}")

    # ---------- A) Site-day KPI label ----------
    kpi_day = (
        site_day_out
        .with_columns([
            pl.when(pl.col("pct_time_disconnected_during_events").is_null())
              .then(pl.lit("No KPI"))
              .when(pl.col("pct_time_disconnected_during_events") >= T)
              .then(pl.lit("Compliant"))
              .otherwise(pl.lit("Non-compliant"))
              .alias("LABEL_DAY")
        ])
    )
    # Exclude No-KPI days for compliance classification
    kpi_day_valid = kpi_day.filter(pl.col("LABEL_DAY") != "No KPI")

    # ---------- B) Mixed-Compliance sites (across days) ----------
    per_site_day_lists = (
        kpi_day_valid
        .group_by("site_id")
        .agg([
            pl.col("event_date").filter(pl.col("LABEL_DAY") == "Compliant").unique().sort().alias("COMPLIANT_DATES"),
            pl.col("event_date").filter(pl.col("LABEL_DAY") == "Non-compliant").unique().sort().alias("NON_COMPLIANT_DATES"),
            pl.col("LABEL_DAY").eq("Compliant").sum().alias("n_days_compliant"),
            pl.col("LABEL_DAY").eq("Non-compliant").sum().alias("n_days_non_compliant"),
        ])
        .with_columns([
            (100.0 * pl.col("n_days_compliant")
             / (pl.col("n_days_compliant") + pl.col("n_days_non_compliant"))).alias("pct_days_compliant"),
        ])
        .with_columns([
            (100.0 - pl.col("pct_days_compliant")).alias("pct_days_non_compliant")
        ])
    )

    mixed_sites = (
        per_site_day_lists
        .filter((pl.col("n_days_compliant") > 0) & (pl.col("n_days_non_compliant") > 0))
        .select("site_id")
    )

    # ---------- C) Event-level KPI + event_date ----------
    events = (
        reconnection_result
        .with_columns(pl.col("t_last_event").dt.date().alias("event_date"))
        .with_columns([
            pl.when(pl.col("event_duration_sec") > 0)
              .then(100.0 * pl.col("disc_time_in_event_sec") / pl.col("event_duration_sec"))
              .otherwise(pl.lit(None))
              .alias("event_KPI")
        ])
        .select(["site_id", "event_date", "event_duration_sec", "event_KPI"])
    )

    # Same-day compliance conflict: both compliant & non-compliant events present?
    same_day_conflict = (
        events
        .group_by(["site_id", "event_date"])
        .agg([
            (pl.col("event_KPI") >= T).any().alias("has_Compliant_Event"),
            (pl.col("event_KPI") <  T).any().alias("has_NonCompliant_Event"),
        ])
        .with_columns([
            pl.when(pl.col("has_Compliant_Event") & pl.col("has_NonCompliant_Event"))
              .then(pl.lit("Yes"))
              .otherwise(pl.lit("No"))
              .alias("SAME_DAY_COMPLIANCE_CONFLICT")
        ])
        .select(["site_id", "event_date", "SAME_DAY_COMPLIANCE_CONFLICT"])
    )

    # ---------- D) Day-level time % split (separate variable) ----------
    site_day_time_pct = (
        events
        .group_by(["site_id", "event_date"])
        .agg([
            pl.col("event_duration_sec").sum().alias("sum_event_time"),
            pl.col("event_duration_sec").filter(pl.col("event_KPI") >= T).sum().alias("sum_compliant_time"),
        ])
        .with_columns([
            pl.when(pl.col("sum_event_time") > 0)
              .then(100.0 * pl.col("sum_compliant_time") / pl.col("sum_event_time"))
              .otherwise(pl.lit(None))
              .alias("pct_time_compliant_day")
        ])
        .with_columns([
            (100.0 - pl.col("pct_time_compliant_day")).alias("pct_time_non_compliant_day")
        ])
        .select(["site_id", "event_date", "pct_time_compliant_day", "pct_time_non_compliant_day"])
    )
    # Limit day-level time % to Mixed sites
    site_day_time_pct = site_day_time_pct.join(mixed_sites, on="site_id", how="inner").sort(["site_id", "event_date"])

    # ---------- E) Site-day detail (Mixed sites only; exclude "No KPI") ----------
    mixed_sites_day_detail = (
        kpi_day_valid
        .join(mixed_sites, on="site_id", how="inner")
        .join(same_day_conflict, on=["site_id", "event_date"], how="left")
        .select([
            "site_id",
            "event_date",
            "pct_time_disconnected_during_events",
            "LABEL_DAY",
            "SAME_DAY_COMPLIANCE_CONFLICT",
        ])
        .sort(["site_id", "event_date"])
    )

    # ---------- F) Site-level time % (Mixed sites only) ----------
    site_time_pct = (
        events
        .join(mixed_sites, on="site_id", how="inner")
        .group_by("site_id")
        .agg([
            pl.col("event_duration_sec").sum().alias("sum_event_time_site"),
            pl.col("event_duration_sec").filter(pl.col("event_KPI") >= T).sum().alias("sum_compliant_time_site"),
        ])
        .with_columns([
            pl.when(pl.col("sum_event_time_site") > 0)
              .then(100.0 * pl.col("sum_compliant_time_site") / pl.col("sum_event_time_site"))
              .otherwise(pl.lit(None))
              .alias("pct_time_compliant_site")
        ])
        .with_columns([
            (100.0 - pl.col("pct_time_compliant_site")).alias("pct_time_non_compliant_site")
        ])
        .select(["site_id", "pct_time_compliant_site", "pct_time_non_compliant_site"])
    )

    # ---------- G) Final site-level summary (Mixed only) ----------
    mixed_sites_summary = (
        per_site_day_lists
        .join(mixed_sites, on="site_id", how="inner")
        .join(site_time_pct, on="site_id", how="left")
        .select([
            "site_id",
            "COMPLIANT_DATES",
            "NON_COMPLIANT_DATES",
            "n_days_compliant",
            "n_days_non_compliant",
            # "pct_days_compliant",
            # "pct_days_non_compliant",
            "pct_time_compliant_site",
            "pct_time_non_compliant_site",
        ])
        .sort("site_id")
    )

    # Return three DataFrames (tuple) — unpack as: site_summary, site_day_detail, site_day_time_pct = build_mixed_compliance_views(...)
    return mixed_sites_summary, mixed_sites_day_detail, site_day_time_pct
