"""Legacy never-reconnected event analysis retained for reference."""

import polars as pl
from datetime import timedelta
from typing import Optional, List

import polars as pl
from datetime import timedelta
from typing import Optional, List, Tuple, Any
import polars as pl
from datetime import timedelta
from typing import Optional, List, Tuple, Any

def analyze_never_reconnected_events(
    df: pl.DataFrame,
    reconnection_result: pl.DataFrame,
    vmean_cols: List[str],
    vThreshold: float,
    df_d1: Optional[pl.DataFrame] = None,
    df_d2: Optional[pl.DataFrame] = None,
    *,
    # Tunables
    MIN_DAYLIGHT_COVERAGE: float = 0.10,      # Step 1: ≥10% usable voltage data
    NO_OV_SCREEN_SECONDS: int = 600,          # Step 3: 10 min continuous NO-OV screen (informational)
    INTERMITTENT_STABLE_TOTAL_S: int = 300,   # Step 3: <5 min total ALL≤(Vmax-1.0V) → INTERMITTENT_OV
    T_REC_S: int = 60,                        # Step 4: dwell ≥60s
    DELTA_LIST: Tuple[float, ...] = (1.0, 2.0, 2.3, 3.0, 4.6),  # Step 4 Δ sweep
    VREC_FLOOR_V: float = 251.1,              # test lower bound for V_rec
    VTEST_EPS_V: float = 0.2,                 # Step 5: strictly-below margin for min trip
    sunrise_d0 = None, 
    sunset_d0 = None
) -> pl.DataFrame:
    """
    Per-event NR analysis (Steps 1–5), returning a CLEAN reconnection_result:
      - No *_right columns
      - Only agreed fields added
      - behavior_tag updated to final reason (NIGHTFALL / RECONNECTED_* / NO_STABLE_CONDITIONS / NORMAL_RECONNECT_BEHAVIOUR / LOCKOUT/PROTECTION_HOLD)

    Inputs:
      df: Day-0 daylight-only (has 'local_tstamp','duration','over_vmax','is_disc', and vmean_cols)
      reconnection_result: includes ['active_event_id','behavior_tag','t_last_event','site_id'(opt)]
      vmean_cols: your 'vmean_rolling_10m*' columns
      vThreshold: Vmax used to derive over_vmax

    Optional:
      df_d1, df_d2: daylight-only Day+1/Day+2 frames (same schema). Use None if unavailable.
    """

    # ---------------------------
    # Day 0 bounds (from df)
    # ---------------------------
    if sunrise_d0 is None:
        sunrise_d0 = df.select(pl.col("local_tstamp").first()).item()
    if sunset_d0 is None:
        sunset_d0  = df.select(pl.col("local_tstamp").last()).item()
    Vmax = float(vThreshold)
    SEARCH_HORIZON_H = 48

    # ---------------------------
    # Helpers
    # ---------------------------
    def coverage_seconds(df_window: Optional[pl.DataFrame]) -> float:
        if df_window is None or df_window.is_empty():
            return 0.0
        usable_mask = pl.any_horizontal([pl.col(c).is_not_null() for c in vmean_cols])
        dur_all = float(df_window.select(pl.col("duration").sum()).item() or 0.0)
        if dur_all <= 0.0:
            return 0.0
        dur_usable = float(df_window.filter(usable_mask).select(pl.col("duration").sum()).item() or 0.0)
        return dur_usable / dur_all

    def longest_run_seconds(df_window: Optional[pl.DataFrame], bool_col: str) -> float:
        if df_window is None or df_window.is_empty():
            return 0.0
        runs = (
            df_window
            .with_columns(pl.col(bool_col).cast(pl.Int8).alias("flag_i"))
            .with_columns(
                pl.when(pl.col("flag_i") != pl.col("flag_i").shift(1))
                  .then(1).otherwise(0)
                  .cum_sum()
                  .alias("run_id")
            )
        )
        true_runs = (
            runs.filter(pl.col("flag_i") == 1)
                .group_by("run_id")
                .agg(pl.col("duration").sum().alias("dur_s"))
        )
        if true_runs.is_empty():
            return 0.0
        return float(true_runs.select(pl.col("dur_s").max()).item())

    def total_seconds_where(df_window: Optional[pl.DataFrame], expr: pl.Expr) -> float:
        if df_window is None or df_window.is_empty():
            return 0.0
        return float(df_window.filter(expr).select(pl.col("duration").sum()).item() or 0.0)

    def first_reconnect_within(df_win: Optional[pl.DataFrame], t_last_event, horizon_h: int):
        if df_win is None or df_win.is_empty():
            return None
        t_end = t_last_event + timedelta(hours=horizon_h)
        hits = (
            df_win
            .filter(
                (~pl.col("is_disc")) &
                (pl.col("local_tstamp") > pl.lit(t_last_event)) &
                (pl.col("local_tstamp") <= pl.lit(t_end))
            )
            .select(pl.col("local_tstamp"))
            .sort("local_tstamp")
        )
        return None if hits.is_empty() else hits.item()

    def earliest_qualifying_dwell(df_window: Optional[pl.DataFrame], V_rec: float, T_rec_s: int) -> Tuple[Optional[Any], float]:
        if df_window is None or df_window.is_empty():
            return (None, 0.0)
        safe_expr = pl.all_horizontal([pl.col(c).fill_null(pl.lit(float("inf"))) <= V_rec for c in vmean_cols])
        win = df_window.with_columns(safe_expr.alias("safe"))
        if win.filter(pl.col("safe")).is_empty():
            return (None, 0.0)
        runs = (
            win
            .with_columns(pl.col("safe").cast(pl.Int8).alias("safe_i"))
            .with_columns(
                pl.when(pl.col("safe_i") != pl.col("safe_i").shift(1))
                  .then(1).otherwise(0)
                  .cum_sum()
                  .alias("run_id")
            )
        )
        safe_runs = (
            runs.filter(pl.col("safe_i") == 1)
                .group_by("run_id")
                .agg([
                    pl.col("duration").sum().alias("dur_s"),
                    pl.col("local_tstamp").min().alias("start_ts"),
                    pl.col("local_tstamp").max().alias("end_ts"),
                ])
                .sort("start_ts")
        )
        if safe_runs.is_empty():
            return (None, 0.0)
        qual = safe_runs.filter(pl.col("dur_s") >= T_REC_S)
        if qual.is_empty():
            return (None, 0.0)
        start_ts = qual.select(pl.col("start_ts").min()).item()
        dur_s = float(qual.sort("start_ts").select(pl.col("dur_s").first()).item())
        return (start_ts, dur_s)

    def rowwise_vmax_expr():
        return pl.max_horizontal([pl.col(c) for c in vmean_cols]).cast(pl.Float64)

    # ---------------------------
    # Filter to NR rows
    # ---------------------------
    nr_mask = (reconnection_result["behavior_tag"] == "Never Reconnected")
    nr_events = reconnection_result.filter(nr_mask)

    updates = []  # one dict per active_event_id

    for row in nr_events.iter_rows(named=True):
        ev_id = row["active_event_id"]
        t_last_event = row["t_last_event"]

        # Windows
        d0_start = max(t_last_event, sunrise_d0)
        df_D0_after = df.filter(
            (pl.col("local_tstamp") >= pl.lit(d0_start)) &
            (pl.col("local_tstamp") <= pl.lit(sunset_d0))
        )
        df_D1 = df_d1
        df_D2 = df_d2

        # ---------------- Step 1: Observability
        D0_after_cov = coverage_seconds(df_D0_after)
        D1_cov = coverage_seconds(df_D1)
        D2_cov = coverage_seconds(df_D2)
        nightfall_flag = (D0_after_cov < MIN_DAYLIGHT_COVERAGE) and (D1_cov < MIN_DAYLIGHT_COVERAGE)

        # Defaults
        behavior_tag_final = row["behavior_tag"]
        reconnect_time_step2 = None
        reconnection_time_step2_s = None
        behavior_tag_step2 = "Never Reconnected"

        no_OV_longest_s = None
        reconnect_feasible_screen = None
        context_tag = None

        # Step 4 data we will record (no decision here)
        dwell_found = False
        dwell_first_start = None
        dwell_first_window = None
        dwell_first_hysteresis = None

        # ---------------- NIGHTFALL early stop
        if nightfall_flag:
            behavior_tag_final = "NIGHTFALL"
        else:
            # --------------- Step 2: Reconnection search in D1/D2 within +48h
            candidates = []
            t_rec_d1 = first_reconnect_within(df_D1, t_last_event, SEARCH_HORIZON_H) if df_D1 is not None else None
            if t_rec_d1 is not None:
                candidates.append(("D1", t_rec_d1))
            t_rec_d2 = first_reconnect_within(df_D2, t_last_event, SEARCH_HORIZON_H) if df_D2 is not None else None
            if t_rec_d2 is not None:
                candidates.append(("D2", t_rec_d2))

            if candidates:
                win_tag, t_rec = sorted(candidates, key=lambda x: x[1])[0]
                reconnect_time_step2 = t_rec
                reconnection_time_step2_s = (t_rec - t_last_event).total_seconds()
                behavior_tag_step2 = "RECONNECTED_NEXT_DAY" if win_tag == "D1" else "RE reconnected_late"
                behavior_tag_final = behavior_tag_step2
            else:
                # --------------- Step 3: Screen (informational only)
                def longest_no_ov(win):
                    return 0.0 if (win is None or win.is_empty()) else \
                        longest_run_seconds(win.with_columns((~pl.col("over_vmax")).alias("not_ov")), "not_ov")

                noOV_D0 = longest_no_ov(df_D0_after)
                noOV_D1 = longest_no_ov(df_D1)
                noOV_D2 = longest_no_ov(df_D2)
                no_OV_longest_s = max(noOV_D0, noOV_D1, noOV_D2)
                reconnect_feasible_screen = bool(no_OV_longest_s >= NO_OV_SCREEN_SECONDS)

                if any(w is not None and not w.is_empty() for w in [df_D0_after, df_D1, df_D2]):
                    V_rec_ref = Vmax - 1.0
                    dur_ov_total, dur_stable_ref = 0.0, 0.0
                    for w in [x for x in [df_D0_after, df_D1, df_D2] if (x is not None and not x.is_empty())]:
                        dur_ov_total += total_seconds_where(w, pl.col("over_vmax"))
                        dur_stable_ref += total_seconds_where(
                            w,
                            pl.all_horizontal([pl.col(c).fill_null(pl.lit(float("inf"))) <= V_rec_ref for c in vmean_cols])
                        )
                    if (dur_ov_total > 0.0) and (dur_stable_ref < INTERMITTENT_STABLE_TOTAL_S):
                        context_tag = "INTERMITTENT_OV"

                # --------------- Step 4: Stability (fairness) test
                # Only decide NO_STABLE here; if any dwell exists, defer to Step 5.
                dwell_any = False
                dwell_info_captured = False

                for win_name, df_win in [("D0_after", df_D0_after), ("D1", df_D1), ("D2", df_D2)]:
                    if df_win is None or df_win.is_empty():
                        continue
                    for DELTA in DELTA_LIST:
                        V_rec_cur = max(Vmax - DELTA, VREC_FLOOR_V)
                        start_ts, dur_s = earliest_qualifying_dwell(df_win, V_rec_cur, T_REC_S)
                        if start_ts is not None:
                            dwell_any = True
                            if not dwell_info_captured:
                                dwell_first_start = start_ts
                                dwell_first_window = win_name
                                dwell_first_hysteresis = DELTA
                                dwell_info_captured = True
                            # keep scanning other windows/Δ only to know if *any* dwell exists;
                            # we don't need to find more than existence for Step 5.
                    # continue to next window
                if not dwell_any:
                    behavior_tag_final = "NO_STABLE_CONDITIONS"
                else:
                    dwell_found = True

        # ---------------- Step 5 — Label via min disconnect voltage rule
        # Applies only if still NR AND dwell_found True AND not already labeled in Steps 1-2-4
        if (behavior_tag_final == "Never Reconnected") and dwell_found:
            # Build event-scope edges and observed voltages
            df_ev = df.filter(pl.col("active_event_id") == pl.lit(ev_id)).with_columns([
                pl.col("is_disc").cast(pl.Int8).alias("is_disc_i"),
                pl.col("is_disc").shift(1).fill_null(False).cast(pl.Int8).alias("is_disc_prev_i"),
                rowwise_vmax_expr().alias("v_obs")
            ])
            edge_disc = df_ev.filter((pl.col("is_disc_i") == 1) & (pl.col("is_disc_prev_i") == 0))
            edge_rec  = df_ev.filter((pl.col("is_disc_i") == 0) & (pl.col("is_disc_prev_i") == 1))

            V_trip_obs_list = edge_disc.select("v_obs")["v_obs"].to_list() if not edge_disc.is_empty() else []
            V_rec_obs_list  = edge_rec.select("v_obs")["v_obs"].to_list() if not edge_rec.is_empty() else []

            # If no disconnect edges → cannot form V_test → fall back to NO_STABLE_CONDITIONS
            if not V_trip_obs_list:
                behavior_tag_final = "NO_STABLE_CONDITIONS"
            else:
                V_trip_min = min(V_trip_obs_list)
                V_test = V_trip_min - VTEST_EPS_V

                # Build a combined forward window from t_last_event across D0_after/D1/D2
                frames = []
                if df_D0_after is not None and not df_D0_after.is_empty():
                    frames.append(df_D0_after)
                if df_D1 is not None and not df_D1.is_empty():
                    frames.append(df_D1)
                if df_D2 is not None and not df_D2.is_empty():
                    frames.append(df_D2)
                df_all = pl.concat(frames, how="vertical") if frames else pl.DataFrame()

                if df_all.is_empty():
                    behavior_tag_final = "NO_STABLE_CONDITIONS"
                else:
                    df_forward = df_all.filter(pl.col("local_tstamp") > pl.lit(t_last_event))

                    # Did ALL phases go <= V_test at any time?
                    safe_expr_test = pl.all_horizontal([pl.col(c).fill_null(pl.lit(float("inf"))) <= V_test for c in vmean_cols])
                    df_forward = df_forward.with_columns(safe_expr_test.alias("safe_below_min_trip"))
                    below_rows = df_forward.filter(pl.col("safe_below_min_trip"))

                    if below_rows.is_empty():
                        # Never went below min-trip → no fair chance per your Step 5 rule
                        behavior_tag_final = "NO_STABLE_CONDITIONS"
                    else:
                        # Find first time we went below min-trip
                        t_safe = below_rows.select(pl.col("local_tstamp").min()).item()
                        # Is there any reconnect edge later?
                        has_rec_after = False
                        if not edge_rec.is_empty():
                            has_rec_after = bool(
                                edge_rec.filter(pl.col("local_tstamp") > pl.lit(t_safe)).height > 0
                            )
                        behavior_tag_final = "NORMAL_RECONNECT_BEHAVIOUR" if has_rec_after else "LOCKOUT/PROTECTION_HOLD"

            # We also keep the lists for audit/metadata
            trip_stats = _stats = _rec_stats = None  # (we won't compute spread unless you want)
        else:
            # If we didn't enter Step 5 (or already set by earlier steps), compute lists for metadata anyway
            df_ev = df.filter(pl.col("active_event_id") == pl.lit(ev_id)).with_columns([
                pl.col("is_disc").cast(pl.Int8).alias("is_disc_i"),
                pl.col("is_disc").shift(1).fill_null(False).cast(pl.Int8).alias("is_disc_prev_i"),
                rowwise_vmax_expr().alias("v_obs")
            ])
            edge_disc = df_ev.filter((pl.col("is_disc_i") == 1) & (pl.col("is_disc_prev_i") == 0))
            edge_rec  = df_ev.filter((pl.col("is_disc_i") == 0) & (pl.col("is_disc_prev_i") == 1))
            V_trip_obs_list = edge_disc.select("v_obs")["v_obs"].to_list() if not edge_disc.is_empty() else []
            V_rec_obs_list  = edge_rec.select("v_obs")["v_obs"].to_list() if not edge_rec.is_empty() else []

        # --------------- Collect update row
        updates.append({
            "active_event_id": ev_id,
            # Step 1
            "D0_after_coverage": D0_after_cov,
            "D1_coverage": D1_cov,
            "D2_coverage": D2_cov,
            "nightfall_flag": nightfall_flag,
            # Step 2
            "reconnect_time_step2": reconnect_time_step2,
            "reconnection_time_step2_s": reconnection_time_step2_s,
            "behavior_tag_step2": behavior_tag_step2,
            # Step 3
            "no_OV_longest_s": no_OV_longest_s,
            "reconnect_feasible_screen": reconnect_feasible_screen,
            "context_tag": context_tag,
            # Step 4 (informational only)
            "dwell_found": dwell_found,
            "dwell_first_start": dwell_first_start,
            "dwell_first_window": dwell_first_window,
            "dwell_first_hysteresis": dwell_first_hysteresis,
            # Step 5 (lists for audit)
            "V_trip_obs_list": V_trip_obs_list,
            "V_rec_obs_list":  V_rec_obs_list,
            # Final reason
            "behavior_tag_final": behavior_tag_final,
        })

    # ---------------------------
    # Single clean join back (no duplicates)
    # ---------------------------
    updates_df = pl.DataFrame(updates) if updates else pl.DataFrame({"active_event_id": pl.Series([], dtype=pl.Int64)})

    # Drop any of these columns if they already exist on reconnection_result to avoid *_right
    new_cols = [
        "D0_after_coverage","D1_coverage","D2_coverage","nightfall_flag",
        "reconnect_time_step2","reconnection_time_step2_s","behavior_tag_step2",
        "no_OV_longest_s","reconnect_feasible_screen","context_tag",
        "dwell_found","dwell_first_start","dwell_first_window","dwell_first_hysteresis",
        "V_trip_obs_list","V_rec_obs_list",
        "behavior_tag_final"
    ]
    cols_to_drop = [c for c in new_cols if c in reconnection_result.columns]
    if cols_to_drop:
        reconnection_result = reconnection_result.drop(cols_to_drop)

    # Ensure key dtype matches
    reconnection_result = reconnection_result.with_columns(pl.col("active_event_id").cast(pl.Int64))
    if "active_event_id" in updates_df.columns:
        updates_df = updates_df.with_columns(pl.col("active_event_id").cast(pl.Int64))

    # Single left join
    rr = reconnection_result.join(updates_df, on="active_event_id", how="left")

    # Promote final behavior_tag (only for NR rows)
    rr = rr.with_columns([
        pl.when(pl.col("behavior_tag_final").is_not_null() & (pl.col("behavior_tag") == "Never Reconnected"))
          .then(pl.col("behavior_tag_final"))
          .otherwise(pl.col("behavior_tag"))
          .alias("behavior_tag")
    ]).drop(["behavior_tag_final"])

    return rr

def handleNulls(neverReconnectedLOS):
     # handle nulls in some dfs need to be converted into the right types
    # put this in a function or somehwere
    all_cols = set(col for df in neverReconnectedLOS for col in df.columns)
    # Determine the dtype per column based on non-null DFs
    col_type_map = {}
    for col in all_cols:
        for df in neverReconnectedLOS:
            if col in df.columns and df[col].dtype != pl.Null:
                col_type_map[col] = df[col].dtype
                break
        else:
            # If all DFs are Null for this col, default to Utf8
            col_type_map[col] = pl.Utf8

    # Now safely cast Null-only columns and add missing columns
    safe_dfs = []
    for df in neverReconnectedLOS:
        cols_to_cast = []
        for col, target_dtype in col_type_map.items():
            if col not in df.columns:
                # Add missing column
                df = df.with_columns(pl.lit(None).cast(target_dtype).alias(col))
            elif df[col].dtype == pl.Null:
                # Cast Null-only column to target dtype
                cols_to_cast.append(pl.col(col).cast(target_dtype))
        if cols_to_cast:
            df = df.with_columns(cols_to_cast)
        safe_dfs.append(df)
    return safe_dfs
