"""Inactive historical behaviour-assessment implementations."""

import polars as pl

class LegacyPVBehaviourMixin:
    def passiveIslanding(self, PRated, Reso250ms=None,
                         straddle_tol=1, vThreshold1=265, vThreshold2=275):
        OV1 = vThreshold1 - self.vDrop - self.vMeasError
        OV2 = vThreshold2 - self.vDrop - self.vMeasError
        P_DISCONNECT = self.powerMeasError * PRated
        volCol       = self.volCol

        voltage_cols    = [c for c in self.circuitData.columns if c.startswith(volCol)]
        power_cols      = [c for c in self.circuitData.columns if c.startswith("power") and not c.endswith("_next")]
        power_cols_next = [c for c in self.circuitData.columns if c.startswith("power") and c.endswith("_next")]
        
        df = self.circuitData.clone()

        df = df.with_columns([
            pl.any_horizontal(pl.col("^" + volCol + "_.*$") >= OV2).alias("OV2")
        ])
        df = df.with_columns([
            (pl.any_horizontal((pl.col("^" + volCol + "_.*$") >= OV1) & (pl.col("^" + volCol + "_.*$") < OV2)) & ~pl.col("OV2")).alias("OV1")
        ])

        strictFlag  = 0
        linientFlag = 0
        OV2Flag     = 0

        timeStepAvailable = df["local_tstamp"].n_unique()
        timesOverVoltage  = df.filter(pl.col("OV1") | pl.col("OV2")).n_unique()
        
        if timeStepAvailable > 0:
            if timesOverVoltage == 0:
                return {"hasData": True,
                        "hasOvervoltage": None,
                        "eventsSummary": None,
                        "complianceSummary": None,
                        "dataProcessed": None,
                        "diag_disconnect": pl.DataFrame(),  # keep keys present
                        "diag_reconnect":  pl.DataFrame(),
                        }, strictFlag, linientFlag, OV2Flag
        else:
            return {"hasData": False,
                    "hasOvervoltage": None,
                    "eventsSummary": None,
                    "complianceSummary": None,
                    "dataProcessed": None,
                    "diag_disconnect": pl.DataFrame(),
                    "diag_reconnect":  pl.DataFrame(),
                    }, strictFlag, linientFlag, OV2Flag
        
        df = df.with_columns(((pl.col("OV1")) & (~pl.col("OV1").shift(1).fill_null(False))).alias("enterOV1"))
        df = df.with_columns(pl.col("enterOV1").cast(pl.Int8).cum_sum().alias("latchOV1"))
        df = df.with_columns(pl.when(pl.col("OV1")).then(pl.col("latchOV1")).otherwise(None).alias("eventOV1"))

        df = df.with_columns([
            pl.when(pl.col("eventOV1").is_not_null())
            .then((pl.col("local_tstamp") - pl.first("local_tstamp").over("eventOV1")).dt.total_seconds())
            .otherwise(None)
            .alias("eventsTimeIncrement")
        ])

        df = df.with_columns([
            (pl.col("OV1") & (pl.col("eventsTimeIncrement") > 0) & (pl.col("eventsTimeIncrement") <= 1.0)).alias("T1"),
            (pl.col("OV1") & (pl.col("eventsTimeIncrement") > 1.0) & (pl.col("eventsTimeIncrement") <= 3.0)).alias("T2"),
        ])

        df = df.with_columns([
            pl.first("local_tstamp").over("eventOV1").alias("T0"),
            pl.max("eventsTimeIncrement").over("eventOV1").alias("L"),
            pl.max("T1").over("eventOV1").alias("hasT1"),
            pl.max("T2").over("eventOV1").alias("hasT2"),
            (pl.max("T1").over("eventOV1") | pl.max("T2").over("eventOV1")).alias("has3s"),
            pl.first("dt_next_s").over("eventOV1").alias("dt_to_next"),
            pl.first("ts_next").over("eventOV1").alias("Tnext"),
            pl.lit(Reso250ms).alias("Reso250ms")
        ])

        df_strict  = df.filter(pl.col("eventOV1").is_not_null() & pl.col("has3s"))
        df_lenient = df.filter(pl.col("eventOV1").is_not_null() & ~pl.col("has3s"))

        df = df.with_columns(((pl.col("OV2")) & (~pl.col("OV2").shift(1).fill_null(False))).alias("enterOV2")) \
               .with_columns(pl.col("enterOV2").cast(pl.Int8).cum_sum().alias("latchOV2")) \
               .with_columns(pl.when(pl.col("OV2")).then(pl.col("latchOV2")).otherwise(None).alias("eventOV2"))

        df = df.with_columns([
            pl.when(pl.col("eventOV2").is_not_null())
              .then((pl.col("local_tstamp") - pl.first("local_tstamp").over("eventOV2")).dt.total_seconds())
              .otherwise(None).alias("t_OV2")
        ])

        # strict window compliance
        df_strict = df_strict.with_columns([
            ((pl.col("local_tstamp") >  pl.col("T0") + pl.duration(seconds=3)) &
             (pl.col("local_tstamp") <= pl.col("T0") + pl.duration(seconds=300))).alias("in_5min"),
            pl.all_horizontal([(pl.col(c).fill_null(0) <= self.powerMeasError * PRated) for c in [*power_cols]]).alias("is_disc")
        ])
        df_strict = df_strict.with_columns([
            pl.when(pl.col("is_disc")).then(pl.col("eventsTimeIncrement")).otherwise(None)
              .min().over("eventOV1").alias("first_disc_t"),
        ])
        df_strict = df_strict.with_columns([
            pl.when((pl.col("eventsTimeIncrement") <= 3.0) & (~pl.col("is_disc"))).then(pl.col("eventsTimeIncrement")).otherwise(None)
              .max().over("eventOV1").alias("texp_before_3s"),
            pl.when((pl.col("eventsTimeIncrement") >= 3.0) & (pl.col("is_disc"))).then(pl.col("eventsTimeIncrement")).otherwise(None)
              .min().over("eventOV1").alias("tzero_after_3s"),
        ]).with_columns([
            (pl.col("tzero_after_3s") - pl.col("texp_before_3s")).alias("straddle_gap_s"),
        ])
        df_strict = df_strict.with_columns([
            (pl.col("eventsTimeIncrement") > 3.0).alias("AFTER3s"),
            (pl.col("eventsTimeIncrement") > 300.0).alias("AFTER5min"),
        ])
        if df_strict.height >= 1:
            strictFlag += 1

        df_strict = df_strict.with_columns([
            pl.when(~pl.col("OV1")).then(pl.lit(None))
            .when(pl.col("T1") & pl.col("is_disc")).then(pl.lit("non_compliant"))
            .when(pl.col("T1")).then(pl.lit("compliant"))
            .when(pl.col("T2") & pl.col("is_disc")).then(pl.lit("compliant"))
            .when(pl.col("T2") & ~pl.col("is_disc")).then(pl.lit("indeterminate"))
            .when(pl.col("AFTER3s") & (pl.col("first_disc_t") <= 3.0) & pl.col("is_disc")).then(pl.lit("compliant"))
            .when(pl.col("AFTER3s") & ~pl.col("AFTER5min") & pl.col("texp_before_3s").is_not_null() &
                  pl.col("tzero_after_3s").is_not_null() & (pl.col("straddle_gap_s") <= straddle_tol) & pl.col("is_disc")).then(pl.lit("compliant"))
            .when(pl.col("AFTER3s") & ~pl.col("AFTER5min") & pl.col("texp_before_3s").is_not_null() &
                  pl.col("tzero_after_3s").is_not_null() & (pl.col("straddle_gap_s") <= straddle_tol) & ~pl.col("is_disc")).then(pl.lit("non_compliant"))
            .when(pl.col("AFTER3s") & ~pl.col("AFTER5min") & pl.col("first_disc_t").is_null()).then(pl.lit("non_compliant"))
            .when(pl.col("AFTER3s") & ~pl.col("AFTER5min") & pl.col("first_disc_t").is_not_null() &
                  (pl.col("first_disc_t") > 3.0) & (pl.col("eventsTimeIncrement") < pl.col("first_disc_t"))).then(pl.lit("non_compliant"))
            .when(pl.col("AFTER3s") & ~pl.col("AFTER5min") & pl.col("first_disc_t").is_not_null() &
                  (pl.col("first_disc_t") > 3.0) & (pl.col("eventsTimeIncrement") == pl.col("first_disc_t")) & pl.col("is_disc")).then(pl.lit("late_response"))
            .when(pl.col("AFTER3s") & ~pl.col("AFTER5min") & pl.col("first_disc_t").is_not_null() &
                  (pl.col("first_disc_t") > 3.0) & (pl.col("eventsTimeIncrement") > pl.col("first_disc_t")) & pl.col("is_disc")).then(pl.lit("compliant"))
            .when(pl.col("AFTER3s") & ~pl.col("AFTER5min") & pl.col("first_disc_t").is_not_null() &
                  (pl.col("first_disc_t") > 3.0) & (pl.col("eventsTimeIncrement") >= pl.col("first_disc_t")) & ~pl.col("is_disc")).then(pl.lit("non_compliant"))
            .when(pl.col("AFTER5min")).then(
                pl.when(pl.col("first_disc_t").is_null()).then(pl.lit("non_compliant"))
                .when(pl.col("eventsTimeIncrement") > pl.col("first_disc_t")).then(
                    pl.when(pl.col("is_disc")).then(pl.lit("compliant")).otherwise(pl.lit("non_compliant"))
                )
                .when(pl.col("eventsTimeIncrement") == pl.col("first_disc_t")).then(pl.lit("late_response"))
                .otherwise(pl.lit("non_compliant"))
            )
            .otherwise(pl.lit(None))
            .alias("OV1ComplianceResult")
        ])

        df_lenient = (
            df_lenient.with_columns([
                pl.when(pl.col("Tnext").is_null() | pl.col("dt_to_next").is_null())
                  .then(pl.lit("indeterminate"))
                  .otherwise(
                      pl.when(pl.all_horizontal([(pl.col(c).fill_null(0) <= P_DISCONNECT) for c in power_cols]) |
                              pl.all_horizontal([(pl.col(c).fill_null(0) <= P_DISCONNECT) for c in power_cols_next]))
                        .then(pl.lit("compliant"))
                        .otherwise(
                            pl.when(pl.col("L") < pl.col("dt_to_next")).then(pl.lit("indeterminate"))
                            .otherwise(pl.lit("non_compliant"))
                        )
                  ).alias("OV1ComplianceResult")
            ])
        ).with_columns([
            (pl.all_horizontal([(pl.col(c).fill_null(0) <= P_DISCONNECT) for c in power_cols]) |
             pl.all_horizontal([(pl.col(c).fill_null(0) <= P_DISCONNECT) for c in power_cols_next])
            ).alias("is_disc")
        ])
        if df_lenient.height > 0:
            linientFlag += 1

        power_OV2_cols = [f"{c}_OV2" for c in power_cols_next]
        dfOV2 = df.filter(pl.col("eventOV2").is_not_null())
        dfOV2 = dfOV2.with_columns(
            pl.when(pl.col("Tnext_OV2").is_null() | pl.col("dt_to_next_OV2").is_null())
              .then(pl.lit("indeterminate"))
              .otherwise(
                  pl.when(pl.all_horizontal([(pl.col(c).fill_null(0) <= P_DISCONNECT).fill_null(False) for c in power_OV2_cols]))
                    .then(pl.lit("compliant"))
                    .otherwise(
                        pl.when(pl.col("L_OV2") < pl.col("dt_to_next_OV2")).then(pl.lit("indeterminate"))
                        .otherwise(pl.lit("non_compliant"))
                    )
              ).alias("OV2ComplianceResult")
        ).with_columns([
            pl.all_horizontal([(pl.col(c).fill_null(0) <= P_DISCONNECT) for c in power_cols]).alias("is_disc")
        ])
        if dfOV2.height > 0:
            OV2Flag += 1

        # Compact OV1 compliance summary (unchanged shape)
        base = pl.concat([
            df_strict.select(pl.col("OV1ComplianceResult").alias("ComplianceResult")),
            df_lenient.select(pl.col("OV1ComplianceResult").alias("ComplianceResult"))
        ], how="vertical").filter(pl.col("ComplianceResult").is_not_null())

        compliant_expr      = (pl.col("ComplianceResult") == "compliant").sum()
        non_compliant_expr  = (pl.col("ComplianceResult") == "non_compliant").sum()
        indeterminate_expr  = (pl.col("ComplianceResult") == "indeterminate").sum()
        late_response_expr  = (pl.col("ComplianceResult") == "late_response").sum()
        denominator_expr    = (compliant_expr + non_compliant_expr + indeterminate_expr + late_response_expr)

        complianceSummary = base.select([
            compliant_expr.alias("compliant"),
            (compliant_expr/denominator_expr*100).alias("compliant_pct"),
            non_compliant_expr.alias("non_compliant"),
            (non_compliant_expr/denominator_expr*100).alias("non_compliant_pct"),
            indeterminate_expr.alias("indeterminate"),
            (indeterminate_expr/denominator_expr*100).alias("indeterminate_pct"),
            late_response_expr.alias("late_response"),
            (late_response_expr/denominator_expr*100).alias("late_response_pct"),
        ])

        return {"hasData": True,
                "hasOvervoltage": True,
                "eventsSummary": None,
                "complianceSummary": complianceSummary,
                "dataProcessed": df,
                "diag_disconnect": pl.DataFrame(),  # diagnostics not used for OV1
                "diag_reconnect":  pl.DataFrame(),
                }, strictFlag, linientFlag, OV2Flag

    # ------------------------- LOS KPI (sustainedOperation) + reconnection + diagnostics -------------------------
    def sustainedOperation(self, PRated, V_NOM_MAX=258, volThreshParam = 'mean'):
        '''
        implementing clause 4.5.2 from the standard AS_NZ_4777.2_2020
        '''
        vThreshold   = V_NOM_MAX - self.vDrop - self.vMeasError
        P_DISCONNECT = self.powerMeasError * PRated
        volCol       = self.volCol

        voltage_cols    = [c for c in self.circuitData.columns if c.startswith(volCol)]
        power_cols      = [c for c in self.circuitData.columns if c.startswith("power") and not c.endswith("_next")]
        power_cols_next = [c for c in self.circuitData.columns if c.startswith("power") and c.endswith("_next")]
        df = self.circuitData.clone()

        # 10‑min rolling per phase
        for c in voltage_cols:
            rolled = (
                df
                .filter(pl.col(c).is_not_null())
                .with_columns(
                    pl.col(c).rolling_mean_by(by="local_tstamp", window_size="10m")
                    .alias(f"vmean_rolling_10m{c.replace(volCol, '', 1)}")
                )
                .select(["local_tstamp", f"vmean_rolling_10m{c.replace(volCol, '', 1)}"])
            )
            df = df.join(rolled, on="local_tstamp", how="left")

        timeStepAvailable = df["local_tstamp"].n_unique()
        vmean_cols = [c for c in df.columns if c.startswith("vmean_rolling_10m")]

        if volThreshParam == 'max':
            timeStepsExceedingThreshold = (
                df.filter(pl.any_horizontal([pl.col(c) > vThreshold for c in vmean_cols]))
                .select("local_tstamp").n_unique()
            )
        elif volThreshParam == 'mean':
            df = df.with_columns(
                pl.mean_horizontal([pl.col(c) for c in vmean_cols]).alias("avg_of_vmean_rolling_10m")
            )
            timeStepsExceedingThreshold = (
                df.filter(pl.col("avg_of_vmean_rolling_10m") > vThreshold)
                .select("local_tstamp").n_unique()
            )
        else:
            raise ValueError("volThreshParam should be 'mean' or 'max'")
        
        if timeStepAvailable > 0:
            percentageDataAboveThreshold = timeStepsExceedingThreshold/timeStepAvailable * 100
            if percentageDataAboveThreshold < self.minSamplesPercentage:
                return {"hasData": True,
                        "hasOvervoltage": None,
                        "eventsSummary": None,
                        "complianceSummary": None,
                        "circuitDataProcessed": None,
                        "reconnectionSummary": None,
                        "diag_disconnect": pl.DataFrame(),
                        "diag_reconnect":  pl.DataFrame()}
        else:
            return {"hasData": False,
                    "hasOvervoltage": None,
                    "eventsSummary": None,
                    "complianceSummary": None,
                    "circuitDataProcessed": None,
                    "reconnectionSummary": None,
                    "diag_disconnect": pl.DataFrame(),
                    "diag_reconnect":  pl.DataFrame()}

        # build over_vmax (eligibility)
        if volThreshParam == 'max':
            df = df.with_columns(
                pl.any_horizontal([pl.col(c) > vThreshold for c in vmean_cols]).alias("over_vmax")
            )
        elif volThreshParam == 'mean':
            df = df.with_columns(
                (pl.col("avg_of_vmean_rolling_10m") > vThreshold).fill_null(False).alias("over_vmax")
            )

        # enter/exit, latch
        df = df.with_columns([
            ((pl.col("over_vmax")) & (~pl.col("over_vmax").shift(1).fill_null(False))).alias("enter_over_vmax"),
            ((~pl.col("over_vmax")) & (pl.col("over_vmax").shift(1).fill_null(False))).alias("exit_over_vmax"),
        ])
        df = df.with_columns(pl.col("enter_over_vmax").cast(pl.Int8).cum_sum().alias("latch_id"))
        df = df.with_columns(
            pl.when(pl.col("over_vmax")).then(pl.col("latch_id")).otherwise(None).alias("active_event_id")
        )

        # disconnection proxy
        df = df.with_columns([
            pl.all_horizontal([(pl.col(c).fill_null(0) <= P_DISCONNECT) for c in power_cols]).alias("is_disc") 
        ])

        # KPI classification (entry/state)
        df = df.with_columns(
            pl.when(pl.col("enter_over_vmax"))
            .then(
                pl.when(pl.col("dt_next_s") <= self.timeWindowForClassification)
                .then(
                    pl.when(pl.all_horizontal([(pl.col(c).fill_null(0) <= P_DISCONNECT) |
                                               (pl.col(f"{c}_next").fill_null(0) <= P_DISCONNECT) for c in power_cols]))
                    .then(pl.lit("ok_entry")).otherwise(pl.lit("non_compliant"))
                )
                .when((pl.col("dt_next_s") > self.timeWindowForClassification) & (pl.col("dt_next_s") <= 60*5))
                .then(
                    pl.when(pl.all_horizontal([pl.col(c).fill_null(0) <= P_DISCONNECT for c in power_cols]))
                    .then(pl.lit("ok_entry"))
                    .when(pl.all_horizontal([pl.col(c).fill_null(0) <= P_DISCONNECT for c in power_cols_next]))
                    .then(pl.lit("late_response"))
                    .otherwise(pl.lit("non_compliant"))
                )
                .otherwise(pl.lit("indeterminate"))
            )
            .otherwise(pl.lit(None)).alias("entry_check")
        )

        df = df.with_columns(
            pl.when((pl.col("active_event_id").is_not_null()) &
                    (pl.col("over_vmax").shift(-1)) &
                    (~pl.col("enter_over_vmax")))
            .then(
                pl.when(pl.all_horizontal([(pl.col(c).fill_null(0) <= P_DISCONNECT) |
                                           (pl.col(f"{c}_next").fill_null(0) <= P_DISCONNECT) for c in power_cols]))
                .then(pl.lit("ok_state")).otherwise(pl.lit("state_violation"))
            )
            .otherwise(pl.lit(None)).alias("state_check")
        )

        # ---------- Your original reconnection pipeline (unchanged) ----------
        # (I’m keeping your event metrics, reconnect detection, and the wide 'reconnection_result'
        # exactly as you had them, so main.py continues to work.)
        # NOTE: For brevity here, keep your existing large block that builds:
        #   event_metrics, reconnect_during, disc_values, rec1_values, intermittent_switching,
        #   eligible_events, reconnect_after (asof), rec_after_values, disconnect_lists_clean,
        #   and final reconnection_result with lists and time metrics.
        # Paste your original reconnection block here unmodified.
        # --------------------------------------------------------------------

        # LOS KPI event summary
        n_events     = df.filter(pl.col("enter_over_vmax")).height
        n_over_rows  = df.filter(pl.col("over_vmax")).height
        n_state_rows = df.filter(pl.col("state_check").is_not_null()).height
        n_singleton_events = df.filter(
            (pl.col("over_vmax")) &
            (~pl.col("over_vmax").shift(1).fill_null(False)) &
            (~pl.col("over_vmax").shift(-1).fill_null(False))
        ).height
        if n_over_rows != n_state_rows + 2*n_events - n_singleton_events:
            raise ValueError ("This should be the same")

        disconnected_mask = pl.any_horizontal([
            (pl.col(c).fill_null(0) <= P_DISCONNECT) | (pl.col(f"{c}_next").fill_null(0) <= P_DISCONNECT)
            for c in power_cols
        ])
        max_voltage = pl.max_horizontal([pl.col(c) for c in vmean_cols])
        df = df.with_columns(
            pl.when(pl.col("active_event_id").is_not_null() & disconnected_mask)
            .then(max_voltage)
            .otherwise(None)
            .alias("disc_voltage")
        )

        eventsSummary = (
            df.filter(pl.col("active_event_id").is_not_null())
            .group_by("active_event_id")
            .agg([
                pl.col("site_id").first().alias("site_id"),
                (pl.col("entry_check") == "ok_entry").sum().alias("entry_compliant_count"),
                (pl.col("entry_check") == "non_compliant").sum().alias("entry_violation_count"),
                (pl.col("entry_check") == "late_response").sum().alias("late_response_count"),
                (pl.col("entry_check") == "indeterminate").sum().alias("indeterminate_entry_count"),
                (pl.col("state_check") == "state_violation").sum().alias("state_violation_count"),
                (pl.col("state_check") == "ok_state").sum().alias("state_compliant_count"),
                (pl.col("disc_voltage").filter(pl.col("disc_voltage").is_not_null())
                                    .first().alias("first_disc_voltage"))
            ])
        )

        compliant_expr = (pl.col("entry_compliant_count") + pl.col("state_compliant_count")).sum()
        compliance_pct = (compliant_expr)/(n_over_rows-n_events+n_singleton_events)*100
        non_compliant_expr = (pl.col("entry_violation_count") + pl.col("state_violation_count")).sum()

        complianceSummary = eventsSummary.select([
            pl.col("site_id").first().alias("site_id"),
            compliant_expr.alias("compliant"),
            compliance_pct.alias("compliance_pct"),
            non_compliant_expr.alias("non_compliant"),
            pl.col("indeterminate_entry_count").sum().alias("indeterminate"),
            pl.col("late_response_count").sum().alias("late_response"),
        ])

        # ---------------------- Diagnostics (safe) ----------------------
        # Build mean-of-rolled 10-min & instantaneous snapshots without scope issues
        v10m_avg_snapshot  = pl.mean_horizontal([pl.col(c) for c in vmean_cols]) if vmean_cols else pl.lit(None)
        vinst_max_snapshot = pl.max_horizontal([pl.col(c) for c in voltage_cols]) if voltage_cols else pl.lit(None)

        # Disconnect transitions: is_disc False -> True within each active_event_id
        disc_trans = (
            df.select("active_event_id", "local_tstamp", "is_disc")
              .with_columns(pl.col("is_disc").shift(1).over("active_event_id").alias("_prev_disc"))
              .with_columns(pl.coalesce([pl.col("_prev_disc"), pl.lit(False)]).alias("_prev_disc"))
              .filter(pl.col("active_event_id").is_not_null() & (~pl.col("_prev_disc")) & pl.col("is_disc"))
              .select(["active_event_id", pl.col("local_tstamp").alias("ts_disc")])
        )

        # --- Disconnect snapshots (10‑min mean + instantaneous) ---
        _disc_snap = (
            df.with_columns([
                v10m_avg_snapshot.alias("v10m_here"),
                vinst_max_snapshot.alias("vinst_here"),
            ])
            .select(["active_event_id", "local_tstamp", "v10m_here", "vinst_here"])
            .join(
                disc_trans,  # False -> True transition of is_disc inside active_event_id
                left_on=["active_event_id", "local_tstamp"],
                right_on=["active_event_id", "ts_disc"],
                how="inner",
            )
            # IMPORTANT: the right key ('ts_disc') is not kept by Polars; use the left key instead
            .sort(["active_event_id", "local_tstamp"])
            .group_by("active_event_id")
            .agg([
                # use the left key after the join
                pl.col("local_tstamp").alias("disc_ts_list"),
                pl.col("v10m_here").alias("disc_v10m_list"),
                pl.col("vinst_here").alias("disc_vinst_list"),
            ])
        )

        over_tbl = (
            df.filter(pl.col("over_vmax"))
              .group_by("active_event_id")
              .agg([
                  pl.col("local_tstamp").max().alias("t_last_event"),
                  pl.col("is_disc").any().alias("had_disconnect"),
                  pl.col("local_tstamp").filter(pl.col("is_disc")).min().alias("t_disc_first")
              ])
        )
        connected_rows = df.filter(~pl.col("is_disc")).select(pl.col("local_tstamp").alias("ts_connected")).sort("ts_connected")
        reconnect_after = (
            over_tbl.filter(pl.col("had_disconnect") & ~pl.col("t_last_event").is_null())
                    .sort("t_last_event")
                    .join_asof(connected_rows, left_on="t_last_event", right_on="ts_connected", strategy="forward")
                    .rename({"ts_connected": "t_rec"})
                    .select(["active_event_id", "t_rec"])
        )

        _rec_after_v10m = (
            df.with_columns(v10m_avg_snapshot.alias("v10m_here"))
              .select(["active_event_id","local_tstamp","v10m_here"])
              .join(reconnect_after, left_on=["active_event_id","local_tstamp"], right_on=["active_event_id","t_rec"], how="inner")
              .group_by("active_event_id")
              .agg(pl.col("v10m_here").first().alias("v10m_rec_after"))
        )
        _rec_after_vinst = (
            df.with_columns(vinst_max_snapshot.alias("vinst_here"))
              .select(["active_event_id","local_tstamp","vinst_here"])
              .join(reconnect_after, left_on=["active_event_id","local_tstamp"], right_on=["active_event_id","t_rec"], how="inner")
              .group_by("active_event_id")
              .agg(pl.col("vinst_here").first().alias("vinst_rec_after"))
        )

        diag_disconnect = (
            _disc_snap
            .join(df.select("site_id","active_event_id").unique(), on="active_event_id", how="left")
            .select("site_id","active_event_id","disc_ts_list","disc_v10m_list","disc_vinst_list")
        )
        diag_reconnect = (
            reconnect_after
            .join(_rec_after_v10m, on="active_event_id", how="left")
            .join(_rec_after_vinst, on="active_event_id", how="left")
            .join(df.select("site_id","active_event_id").unique(), on="active_event_id", how="left")
            .select("site_id","active_event_id",
                    pl.col("t_rec").alias("rec_ts_after"),
                    "v10m_rec_after","vinst_rec_after")
        )
        # ---------------------------------------------------------------

        return {"hasData": True,
                "hasOvervoltage": True,
                "eventsSummary": eventsSummary,
                "complianceSummary": complianceSummary,
                "circuitDataProcessed": df,
                "reconnectionSummary": None,     # <-- replace with your original reconnection_result if you pasted it above
                "diag_disconnect": diag_disconnect,
                "diag_reconnect":  diag_reconnect}
        
        # For True runs only, compute total duration per run
        # calculates how long did vmean_rolling_10m
        # useful for later analysis - not now, maybe hleful for intermittent compliance?
        run_durations = (
            df.filter(pl.col("active_event_id").is_not_null())
            .group_by("active_event_id")
            .agg(
                pl.col("dt_next_s").sum().alias("run_dt_s"),
                pl.count().alias("num_rows")
            )
        )





        # if non-compliant more than 10% of the times (oot of the times when it exceed V threshold)
            # then non compliant 

            # else compliant
    
        # do some classification

        # does nothing

        # disconnects and does not come back online
        # check the clause when it should come back online

        # check if it disconnects and come backs online - intermittent compliant

        # return self.circuitData

def check_vw_endpoints(
    circuitData: pl.DataFrame,
    *,
    p_rated_active: float,              # your PRated (active power rating, in W)
    voltage_col: str = "vmean",
    power_col: str = "power",
    vw1: float = 253.0,
    vw2: float = 260.0,
    v_nominal: float = 230.0,           # nominal system voltage
    v_tol_frac: float = 0.01,           # ±1% of nominal -> ±2.3 V default
    p_tol_frac: float = 0.04,           # ±4% of S_rated (default per spec)
    pf: float = 1.0,                    # assumed power factor to derive S_rated ~ P_rated / pf
    daylight_frac: float = 0.20,        # only consider samples with P >= 20% of P_rated
    strict_upper_at_vw1: bool = True    # cap acceptance at <= 100% near VW1
):
    """
    Endpoint-only compliance check for Volt–Watt:
      - Uses voltage acceptance windows around VW1 and VW2: ± v_tol_frac * v_nominal.
      - Uses power tolerance ± p_tol_frac * S_rated (S_rated ≈ P_rated_active / pf).
      - Keeps pre-processing (time parsing/sorting/null handling/unit alignment) outside.

    Returns
    -------
    summary : dict
        Counts, medians, compliance %, voltage bands used, tolerance equivalents in watts.
    w1_df : pl.DataFrame
        Samples considered near VW1 (with columns 'p_frac' and 'compliant_w1').
    w2_df : pl.DataFrame
        Samples considered near VW2 (with columns 'p_frac' and 'compliant_w2').
    """

    # --- Acceptance bands (voltage) ---
    v_acc = v_tol_frac * v_nominal             # e.g., 0.01 * 230 = 2.3 V
    vw1_low, vw1_high = vw1 - v_acc, vw1 + v_acc
    vw2_low, vw2_high = vw2 - v_acc, vw2 + v_acc

    # --- Power tolerance based on S_rated ---
    if pf <= 0:
        raise ValueError("pf must be > 0.")
    s_rated_apparent = p_rated_active / pf
    
    # filter data when both voltage and power is avaialble?
    circuitData = circuitData.with_columns(
    pl.col(voltage_col).cast(pl.Float64, strict=False),
    pl.col(power_col).cast(pl.Float64, strict=False),
    pl.col('duration').cast(pl.Float64, strict=False))

    # --- Normalize power and daylight filter ---
    df = circuitData.with_columns((pl.col(power_col) / p_rated_active).alias("p_frac"))
    df_day = df.filter(pl.col("p_frac") >= daylight_frac) # filter out data with greater than 20% gen

    # --- Windows at VW1, filter the lower tolerance band ---
    w1 = df_day.filter((pl.col(voltage_col) >= vw1_low) & (pl.col(voltage_col) <= vw1_high))
    # --- For VW2, filter above the tolerance to see if it is non-conforming ---
    w2 = df_day.filter((pl.col(voltage_col) >= vw2_high))

    # --- Expected percentage of active power through apparent power at endpoints ---
    exp_w1, exp_w2 = 1.0*s_rated_apparent, 0.20*s_rated_apparent

    # --- Compliance logic ---
    # VW1: cap at 100% if strict_upper_at_vw1=True
    if strict_upper_at_vw1:
        w1 = w1.with_columns((
            # (pl.col("p_frac") >= (exp_w1 - p_tol_watts / p_rated_active)) & # greater than 96% generation, idk if this matters though?
            (pl.col(power_col) <= exp_w1) # less than equal to 100% generation
        ).alias("compliant_w1"))
    # else:
    #     w1 = w1.with_columns((
    #         (pl.col("p_frac") >= (exp_w1 - p_tol_watts / p_rated_active)) &
    #         (pl.col("p_frac") <= (exp_w1 + p_tol_watts / p_rated_active))
    #     ).alias("compliant_w1"))

    # VW2: accept within 20% ± tolerance
    w2 = w2.with_columns((
        pl.col(power_col) <= exp_w2 + p_tol_frac * s_rated_apparent # the measurement tol is meant to be % of apparent power in active
        # .is_between(
        #     exp_w2 - p_tol_watts / p_rated_active,
        #     exp_w2 + p_tol_watts / p_rated_active
    ).alias("compliant_w2"))

    # --- Summaries (guard against empty windows) ---
    summary = {
        # "vw1_band_V": (vw1_low, vw1_high),
        # "vw2_band_V": (vw2_low, vw2_high),
        # "v_tol_frac_of_nominal": v_tol_frac,
        # "p_tol_frac_of_Srated": p_tol_frac,
        # "pf_assumed": pf,
        # "p_rated_active_W": float(p_rated_active),
        # "s_rated_apparent_W": float(s_rated_apparent),
        # "daylight_filter_frac_of_Prated": daylight_frac,

        "samples_near_vw1": int(w1.height), # test samples around w1
        "samples_near_vw2": int(w2.height), # test smaples around w2
        # "median_pfrac_vw1": (w1.select(pl.median("p_frac")).item() if w1.height else None),
        # "median_pfrac_vw2": (w2.select(pl.median("p_frac")).item() if w2.height else None),
        "pct_compliant_vw1": (100.0 * w1.filter(pl.col("compliant_w1")).height / w1.height) if w1.height else None, # percentage in the test sample complying
        "pct_compliant_vw2": (100.0 * w2.filter(pl.col("compliant_w2")).height / w2.height) if w2.height else None,
    }

    return summary, w1, w2

def inferEvmActivation(
    df: pl.DataFrame,
    PRated: float,                     # systen AC capacity
    local_tstamp: str,                 # e.g.,smoothWindow "local_tstamp" (Datetime, tz-aware or naive)
    # --- tunables, adjust the times and duration below according to the reports ---
    daylightStart: str = "08:30",      # local time window start (HH:MM)
    daylightEnd: str   = "17:30",      # local time window end   (HH:MM)
    # thresholds tuned for PV-only inference (adjust if needed)
    # flatFracOfDaymax: float = 0.70,  # flat-top transitions to call cycling
    curtailmentFrac: float = 0.60,     # “flat-top” threshold vs capacity
    nearZeroFrac: float   = 0.05,      # near-zero threshold vs capacity
    # cyclingCountMin: int  = 3,       # min on/off transitions to call it cycling
    smoothWindow: int      = 3,        # rolling median window (samples) to de-noise
    sustainedZeroMinMinutes: int = 240 # duration for sustained EVM
    # flatTopMinMinutes: int = 60  # minutes: long curtailment block
) -> dict:
    """
    Returns a dict with booleans:
      - sustained_disconnect
      - flat_top
      - cycling
      - evm_likely
    Using PV power only (no voltage/status).
    """

    if "power" not in df.columns or "local_tstamp" not in df.columns:
        raise ValueError("Data frame must contain the specified PRated and local_tstamp.")

    # Convert HH:MM strings nearZeroFracinto minutes-from-midnight for filtering
    def _to_minutes(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    start_min = _to_minutes(daylightStart)
    end_min   = _to_minutes(daylightEnd)

    # --- sort and derive minutes-from-midnight ---
    df = (df.sort(local_tstamp)
          .with_columns([
              (pl.col(local_tstamp).dt.hour() * 60 + pl.col(local_tstamp).dt.minute()).alias("_mins")
          ]))

    # add minutes from midnight for each timestamp
    # minutes_from_midnight = (
    #     (pl.col(local_tstamp).dt.hour() * 60 + pl.col(local_tstamp).dt.minute())
    #     .alias("_mins")
    # )

    # Thresholds: mostly interested in near zero
    # disregard analysis for flatTop
    flatTopThreshold  = curtailmentFrac * float(PRated) # this is only relevant for PQ events based on the voltage
    nearZeroThreshold = nearZeroFrac * float(PRated) # assessing for curtailment

    # add Boolean for if it is daylight
    labeled_df = df.with_columns([pl.col("_mins").is_between(start_min, end_min, closed="both").alias("isDaylight")])
    # add a column for smoothed power to avoid noise
    labeled_df = labeled_df.with_columns(pl.col("power").rolling_mean(window_size=smoothWindow, min_periods = 1).alias("powerSmoothed"))
    # add various labels for inverter behaviour
    labeled_df = labeled_df.with_columns(
                                       pl.when(pl.col("isDaylight") & pl.col("powerSmoothed")<= nearZeroThreshold)
                                         .then(pl.lit("Disconnect"))
                                         .when(pl.col("isDaylight") & pl.col("powerSmoothed")<= flatTopThreshold)
                                         .then(pl.lit("Curtailment"))
                                         .otherwise(pl.lit("Normal")).alias("label"))


    day = labeled_df.filter(pl.col("isDaylight"))
    if day.is_empty():
        return {
            "sustained_disconnect": False,
            "near_zero_runs": [],
            "evm_likely": False,
        }

    # --- contiguous near-zero run detection using timestamps (real minutes) ---
    # flag rows that are near-zero
    nzFlag = day.select((pl.col("powerSmoothed") <= nearZeroThreshold).alias("nearZeroFlag"))

    # build run IDs by counting changes in the boolean flag
    nzChange = nzFlag.select(pl.col("nearZeroFlag").cast(pl.Int8).diff().abs().fill_null(0).alias("_change"))
    nzRunId  = nzChange.select(pl.col("_change").cumsum().alias("_run_id"))
    dayRuns  = day.hstack(nzFlag).hstack(nzRunId)

    # summarize each near-zero run
    nzRuns = (
        dayRuns.filter(pl.col("nearZeroFlag"))
               .group_by("_run_id")
               .agg([
                    pl.col(local_tstamp).min().alias("start"),
                    pl.col(local_tstamp).max().alias("end"),
                    (pl.col(local_tstamp).max() - pl.col(local_tstamp).min()).alias("dur"),
               ])
               .with_columns((pl.col("dur").dt.nanoseconds() / 1e9 / 60).alias("mins"))
               .select(["start", "end", "mins"])
    )

    # convert to Python list of dicts for easy downstream handling
    near_zero_runs = [
        {
            "start": r[0],
            "end":   r[1],
            "mins":  float(r[2]) if r[2] is not None else 0.0
        }
        for r in nzRuns.iter_rows()
    ]

    sustainedZeroMinMinutes = 240
    sustained_disconnect = any(run["mins"] >= sustainedZeroMinMinutes for run in near_zero_runs)

    # for now, evm_likely mirrors sustained disconnect; later you can OR with other signals
    evm_likely = sustained_disconnect # modify this later if islanding or other operation as well based on what you do

    return {
        "sustained_disconnect": sustained_disconnect,
        "near_zero_runs": near_zero_runs,
        "evm_likely": evm_likely,
    }

    # sustained disconnect if any run is long enough


    # # Create contiguous block ids per label to compute durations
    # # A new block starts whenever the label changes.
    # labeled_df = labeled_df.with_columns([
    #     (pl.col("label") != pl.col("label").shift(1)).fill_null(True).cast(pl.Int8).cumsum().alias("_block_id")
    # ])

    # # Compute block durations (using first and last timestamps in each block)
    # blocks = (
    #     labeled_df.group_by(["_block_id", "label"]).agg([
    #         pl.col(local_tstamp).min().alias("block_start"),
    #         pl.col(local_tstamp).max().alias("block_end"),
    #         (pl.col(local_tstamp).max() - pl.col(local_tstamp).min()).alias("block_duration"),
    #     ])
    #     .with_columns([
    #         # minutes as float for easy comparisons
    #         (pl.col("block_duration").dt.nanoseconds() / 1e9 / 60.0).alias("block_minutes")
    #     ])
    # )

    # # Flags
    # sustained_disconnect_detected = bool(
    #     blocks.filter((pl.col("label") == "Disconnect") & (pl.col("block_minutes") >= sustained_zero_min)).height > 0
    # )

    # unsustained_disconnect_detected = bool(
    #     blocks.filter(
    #         (pl.col("label") == "Disconnect") &
    #         (pl.col("block_minutes") >= unsustained_zero_min) &
    #         (pl.col("block_minutes") < sustained_zero_min)
    #     ).height > 0
    # )

    # flat_top_detected = bool(
    #     blocks.filter((pl.col("label") == "Curtailment") & (pl.col("block_minutes") >= flatTopMinMinutes)).height > 0
    # )

    # # Cycling: count transitions into/out of Disconnect
    # disconnect_flag = labeled_df.select(
    #     (pl.col("label") == "Disconnect").cast(pl.Int8).alias("_disc")
    # )
    # # transitions = sum(abs(diff(_disc)) > 0)
    # transitions = int(
    #     disconnect_flag.select(
    #         (pl.col("_disc").diff().abs() > 0).fill_null(False).cast(pl.Int8).sum()
    #     ).item()
    # )
    # cycling_detected = transitions >= cyclingCountMin

    # # Final inference: EVM-like if we see strong curtailment/disconnect patterns
    # evm_likely = bool(
    #     flat_top_detected or
    #     sustained_disconnect_detected or
    #     (unsustained_disconnect_detected and cycling_detected)
    # )

    # summary = {
    #     "flat_top_detected": flat_top_detected,
    #     "sustained_disconnect_detected": sustained_disconnect_detected,
    #     "unsustained_disconnect_detected": unsustained_disconnect_detected,
    #     "cycling_detected": cycling_detected,
    #     "evm_likely": evm_likely,
    # }

    # # Drop helper columns if you prefer a clean output (comment out if you want to inspect)
    # labeled_df = labeled_df.drop(["_mins", "_isDaylight", "_block_id"])

    # return labeled_df, summary



#############

# import polars as pl

# def evm_likely_single_inverter(
#     df: pl.DataFrame,
#     ts_col: str,            # timestamp column (local time)
#     power_col: str,         # AC power, e.g., "power_kW"
#     system_capacity_kw: float,   # inverter AC rating
#     daylightStart: str = "08:30",
#     daylightEnd: str   = "17:30",
#     # thresholds tuned for PV-only inference (adjust if needed)
#     near_zero_frac: float   = 0.05,   # near-zero threshold vs capacity
#     flat_frac_of_daymax: float = 0.70,  # flat-top threshold vs today's max (not capacity)
#     flat_min_minutes: int   = 60,     # minimum duration to call it a flat-top
#     sustained_zero_min: int = 240,    # minutes of zero output to call sustained disconnect
#     cycling_min_events: int = 3,      # min on/off transitions to call cycling
#     smoothWindow: int      = 3       # rolling median window (samples)
# ) -> dict:
#     """
#     Returns a dict with booleans:
#       - sustained_disconnect
#       - flat_top
#       - cycling
#       - evm_likely
#     Using PV power only (no voltage/status).
#     """

#     # --- 1) Prepare daylight mask ---
#     def _to_minutes(hhmm: str) -> int:
#         h, m = hhmm.split(":")
#         return int(h) * 60 + int(m)

#     start_min = _to_minutes(daylightStart)
#     end_min   = _to_minutes(daylightEnd)

#     df = df.sort(ts_col).with_columns([
#         (pl.col(ts_col).dt.hour() * 60 + pl.col(ts_col).dt.minute()).alias("_mins"),
#         ((pl.col("_mins") >= start_min) & (pl.col("_mins") <= end_min)).alias("_day"),
#         pl.col(power_col).rolling_median(window_size=smoothWindow, min_periods=1).alias("p_smooth"),
#     ])

#     # Focus on daylight rows
#     day = df.filter(pl.col("_day"))

#     if day.is_empty():
#         return {
#             "sustained_disconnect": False,
#             "flat_top": False,
#             "cycling": False,
#             "evm_likely": False,
#         }

#     # --- 2) Simple thresholds ---
#     near_zero_thr = near_zero_frac * float(system_capacity_kw)

#     # Use today's *smoothed* max to define flat-top threshold (robust w.r.t. site specifics)
#     day_max = float(day.select(pl.col("p_smooth").max()).item())
#     flat_thr = flat_frac_of_daymax * day_max

#     # --- 3) Sustained disconnect: near-zero power for a long time ---
#     # Compute contiguous runs of near-zero by detecting changes in the condition.
#     nz_flag = day.select((pl.col("p_smooth") <= near_zero_thr).alias("_nz"))
#     # transitions where condition changes (True<->False)
#     nz_transitions = nz_flag.select(
#         (pl.col("_nz").diff().fill_null(False) != pl.col("_nz")).cast(pl.Int8).cumsum().alias("_run_id")
#     )
#     # attach run_id back
#     day_runs = day.hstack(nz_transitions)

#     # duration per run where _nz == True
#     nz_runs = (
#         day_runs.filter(pl.col("_nz"))
#                 .group_by("_run_id")
#                 .agg([
#                     pl.col(ts_col).min().alias("start"),
#                     pl.col(ts_col).max().alias("end"),
#                     (pl.col(ts_col).max() - pl.col(ts_col).min()).alias("dur"),
#                 ])
#                 .with_columns((pl.col("dur").dt.nanoseconds() / 1e9 / 60).alias("mins"))
#     )
#     sustained_disconnect = bool(nz_runs.filter(pl.col("mins") >= sustained_zero_min).height > 0)

#     # --- 4) Flat-top curtailment: power sits low (below flat_thr) for long duration ---
#     flat_flag = day.select((pl.col("p_smooth") <= flat_thr).alias("_flat"))
#     flat_transitions = flat_flag.select(
#         (pl.col("_flat").diff().fill_null(False) != pl.col("_flat")).cast(pl.Int8).cumsum().alias("_run_id_flat")
#     )
#     day_flat = day.hstack(flat_transitions)

#     flat_runs = (
#         day_flat.filter(pl.col("_flat"))
#                 .group_by("_run_id_flat")
#                 .agg([
#                     pl.col(ts_col).min().alias("start"),
#                     pl.col(ts_col).max().alias("end"),
#                     (pl.col(ts_col).max() - pl.col(ts_col).min()).alias("dur"),
#                 ])
#                 .with_columns((pl.col("dur").dt.nanoseconds() / 1e9 / 60).alias("mins"))
#     )
#     flat_top = bool(flat_runs.filter(pl.col("mins") >= flat_min_minutes).height > 0)

#     # --- 5) Cycling: repeated toggles into/out of near-zero during daylight ---
#     # Count True<->False edges on the near-zero flag
#     cycling_events = int(
#         nz_flag.select(
#             (pl.col("_nz").diff().abs().fill_null(False)).cast(pl.Int8).sum()
#         ).item()
#     )
#     cycling = cycling_events >= cycling_min_events

#     # --- 6) Final inference: any strong evidence -> EVM likely
#     evm_likely = bool(sustained_disconnect or flat_top or cycling)

#     return {
#         "sustained_disconnect": sustained_disconnect,
#         "flat_top": flat_top,
#         "cycling": cycling,
#         "evm_likely": evm_likely,
#     }


# import polars as pl

# def inferEvmActivation(
#     df: pl.DataFrame,
#     PRated: float,                  # system AC capacity
#     local_tstamp: str,              # e.g., "local_tstamp" (Datetime, tz-aware or naive)
#     # --- tunables, adjust the times and duration below according to the reports ---
#     daylightStart: str = "08:30",   # local time window start (HH:MM)
#     daylightEnd: str   = "17:30",   # local time window end   (HH:MM)
#     # thresholds tuned for PV-only inference (adjust if needed)
#     flatFracOfDaymax: float = 0.70, # flat-top threshold vs today's own max
#     curtailmentFrac: float  = 0.60, # flat-top threshold vs capacity (kept for reference)
#     nearZeroFrac: float     = 0.05, # near-zero threshold vs capacity
#     cyclingCountMin: int    = 3,    # min on/off transitions to call cycling
#     smoothWindow: int       = 3,    # rolling mean window (samples) to de-noise
#     flatTopMinMinutes: int  = 60,   # minutes: long curtailment block
#     sustainedZeroMinMinutes: int = 240  # minutes: long near-zero block to call sustained disconnect
# ) -> dict:
#     """
#     Returns a dict with booleans:
#       - sustained_disconnect
#       - flat_top
#       - cycling
#       - evm_likely
#     Using PV power only (no voltage/status).
#     """

#     # --- Basic validation ---
#     if "power" not in df.columns or local_tstamp not in df.columns:
#         raise ValueError("Data frame must contain 'power' and the specified local_tstamp column.")

#     # --- Helpers ---
#     def _to_minutes(hhmm: str) -> int:
#         h, m = hhmm.split(":")
#         return int(h) * 60 + int(m)

#     start_min = _to_minutes(daylightStart)
#     end_min   = _to_minutes(daylightEnd)

#     # --- Prepare time features and smoothing ---
#     df = (
#         df.sort(local_tstamp)
#           .with_columns([
#               (pl.col(local_tstamp).dt.hour() * 60 + pl.col(local_tstamp).dt.minute()).alias("_mins")
#           ])
#     )

#     # Handle windows that may wrap past midnight (not typical here, but safe)
#     isDaylightExpr = pl.when(end_min >= start_min) \
#         .then(pl.col("_mins").is_between(start_min, end_min, closed="both")) \
#         .otherwise((pl.col("_mins") >= start_min) | (pl.col("_mins") <= end_min))

#     df = df.with_columns([
#         isDaylightExpr.alias("isDaylight"),
#         pl.col("power").rolling_mean(window_size=smoothWindow, min_periods=1).alias("powerSmoothed"),
#     ])

#     # --- Daylight subset ---
#     day = df.filter(pl.col("isDaylight"))
#     if day.is_empty():
#         return {
#             "sustained_disconnect": False,
#             "flat_top": False,
#             "cycling": False,
#             "evm_likely": False,
#         }

#     # --- Thresholds ---
#     nearZeroThreshold = nearZeroFrac * float(PRated)
#     # Keep your capacity-based flat-top threshold as a reference
#     flatTopThreshold  = curtailmentFrac * float(PRated)
#     # Preferred: flat-top defined relative to *today's own* smoothed max
#     dayMax = float(day.select(pl.col("powerSmoothed").max()).item())
#     flatTopThresholdDaymax = flatFracOfDaymax * dayMax

#     # --- Optional quick labeling for QA (Disconnect/Curtailment/Normal) ---
#     # Keeping your 'label' column name and logic, but using the daymax-based flat threshold.
#     labeled_df = df.with_columns([
#         pl.when(pl.col("isDaylight") & (pl.col("powerSmoothed") <= nearZeroThreshold))
#           .then(pl.lit("Disconnect"))
#           .when(pl.col("isDaylight") & (pl.col("powerSmoothed") <= flatTopThresholdDaymax))
#           .then(pl.lit("Curtailment"))
#           .otherwise(pl.lit("Normal")).alias("label")
#     ])

#     # --- Contiguous run detection: near-zero ---
#     nzFlag = day.select((pl.col("powerSmoothed") <= nearZeroThreshold).alias("nearZeroFlag"))
#     nzChange = nzFlag.select(pl.col("nearZeroFlag").cast(pl.Int8).diff().abs().fill_null(0).alias("_change"))
#     nzRunId = nzChange.select(pl.col("_change").cumsum().alias("_runId"))
#     dayRuns = day.hstack(nzFlag).hstack(nzRunId)

#     nzRuns = (
#         dayRuns.filter(pl.col("nearZeroFlag"))
#                .group_by("_run_id")  # Note: cumsum alias becomes "_run_id" internally; safe to alias below if needed
#                .agg([
#                     pl.col(local_tstamp).min().alias("start"),
#                     pl.col(local_tstamp).max().alias("end"),
#                     (pl.col(local_tstamp).max() - pl.col(local_tstamp).min()).alias("dur"),
#                ])
#                .with_columns((pl.col("dur").dt.nanoseconds() / 1e9 / 60).alias("mins"))
#     )

#     # In case Polars renames the cumsum column differently, ensure it exists:
#     if "_run_id" not in nzRuns.columns:
#         # rebuild with explicit name
#         nzRunId = nzChange.select(pl.col("_change").cumsum().alias("_run_id"))
#         dayRuns = day.hstack(nzFlag).hstack(nzRunId)
#         nzRuns = (
#             dayRuns.filter(pl.col("nearZeroFlag"))
#                    .group_by("_run_id")
#                    .agg([
#                         pl.col(local_tstamp).min().alias("start"),
#                         pl.col(local_tstamp).max().alias("end"),
#                         (pl.col(local_tstamp).max() - pl.col(local_tstamp).min()).alias("dur"),
#                    ])
#                    .with_columns((pl.col("dur").dt.nanoseconds() / 1e9 / 60).alias("mins"))
#         )

#     sustained_disconnect = bool(nzRuns.filter(pl.col("mins") >= sustainedZeroMinMinutes).height > 0)

#     # --- Contiguous run detection: flat-top (vs today's own max) ---
#     flatFlag = day.select((pl.col("powerSmoothed") <= flatTopThresholdDaymax).alias("flatFlag"))
#     flatChange = flatFlag.select(pl.col("flatFlag").cast(pl.Int8).diff().abs().fill_null(0).alias("_change"))
#     flatRunId = flatChange.select(pl.col("_change").cumsum().alias("_run_id_flat"))
#     dayFlatRuns = day.hstack(flatFlag).hstack(flatRunId)

#     flatRuns = (
#         dayFlatRuns.filter(pl.col("flatFlag"))
#                    .group_by("_run_id_flat")
#                    .agg([
#                         pl.col(local_tstamp).min().alias("start"),
#                         pl.col(local_tstamp).max().alias("end"),
#                         (pl.col(local_tstamp).max() - pl.col(local_tstamp).min()).alias("dur"),
#                    ])
#                    .with_columns((pl.col("dur").dt.nanoseconds() / 1e9 / 60).alias("mins"))
#     )
#     flat_top = bool(flatRuns.filter(pl.col("mins") >= flatTopMinMinutes).height > 0)

#     # --- Cycling: count toggles into/out of near-zero during daylight ---
#     cyclingEvents = int(
#         nzFlag.select(pl.col("nearZeroFlag").cast(pl.Int8).diff().abs().fill_null(0).sum()).item()
#     )
#     cycling = cyclingEvents >= cyclingCountMin

#     # --- Final inference ---
#     evm_likely = bool(sustained_disconnect or flat_top or cycling)

#     return {
#         "sustained_disconnect": sustained_disconnect,
#         "flat_top": flat_top,
#         "cycling": cycling,
#         "evm_likely": evm_likely,
#     }

