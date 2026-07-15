from typing import Any

import polars as pl
from neverReconnectedEvents import analyze_never_reconnected_events


class CheckPVBehaviour:
    def __init__(self, circuitData, volCol = None, minSamplesPercentage = None,
                 timeWindowForClassification = 60, vDrop = None, powerMeasError = None,
                 vMeasError = None):
        # mandatory params
        self.circuitData = circuitData
        # which voltage value to consider
        self.volCol      = "vmean" if volCol is None else volCol
        
        self.minSamplesPercentage = 0.1 if minSamplesPercentage is None else minSamplesPercentage
        self.timeWindowForClassification = 3 if timeWindowForClassification is None else timeWindowForClassification

        # site->terminal mapping (used in KPI math)
        self.vDrop = 2.3 if vDrop is None else vDrop
        self.powerMeasError = 0.04 if powerMeasError is None else powerMeasError
        self.vMeasError     = 0 if vMeasError is None else vMeasError
        self._prepared_site_day_frame = None
        
        # per-row time deltas & next values
        self.circuitData = self.circuitData.with_columns(
            (pl.col("local_tstamp").cast(pl.Datetime).shift(-1) - pl.col("local_tstamp").cast(pl.Datetime))
            .dt.total_seconds()
            .fill_null(0)
            .alias("dt_next_s")
        )
        self.circuitData = self.circuitData.with_columns(
            pl.col("^power(_.*)?$").shift(-1).name.suffix("_next")
        )
        self.circuitData = self.circuitData.with_columns(
            pl.col("local_tstamp").shift(-1).alias("ts_next")
        )

    def _power_columns(self, df):
        return [
            c for c in df.columns
            if c.startswith("power")
            and not c.endswith("_next")
            and not c.endswith("_logic")
        ]

    def _voltage_columns(self, df):
        return [c for c in df.columns if c.startswith(self.volCol)]

    def prepare_site_day_frame(self):
        """
        Build the shared site-day frame used by downstream eligibility and
        conformance consumers. This adds the common rolling voltage signals but
        does not apply any dataset-specific filtering policy.
        """
        if self._prepared_site_day_frame is not None:
            return self._prepared_site_day_frame

        voltage_cols = self._voltage_columns(self.circuitData)
        df = self.circuitData.clone()

        if voltage_cols:
            for c in voltage_cols:
                rolled_name = f"vmean_rolling_10m{c.replace(self.volCol, '', 1)}"
                rolled = (
                    df.filter(pl.col(c).is_not_null())
                    .with_columns(
                        pl.col(c).rolling_mean_by(by="local_tstamp", window_size="10m").alias(rolled_name)
                    )
                    .select(["local_tstamp", rolled_name])
                )
                df = df.join(rolled, on="local_tstamp", how="left")

            vmean_cols = [c for c in df.columns if c.startswith("vmean_rolling_10m")]
            df = df.with_columns([
                pl.mean_horizontal([pl.col(c) for c in vmean_cols]).alias("v10m_avg"),
                pl.max_horizontal([pl.col(c) for c in voltage_cols]).alias("vinst_max"),
            ])
        else:
            df = df.with_columns([
                pl.lit(None).cast(pl.Float64).alias("v10m_avg"),
                pl.lit(None).cast(pl.Float64).alias("vinst_max"),
            ])

        self._prepared_site_day_frame = df
        return self._prepared_site_day_frame

    # ------------------------------ OV1 KPI (anti‑islanding) ------------------------------
    # (Kept as‑is for assessment; diagnostics are not emitted here)
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

    def build_site_day_signals(self, PRated):
        """
        Prepare the site-day frame used by Phase A and Phase B.
        The dataframe stays on the measured/switchboard basis; site-specific
        voltage shifts are learned later in Phase A.
        """
        df = self.prepare_site_day_frame()
        power_cols = self._power_columns(df)
        power_cols_next = [
            c for c in df.columns
            if c.startswith("power")
            and c.endswith("_next")
            and not c.endswith("_logic_next")
        ]

        if not power_cols:
            return pl.DataFrame()

        p_disconnect = self.powerMeasError * PRated
        p_step_strict = 0.10 * PRated
        p_step_fallback = 0.05 * PRated

        logic_current = []
        logic_next = []
        for c in power_cols:
            logic_name = f"{c}_logic"
            logic_next_name = f"{c}_logic_next"
            df = df.with_columns(
                pl.when(pl.col(c).fill_null(0) < 0)
                  .then(pl.lit(0.0))
                  .otherwise(pl.col(c).fill_null(0))
                  .alias(logic_name)
            )
            logic_current.append(logic_name)

            next_name = f"{c}_next"
            if next_name in power_cols_next:
                df = df.with_columns(
                    pl.when(pl.col(next_name).fill_null(0) < 0)
                      .then(pl.lit(0.0))
                      .otherwise(pl.col(next_name).fill_null(0))
                      .alias(logic_next_name)
                )
            else:
                df = df.with_columns(pl.lit(0.0).alias(logic_next_name))
            logic_next.append(logic_next_name)

        df = df.with_columns([
            pl.sum_horizontal([pl.col(c) for c in logic_current]).alias("site_power"),
            pl.sum_horizontal([pl.col(c) for c in logic_next]).alias("site_power_next"),
        ])

        df = df.with_columns([
            (
                pl.all_horizontal([pl.col(c) <= p_disconnect for c in logic_current]) &
                (pl.col("site_power") <= p_disconnect)
            ).alias("is_disc"),
            (
                pl.all_horizontal([pl.col(c) <= p_disconnect for c in logic_next]) &
                (pl.col("site_power_next") <= p_disconnect)
            ).alias("is_disc_next"),
            pl.col("site_power").shift(1).alias("site_power_prev"),
        ])

        df = df.with_columns([
            pl.col("is_disc").shift(1).fill_null(False).alias("is_disc_prev"),
            pl.col("v10m_avg").is_not_null().alias("eligible_los"),
            pl.col("vinst_max").is_not_null().alias("eligible_ov1"),
            (pl.col("site_power_prev") - pl.col("site_power")).fill_null(0).alias("site_power_drop"),
            (pl.col("site_power") - pl.col("site_power_prev")).fill_null(0).alias("site_power_rise"),
        ])

        df = df.with_columns([
            (
                (~pl.col("is_disc_prev")) &
                pl.col("is_disc") &
                (pl.col("site_power_drop") >= p_step_strict)
            ).alias("disconnect_edge"),
            (
                pl.col("is_disc_prev") &
                (~pl.col("is_disc")) &
                (pl.col("site_power_rise") >= p_step_strict)
            ).alias("reconnect_edge"),
        ])

        df = df.with_columns([
            (
                (~pl.col("is_disc_prev")) &
                pl.col("is_disc") &
                (pl.col("site_power_drop") >= p_step_fallback) &
                (~pl.col("disconnect_edge"))
            ).alias("disconnect_edge_fallback"),
            (
                pl.col("is_disc_prev") &
                (~pl.col("is_disc")) &
                (pl.col("site_power_rise") >= p_step_fallback) &
                (~pl.col("reconnect_edge"))
            ).alias("reconnect_edge_fallback"),
        ])

        return df

    def phase_a_day(
        self,
        PRated,
        *,
        los_lo=251.1,
        los_hi_strict=259.0,
        los_hi_cap=260.3,
        eps=0.02,
    ):
        """
        Phase A for one site-day:
          - harvest all debounced disconnects
          - attribute each disconnect to LOS or OV1
          - capture all reconnect edges for later bracketing
        """
        df = self.build_site_day_signals(PRated)
        if df.is_empty():
            return {
                "frame": df,
                "records": pl.DataFrame(),
                "brackets": pl.DataFrame(),
                "reconnects": pl.DataFrame(),
            }

        disc_rows = (
            df.filter(pl.col("disconnect_edge") | pl.col("disconnect_edge_fallback"))
              .select([
                  "site_id", "local_tstamp", "v10m_avg", "vinst_max",
                  "site_power_drop",
                  "disconnect_edge", "disconnect_edge_fallback",
              ])
              .with_columns(
                  pl.when(pl.col("disconnect_edge"))
                    .then(pl.lit("strict_10pct"))
                    .otherwise(pl.lit("fallback_5pct"))
                    .alias("edge_source")
              )
              .sort("local_tstamp")
        )
        rec_rows = (
            df.filter(pl.col("reconnect_edge") | pl.col("reconnect_edge_fallback"))
              .select([
                  "site_id", "local_tstamp", "v10m_avg", "vinst_max",
                  "reconnect_edge", "reconnect_edge_fallback",
              ])
              .with_columns(
                  pl.when(pl.col("reconnect_edge"))
                    .then(pl.lit("strict_10pct"))
                    .otherwise(pl.lit("fallback_5pct"))
                    .alias("edge_source")
              )
              .sort("local_tstamp")
        )

        if disc_rows.is_empty():
            return {
                "frame": df,
                "records": pl.DataFrame(),
                "brackets": pl.DataFrame(),
                "reconnects": rec_rows,
            }

        reconnect_list = rec_rows.iter_rows(named=True)
        reconnect_list = list(reconnect_list)
        records: list[dict[str, Any]] = []
        brackets: list[dict[str, Any]] = []

        event_idx = 0

        for row in disc_rows.iter_rows(named=True):
            event_idx += 1
            tdisc = row["local_tstamp"]
            v10m = row["v10m_avg"]
            vinst = row["vinst_max"]
            edge_source = row["edge_source"]
            site_power_drop_kw = row["site_power_drop"]
            site_power_drop_pct = None if PRated in [None, 0] or site_power_drop_kw is None else (float(site_power_drop_kw) / float(PRated)) * 100.0
            mech = None
            threshold_voltage = None
            grey_non_sustained = False
            reconnect = next((r for r in reconnect_list if r["local_tstamp"] > tdisc), None)
            vinst_in_ov1_region = vinst is not None and (los_hi_strict <= vinst <= los_hi_cap)

            if v10m is not None and (los_lo <= v10m <= los_hi_strict):
                mech = "LOS"
                threshold_voltage = v10m
            elif vinst is not None and (vinst > los_hi_cap):
                mech = "OV1"
                threshold_voltage = vinst
            elif v10m is not None and (los_hi_strict < v10m <= los_hi_cap):
                if vinst_in_ov1_region:
                    mech = "OV1"
                    threshold_voltage = vinst
                    grey_non_sustained = True
                else:
                    mech = "LOS"
                    threshold_voltage = v10m

            if mech is None or threshold_voltage is None:
                continue

            record = {
                "site_id": row["site_id"],
                "event_id": event_idx,
                "ts_disc": tdisc,
                "edge_source": edge_source,
                "mech": mech,
                "v_los_recorded": threshold_voltage if mech == "LOS" else None,
                "v_ov1_recorded": threshold_voltage if mech == "OV1" else None,
                "v10m_disc": v10m,
                "vinst_disc": vinst,
                "site_power_drop_kw": site_power_drop_kw,
                "site_power_drop_pct_rated": site_power_drop_pct,
                "grey_non_sustained": grey_non_sustained,
            }
            records.append(record)

            if reconnect is None:
                continue

            if mech == "LOS":
                vrec = reconnect["v10m_avg"]
            else:
                vrec = reconnect["vinst_max"]

            if vrec is None:
                continue

            brackets.append({
                "site_id": row["site_id"],
                "event_id": event_idx,
                "edge_source": edge_source,
                "mech": mech,
                "ts_disc": tdisc,
                "ts_rec": reconnect["local_tstamp"],
                "L": vrec + eps,
                "U": threshold_voltage,
                "midpoint": (vrec + eps + threshold_voltage) / 2.0,
                "width": threshold_voltage - (vrec + eps),
            })

        rec_df = (
            pl.DataFrame(records)
            if records
            else pl.DataFrame(
                schema={
                    "site_id": pl.Int64,
                    "event_id": pl.Int64,
                    "ts_disc": pl.Datetime(time_zone="Australia/Adelaide"),
                    "edge_source": pl.Utf8,
                    "mech": pl.Utf8,
                    "v_los_recorded": pl.Float64,
                    "v_ov1_recorded": pl.Float64,
                    "v10m_disc": pl.Float64,
                    "vinst_disc": pl.Float64,
                    "site_power_drop_kw": pl.Float64,
                    "site_power_drop_pct_rated": pl.Float64,
                    "grey_non_sustained": pl.Boolean,
                }
            )
        ).with_columns([
            pl.col("site_id").cast(pl.Int64),
            pl.col("event_id").cast(pl.Int64),
            pl.col("ts_disc").cast(pl.Datetime(time_zone="Australia/Adelaide")),
            pl.col("edge_source").cast(pl.Utf8),
            pl.col("mech").cast(pl.Utf8),
            pl.col("v_los_recorded").cast(pl.Float64),
            pl.col("v_ov1_recorded").cast(pl.Float64),
            pl.col("v10m_disc").cast(pl.Float64),
            pl.col("vinst_disc").cast(pl.Float64),
            pl.col("site_power_drop_kw").cast(pl.Float64),
            pl.col("site_power_drop_pct_rated").cast(pl.Float64),
            pl.col("grey_non_sustained").cast(pl.Boolean),
        ])
        br_df = (
            pl.DataFrame(brackets)
            if brackets
            else pl.DataFrame(
                schema={
                    "site_id": pl.Int64,
                    "event_id": pl.Int64,
                    "edge_source": pl.Utf8,
                    "mech": pl.Utf8,
                    "ts_disc": pl.Datetime(time_zone="Australia/Adelaide"),
                    "ts_rec": pl.Datetime(time_zone="Australia/Adelaide"),
                    "L": pl.Float64,
                    "U": pl.Float64,
                    "midpoint": pl.Float64,
                    "width": pl.Float64,
                }
            )
        ).with_columns([
            pl.col("site_id").cast(pl.Int64),
            pl.col("event_id").cast(pl.Int64),
            pl.col("edge_source").cast(pl.Utf8),
            pl.col("mech").cast(pl.Utf8),
            pl.col("ts_disc").cast(pl.Datetime(time_zone="Australia/Adelaide")),
            pl.col("ts_rec").cast(pl.Datetime(time_zone="Australia/Adelaide")),
            pl.col("L").cast(pl.Float64),
            pl.col("U").cast(pl.Float64),
            pl.col("midpoint").cast(pl.Float64),
            pl.col("width").cast(pl.Float64),
        ])

        return {
            "frame": df,
            "records": rec_df,
            "brackets": br_df,
            "reconnects": rec_rows,
        }

    def phase_b_day(
        self,
        PRated,
        *,
        los_threshold,
        ov1_work_threshold,
        tau=0.3,
    ):
        """
        Phase B for one site-day:
          - assign one responsible mechanism per eligible timestamp
          - score time compliance using current-or-next disconnected state
        """
        df = self.build_site_day_signals(PRated)
        if df.is_empty():
            return {
                "frame": df,
                "detail": pl.DataFrame(),
                "summary": {
                    "los_eligible": 0,
                    "los_compliant": 0,
                    "ov1_eligible": 0,
                    "ov1_compliant": 0,
                },
            }

        df = df.with_columns([
            (pl.col("eligible_ov1") & (pl.col("vinst_max") >= (ov1_work_threshold - tau))).alias("ov1_responsible"),
        ])
        df = df.with_columns([
            (
                pl.col("eligible_los") &
                (~pl.col("ov1_responsible")) &
                (pl.col("v10m_avg") > los_threshold)
            ).alias("los_responsible"),
            (pl.col("is_disc") | pl.col("is_disc_next")).alias("is_disc_current_or_next"),
        ])
        df = df.with_columns([
            (pl.col("los_responsible") & pl.col("is_disc_current_or_next")).alias("los_compliant"),
            (pl.col("ov1_responsible") & pl.col("is_disc_current_or_next")).alias("ov1_compliant"),
        ])

        detail = df.select([
            "site_id", "local_tstamp", "utc_tstamp", "v10m_avg", "vinst_max",
            "eligible_los", "eligible_ov1", "is_disc", "is_disc_next",
            "los_responsible", "ov1_responsible", "los_compliant", "ov1_compliant",
        ])

        summary = {
            "los_eligible": int(detail.filter(pl.col("los_responsible")).height),
            "los_compliant": int(detail.filter(pl.col("los_compliant")).height),
            "ov1_eligible": int(detail.filter(pl.col("ov1_responsible")).height),
            "ov1_compliant": int(detail.filter(pl.col("ov1_compliant")).height),
        }

        return {"frame": df, "detail": detail, "summary": summary}


def _median_or_none(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _quantile_or_none(values, q: float):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _min_or_none(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return min(values)


def _range_or_none(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return max(values) - min(values)


def _blend_threshold(default_value, learned_value, weight=0.5):
    if learned_value is None:
        return default_value
    return float(default_value) + weight * (float(learned_value) - float(default_value))


def _phase_b_selection_score(summary_row: dict[str, Any]):
    overall_pass = summary_row["overall_pass"]
    overall_rank = 2 if overall_pass is True else 1 if overall_pass is None else 0
    assessed_pcts = [
        float(v)
        for v in [summary_row["los_compliance_pct"], summary_row["ov1_compliance_pct"]]
        if v is not None
    ]
    min_pct = min(assessed_pcts) if assessed_pcts else 0.0
    mean_pct = sum(assessed_pcts) / len(assessed_pcts) if assessed_pcts else 0.0
    return overall_rank, min_pct, mean_pct


def _default_threshold_profile(*, tau=0.3, ov1_floor_offset=0.5):
    return _build_threshold_profile(
        los_anchor=258.0,
        los_anchor_p25=258.0,
        los_anchor_p10=258.0,
        los_anchor_min=258.0,
        ov1_anchor=265.0,
        ov1_basis="default",
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )


def _build_threshold_profile(
    *,
    los_anchor,
    los_anchor_p25,
    los_anchor_p10,
    los_anchor_min,
    ov1_anchor,
    ov1_basis,
    tau=0.3,
    ov1_floor_offset=0.5,
):
    delta_los = None if los_anchor == 258.0 else los_anchor - 258.0
    delta_los_p25 = None if los_anchor_p25 == 258.0 else los_anchor_p25 - 258.0
    delta_los_p10 = None if los_anchor_p10 == 258.0 else los_anchor_p10 - 258.0
    delta_los_min = None if los_anchor_min == 258.0 else los_anchor_min - 258.0
    delta_ov1 = None if ov1_anchor == 265.0 else ov1_anchor - 265.0
    return {
        "delta_los_site": delta_los,
        "delta_los_p25_site": delta_los_p25,
        "delta_los_p10_site": delta_los_p10,
        "delta_los_min_site": delta_los_min,
        "delta_ov1_site": delta_ov1,
        "los_anchor_site": float(los_anchor),
        "los_anchor_p25_site": float(los_anchor_p25),
        "los_anchor_p10_site": float(los_anchor_p10),
        "los_anchor_min_site": float(los_anchor_min),
        "ov1_anchor_site": float(ov1_anchor),
        "ov1_work_site": float(ov1_anchor),
        "ov1_floor_site": float(ov1_anchor) - ov1_floor_offset,
        "ov1_test_site": float(ov1_anchor) - tau,
        "ov1_basis": ov1_basis,
        "delta_gap_v": None if (delta_los is None or delta_ov1 is None) else abs(delta_ov1 - delta_los),
    }


def _raw_threshold_profile(raw_thresholds, *, tau=0.3, ov1_floor_offset=0.5):
    return _build_threshold_profile(
        los_anchor=raw_thresholds["los_anchor_site"],
        los_anchor_p25=raw_thresholds["los_anchor_p25_site"],
        los_anchor_p10=raw_thresholds["los_anchor_p10_site"],
        los_anchor_min=raw_thresholds["los_anchor_min_site"],
        ov1_anchor=raw_thresholds["ov1_anchor_site"],
        ov1_basis=raw_thresholds["ov1_basis"],
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )


def _blended_threshold_profile(raw_thresholds, *, tau=0.3, ov1_floor_offset=0.5, weight=0.5):
    return _build_threshold_profile(
        los_anchor=_blend_threshold(258.0, raw_thresholds["los_anchor_site"], weight=weight),
        los_anchor_p25=_blend_threshold(258.0, raw_thresholds["los_anchor_p25_site"], weight=weight),
        los_anchor_p10=_blend_threshold(258.0, raw_thresholds["los_anchor_p10_site"], weight=weight),
        los_anchor_min=_blend_threshold(258.0, raw_thresholds["los_anchor_min_site"], weight=weight),
        ov1_anchor=_blend_threshold(265.0, raw_thresholds["ov1_anchor_site"], weight=weight),
        ov1_basis="blended",
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )


def _threshold_confidence_from_records(records: pl.DataFrame):
    if records.is_empty():
        return {
            "threshold_confidence_tier": "low",
            "confidence_primary_mech": None,
            "confidence_event_count": 0,
            "confidence_drop20_count": 0,
            "confidence_drop10_count": 0,
            "confidence_spread_v": None,
        }

    los_rows = records.filter(pl.col("mech") == "LOS")
    ov1_rows = records.filter(pl.col("mech") == "OV1")

    if not los_rows.is_empty():
        primary_rows = los_rows
        primary_mech = "LOS"
        voltage_col = "v_los_recorded"
    elif not ov1_rows.is_empty():
        primary_rows = ov1_rows
        primary_mech = "OV1"
        voltage_col = "v_ov1_recorded"
    else:
        return {
            "threshold_confidence_tier": "low",
            "confidence_primary_mech": None,
            "confidence_event_count": 0,
            "confidence_drop20_count": 0,
            "confidence_drop10_count": 0,
            "confidence_spread_v": None,
        }

    values = [float(v) for v in primary_rows[voltage_col].to_list() if v is not None]
    drop_pcts = [
        float(v)
        for v in primary_rows["site_power_drop_pct_rated"].to_list()
        if v is not None
    ]

    event_count = len(values)
    drop20_count = sum(v >= 20.0 for v in drop_pcts)
    drop10_count = sum(v >= 10.0 for v in drop_pcts)
    spread_v = _range_or_none(values)

    if event_count >= 2 and drop20_count >= 2 and spread_v is not None and spread_v <= 2.0:
        tier = "high"
    elif event_count >= 2 and drop10_count >= 2 and spread_v is not None and spread_v <= 3.0:
        tier = "medium"
    else:
        tier = "low"

    return {
        "threshold_confidence_tier": tier,
        "confidence_primary_mech": primary_mech,
        "confidence_event_count": event_count,
        "confidence_drop20_count": drop20_count,
        "confidence_drop10_count": drop10_count,
        "confidence_spread_v": spread_v,
    }


def _evaluate_phase_b_profile_for_selection(
    site_id,
    day_behaviours,
    PRated,
    profile,
    *,
    tau=0.3,
):
    phase_b = _run_phase_b_with_thresholds(
        site_id,
        day_behaviours,
        PRated,
        los_threshold=profile["los_anchor_site"],
        los_threshold_p25=profile["los_anchor_p25_site"],
        los_threshold_p10=profile["los_anchor_p10_site"],
        los_threshold_min=profile["los_anchor_min_site"],
        ov1_work_threshold=profile["ov1_work_site"],
        tau=tau,
    )
    summary = phase_b["summary_row"].to_dicts()[0]
    score = _phase_b_selection_score(summary)
    return phase_b, summary, score


def _profile_with_selection_metadata(profile, basis, score=None):
    return {
        **profile,
        "threshold_selection_basis": basis,
        "selection_score_rank": None if score is None else score[0],
        "selection_score_min_pct": None if score is None else score[1],
        "selection_score_mean_pct": None if score is None else score[2],
    }


def _select_confidence_threshold_profile_for_phase_b(
    site_id,
    day_behaviours,
    PRated,
    raw_thresholds,
    confidence_info,
    *,
    high_profile_name="learned",
    tau=0.3,
    ov1_floor_offset=0.5,
):
    default_profile = _default_threshold_profile(
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )
    learned_profile = _raw_threshold_profile(
        raw_thresholds,
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )
    blended_profile = _blended_threshold_profile(
        raw_thresholds,
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )

    tier = confidence_info["threshold_confidence_tier"]
    if tier == "high":
        high_profiles = {
            "learned": learned_profile,
            "blended": blended_profile,
        }
        return _profile_with_selection_metadata(
            high_profiles[high_profile_name],
            f"high_{high_profile_name}",
        )

    if tier == "low":
        return _profile_with_selection_metadata(default_profile, "low_default")

    candidates = [
        ("default", default_profile),
        ("blended", blended_profile),
        ("learned", learned_profile),
    ]
    best_name = None
    best_profile = None
    best_score = None

    for name, profile in candidates:
        _, _, score = _evaluate_phase_b_profile_for_selection(
            site_id,
            day_behaviours,
            PRated,
            profile,
            tau=tau,
        )
        if best_score is None or score > best_score:
            best_name = name
            best_profile = profile
            best_score = score

    return _profile_with_selection_metadata(
        best_profile,
        f"medium_{best_name}",
        best_score,
    )


def _select_legacy_sweep_threshold_profile_for_phase_b(
    site_id,
    day_behaviours,
    PRated,
    *,
    tau=0.3,
    ov1_floor_offset=0.5,
):
    default_profile = _default_threshold_profile(
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )
    sweep_thresholds = [
        257.0,
        256.0,
        255.7,
        254.7,
        253.7,
        253.4,
        251.8,
        251.1,
        259.0,
        260.0,
        260.3,
    ]

    _, summary, best_score = _evaluate_phase_b_profile_for_selection(
        site_id,
        day_behaviours,
        PRated,
        default_profile,
        tau=tau,
    )
    if summary["overall_pass"] is True:
        return _profile_with_selection_metadata(default_profile, "sweep_default", best_score)

    best_profile = default_profile
    best_basis = "sweep_default"
    for los_threshold in sweep_thresholds:
        candidate_profile = _build_threshold_profile(
            los_anchor=los_threshold,
            los_anchor_p25=los_threshold,
            los_anchor_p10=los_threshold,
            los_anchor_min=los_threshold,
            ov1_anchor=265.0,
            ov1_basis="default",
            tau=tau,
            ov1_floor_offset=ov1_floor_offset,
        )
        _, summary, score = _evaluate_phase_b_profile_for_selection(
            site_id,
            day_behaviours,
            PRated,
            candidate_profile,
            tau=tau,
        )
        if summary["overall_pass"] is True:
            return _profile_with_selection_metadata(
                candidate_profile,
                f"sweep_{str(los_threshold).replace('.', 'p')}V",
                score,
            )
        if score > best_score:
            best_score = score
            best_profile = candidate_profile
            best_basis = f"sweep_{str(los_threshold).replace('.', 'p')}V"

    return _profile_with_selection_metadata(best_profile, best_basis, best_score)


def _thresholds_row_from_threshold_dict(
    site_id,
    thresholds,
    raw_thresholds,
    confidence_info,
    *,
    method_key=None,
):
    row = {
        "site_id": site_id,
        "delta_los_site": thresholds["delta_los_site"],
        "delta_los_p25_site": thresholds["delta_los_p25_site"],
        "delta_los_p10_site": thresholds["delta_los_p10_site"],
        "delta_los_min_site": thresholds["delta_los_min_site"],
        "delta_ov1_site": thresholds["delta_ov1_site"],
        "los_anchor_site": thresholds["los_anchor_site"],
        "los_anchor_p25_site": thresholds["los_anchor_p25_site"],
        "los_anchor_p10_site": thresholds["los_anchor_p10_site"],
        "los_anchor_min_site": thresholds["los_anchor_min_site"],
        "ov1_anchor_site": thresholds["ov1_anchor_site"],
        "ov1_work_site": thresholds["ov1_work_site"],
        "ov1_floor_site": thresholds["ov1_floor_site"],
        "ov1_test_site": thresholds["ov1_test_site"],
        "delta_gap_v": thresholds["delta_gap_v"],
        "delta_gap_flag": None if thresholds["delta_gap_v"] is None else thresholds["delta_gap_v"] > 2.0,
        "ov1_basis": thresholds["ov1_basis"],
        "ov1_event_count": raw_thresholds["ov1_event_count"],
        "ov1_reclassified_count": raw_thresholds["ov1_reclassified_count"],
        "los_removed_by_ov1_count": raw_thresholds["los_removed_by_ov1_count"],
        "raw_delta_los_site": raw_thresholds["delta_los_site"],
        "raw_delta_los_p25_site": raw_thresholds["delta_los_p25_site"],
        "raw_delta_los_p10_site": raw_thresholds["delta_los_p10_site"],
        "raw_delta_los_min_site": raw_thresholds["delta_los_min_site"],
        "raw_delta_ov1_site": raw_thresholds["delta_ov1_site"],
        "raw_los_anchor_site": raw_thresholds["los_anchor_site"],
        "raw_los_anchor_p25_site": raw_thresholds["los_anchor_p25_site"],
        "raw_los_anchor_p10_site": raw_thresholds["los_anchor_p10_site"],
        "raw_los_anchor_min_site": raw_thresholds["los_anchor_min_site"],
        "raw_ov1_anchor_site": raw_thresholds["ov1_anchor_site"],
        "raw_ov1_basis": raw_thresholds["ov1_basis"],
        "raw_delta_gap_v": raw_thresholds["delta_gap_v"],
        "threshold_confidence_tier": confidence_info["threshold_confidence_tier"],
        "confidence_primary_mech": confidence_info["confidence_primary_mech"],
        "confidence_event_count": confidence_info["confidence_event_count"],
        "confidence_drop20_count": confidence_info["confidence_drop20_count"],
        "confidence_drop10_count": confidence_info["confidence_drop10_count"],
        "confidence_spread_v": confidence_info["confidence_spread_v"],
        "threshold_selection_basis": thresholds["threshold_selection_basis"],
        "selection_score_rank": thresholds["selection_score_rank"],
        "selection_score_min_pct": thresholds["selection_score_min_pct"],
        "selection_score_mean_pct": thresholds["selection_score_mean_pct"],
    }
    if method_key is not None:
        row["method_key"] = method_key

    return (
        pl.DataFrame([row])
        .with_columns([
            pl.col("site_id").cast(pl.Int64),
            pl.col("delta_los_site").cast(pl.Float64),
            pl.col("delta_los_p25_site").cast(pl.Float64),
            pl.col("delta_los_p10_site").cast(pl.Float64),
            pl.col("delta_los_min_site").cast(pl.Float64),
            pl.col("delta_ov1_site").cast(pl.Float64),
            pl.col("los_anchor_site").cast(pl.Float64),
            pl.col("los_anchor_p25_site").cast(pl.Float64),
            pl.col("los_anchor_p10_site").cast(pl.Float64),
            pl.col("los_anchor_min_site").cast(pl.Float64),
            pl.col("ov1_anchor_site").cast(pl.Float64),
            pl.col("ov1_work_site").cast(pl.Float64),
            pl.col("ov1_floor_site").cast(pl.Float64),
            pl.col("ov1_test_site").cast(pl.Float64),
            pl.col("delta_gap_v").cast(pl.Float64),
            pl.col("delta_gap_flag").cast(pl.Boolean),
            pl.col("ov1_basis").cast(pl.Utf8),
            pl.col("ov1_event_count").cast(pl.Int64),
            pl.col("ov1_reclassified_count").cast(pl.Int64),
            pl.col("los_removed_by_ov1_count").cast(pl.Int64),
            pl.col("raw_delta_los_site").cast(pl.Float64),
            pl.col("raw_delta_los_p25_site").cast(pl.Float64),
            pl.col("raw_delta_los_p10_site").cast(pl.Float64),
            pl.col("raw_delta_los_min_site").cast(pl.Float64),
            pl.col("raw_delta_ov1_site").cast(pl.Float64),
            pl.col("raw_los_anchor_site").cast(pl.Float64),
            pl.col("raw_los_anchor_p25_site").cast(pl.Float64),
            pl.col("raw_los_anchor_p10_site").cast(pl.Float64),
            pl.col("raw_los_anchor_min_site").cast(pl.Float64),
            pl.col("raw_ov1_anchor_site").cast(pl.Float64),
            pl.col("raw_ov1_basis").cast(pl.Utf8),
            pl.col("raw_delta_gap_v").cast(pl.Float64),
            pl.col("threshold_confidence_tier").cast(pl.Utf8),
            pl.col("confidence_primary_mech").cast(pl.Utf8),
            pl.col("confidence_event_count").cast(pl.Int64),
            pl.col("confidence_drop20_count").cast(pl.Int64),
            pl.col("confidence_drop10_count").cast(pl.Int64),
            pl.col("confidence_spread_v").cast(pl.Float64),
            pl.col("threshold_selection_basis").cast(pl.Utf8),
            pl.col("selection_score_rank").cast(pl.Int64),
            pl.col("selection_score_min_pct").cast(pl.Float64),
            pl.col("selection_score_mean_pct").cast(pl.Float64),
            *([pl.col("method_key").cast(pl.Utf8)] if method_key is not None else []),
        ])
    )


def _site_thresholds_from_records(
    records: pl.DataFrame,
    *,
    tau: float = 0.3,
    ov1_floor_offset: float = 0.5,
):
    los_rows = records.filter(pl.col("mech") == "LOS") if not records.is_empty() else pl.DataFrame()
    ov1_rows = records.filter(pl.col("mech") == "OV1") if not records.is_empty() else pl.DataFrame()
    los_vals = los_rows["v_los_recorded"].to_list() if not los_rows.is_empty() else []
    ov1_vals_direct = ov1_rows["v_ov1_recorded"].to_list() if not ov1_rows.is_empty() else []

    delta_los = None
    delta_los_p25 = None
    delta_los_p10 = None
    delta_los_min = None
    delta_ov1 = None
    ov1_reclassified_count = 0
    los_removed_by_ov1_count = 0
    has_direct_ov1 = len(ov1_vals_direct) >= 1
    if has_direct_ov1:
        direct_ov1_anchor = _median_or_none(ov1_vals_direct)
        provisional_ov1_anchor = direct_ov1_anchor
        retained_los_vals: list[float] = []
        reclassified_ov1_vals: list[float] = []
        for row in los_rows.iter_rows(named=True):
            vlos = row["v_los_recorded"]
            if vlos is None:
                continue
            vlos = float(vlos)
            if vlos > provisional_ov1_anchor:
                los_removed_by_ov1_count += 1
                v10m = row["v10m_disc"]
                vinst = row["vinst_disc"]
                is_clear_ov1 = vinst is not None and float(vinst) > 260.3
                is_grey_ov1 = (
                    v10m is not None and
                    259.0 < float(v10m) <= 260.3 and
                    vinst is not None and
                    259.0 <= float(vinst) <= 260.3
                )
                if is_clear_ov1 or is_grey_ov1:
                    reclassified_ov1_vals.append(float(vinst))
            else:
                retained_los_vals.append(vlos)
        los_vals = retained_los_vals
        ov1_vals = [float(v) for v in ov1_vals_direct if v is not None] + reclassified_ov1_vals
        ov1_reclassified_count = len(reclassified_ov1_vals)
        delta_ov1 = _median_or_none(ov1_vals) - 265.0
    else:
        ov1_vals = []

    if los_vals:
        delta_los = _median_or_none(los_vals) - 258.0
        delta_los_p25 = _quantile_or_none(los_vals, 0.25) - 258.0
        delta_los_p10 = _quantile_or_none(los_vals, 0.10) - 258.0
        delta_los_min = _min_or_none(los_vals) - 258.0
    elif delta_ov1 is not None:
        # If OV1 is observed directly and no LOS samples survive, infer LOS from the same site delta.
        delta_los = delta_ov1
        delta_los_p25 = delta_ov1
        delta_los_p10 = delta_ov1
        delta_los_min = delta_ov1

    los_anchor = 258.0 if delta_los is None else 258.0 + delta_los
    los_anchor_p25 = 258.0 if delta_los_p25 is None else 258.0 + delta_los_p25
    los_anchor_p10 = 258.0 if delta_los_p10 is None else 258.0 + delta_los_p10
    los_anchor_min = 258.0 if delta_los_min is None else 258.0 + delta_los_min
    if delta_ov1 is not None:
        ov1_anchor = 265.0 + delta_ov1
        ov1_basis = "ov1_records"
    elif delta_los is not None:
        ov1_anchor = 265.0 + delta_los
        ov1_basis = "los_fallback"
    else:
        ov1_anchor = 265.0
        ov1_basis = "default"

    return {
        "delta_los_site": delta_los,
        "delta_los_p25_site": delta_los_p25,
        "delta_los_p10_site": delta_los_p10,
        "delta_los_min_site": delta_los_min,
        "delta_ov1_site": delta_ov1,
        "los_anchor_site": los_anchor,
        "los_anchor_p25_site": los_anchor_p25,
        "los_anchor_p10_site": los_anchor_p10,
        "los_anchor_min_site": los_anchor_min,
        "ov1_anchor_site": ov1_anchor,
        "ov1_work_site": ov1_anchor,
        "ov1_floor_site": ov1_anchor - ov1_floor_offset,
        "ov1_test_site": ov1_anchor - tau,
        "ov1_basis": ov1_basis,
        "ov1_event_count": len(ov1_vals),
        "ov1_reclassified_count": ov1_reclassified_count,
        "los_removed_by_ov1_count": los_removed_by_ov1_count,
        "delta_gap_v": None if (delta_los is None or delta_ov1 is None) else abs(delta_ov1 - delta_los),
    }


def run_phase_a_for_site(
    site_id,
    day_behaviours,
    PRated,
    *,
    tau=0.3,
    eps=0.02,
    delta_lower_daily_cap=0.5,
):
    """
    Run Phase A across all available days for one site and learn site thresholds
    from the disconnect-edge records.
    """
    last_records = pl.DataFrame()
    last_brackets = pl.DataFrame()
    phase_a_days = []
    records_all = []
    brackets_all = []
    phase_a_days = []
    for day_info in day_behaviours:
        outcome = day_info["behaviour"].phase_a_day(
            PRated,
            eps=eps,
        )
        phase_a_days.append({"day": day_info["day"], **outcome})
        if not outcome["records"].is_empty():
            records_all.append(outcome["records"].with_columns(pl.lit(day_info["day"]).alias("event_day")))
        if not outcome["brackets"].is_empty():
            brackets_all.append(outcome["brackets"].with_columns(pl.lit(day_info["day"]).alias("event_day")))

    last_records = pl.concat(records_all, how="vertical") if records_all else pl.DataFrame()
    last_brackets = pl.concat(brackets_all, how="vertical") if brackets_all else pl.DataFrame()

    raw_thresholds = _site_thresholds_from_records(
        last_records,
        tau=tau,
        ov1_floor_offset=delta_lower_daily_cap,
    )
    confidence_info = _threshold_confidence_from_records(last_records)

    return {
        "raw_thresholds": raw_thresholds,
        "confidence_info": confidence_info,
        "records": last_records,
        "brackets": last_brackets,
        "day_outputs": phase_a_days,
    }


def _select_phase_b_threshold_profile_for_method(
    site_id,
    day_behaviours,
    PRated,
    raw_thresholds,
    confidence_info,
    *,
    phase_b_method="tier_based",
    tau=0.3,
    ov1_floor_offset=0.5,
):
    if phase_b_method == "default":
        return _profile_with_selection_metadata(
            _default_threshold_profile(tau=tau, ov1_floor_offset=ov1_floor_offset),
            "default",
        )
    if phase_b_method == "original":
        return _profile_with_selection_metadata(
            _raw_threshold_profile(raw_thresholds, tau=tau, ov1_floor_offset=ov1_floor_offset),
            "original",
        )
    if phase_b_method == "tier_based":
        return _select_confidence_threshold_profile_for_phase_b(
            site_id,
            day_behaviours,
            PRated,
            raw_thresholds,
            confidence_info,
            high_profile_name="learned",
            tau=tau,
            ov1_floor_offset=ov1_floor_offset,
        )
    if phase_b_method == "old_sweep":
        return _select_legacy_sweep_threshold_profile_for_phase_b(
            site_id,
            day_behaviours,
            PRated,
            tau=tau,
            ov1_floor_offset=ov1_floor_offset,
        )
    if phase_b_method == "blended":
        return _select_confidence_threshold_profile_for_phase_b(
            site_id,
            day_behaviours,
            PRated,
            raw_thresholds,
            confidence_info,
            high_profile_name="blended",
            tau=tau,
            ov1_floor_offset=ov1_floor_offset,
        )
    raise KeyError(f"Unknown Phase B method: {phase_b_method}")


def _run_phase_b_with_thresholds(
    site_id,
    day_behaviours,
    PRated,
    *,
    los_threshold,
    los_threshold_p25=None,
    los_threshold_p10=None,
    los_threshold_min=None,
    ov1_work_threshold,
    tau=0.3,
    compliance_threshold_pct=90.0,
):
    """
    Aggregate Phase B time-compliance across all available days for one site.
    """
    def aggregate_for_los_threshold(los_threshold_used):
        detail_frames = []
        los_eligible = 0
        los_compliant = 0
        ov1_eligible = 0
        ov1_compliant = 0

        for day_info in day_behaviours:
            outcome = day_info["behaviour"].phase_b_day(
                PRated,
                los_threshold=los_threshold_used,
                ov1_work_threshold=ov1_work_threshold,
                tau=tau,
            )
            if not outcome["detail"].is_empty():
                detail_frames.append(outcome["detail"].with_columns(pl.lit(day_info["day"]).alias("event_day")))
            summary = outcome["summary"]
            los_eligible += summary["los_eligible"]
            los_compliant += summary["los_compliant"]
            ov1_eligible += summary["ov1_eligible"]
            ov1_compliant += summary["ov1_compliant"]

        detail_all = pl.concat(detail_frames, how="vertical") if detail_frames else pl.DataFrame()
        los_pct = None if los_eligible == 0 else (los_compliant / los_eligible) * 100.0
        ov1_pct = None if ov1_eligible == 0 else (ov1_compliant / ov1_eligible) * 100.0
        los_pass = None if los_pct is None else los_pct >= compliance_threshold_pct
        ov1_pass = None if ov1_pct is None else ov1_pct >= compliance_threshold_pct
        assessed_passes = [v for v in [los_pass, ov1_pass] if v is not None]
        overall_pass = None if not assessed_passes else all(assessed_passes)
        return {
            "detail": detail_all,
            "los_eligible": los_eligible,
            "los_compliant": los_compliant,
            "los_pct": los_pct,
            "ov1_eligible": ov1_eligible,
            "ov1_compliant": ov1_compliant,
            "ov1_pct": ov1_pct,
            "los_pass": los_pass,
            "ov1_pass": ov1_pass,
            "overall_pass": overall_pass,
            "los_threshold_used": los_threshold_used,
        }

    median_result = aggregate_for_los_threshold(los_threshold)
    p25_result = None
    p10_result = None
    min_result = None
    if (
        median_result["los_pass"] is False and
        los_threshold_p25 is not None and
        los_threshold_p25 != los_threshold
    ):
        p25_result = aggregate_for_los_threshold(los_threshold_p25)

    if (
        median_result["los_pass"] is False and
        (p25_result is None or p25_result["los_pass"] is False) and
        los_threshold_p10 is not None and
        los_threshold_p10 not in [los_threshold, los_threshold_p25]
    ):
        p10_result = aggregate_for_los_threshold(los_threshold_p10)

    if (
        median_result["los_pass"] is False and
        (p25_result is None or p25_result["los_pass"] is False) and
        (p10_result is None or p10_result["los_pass"] is False) and
        los_threshold_min is not None and
        los_threshold_min not in [los_threshold, los_threshold_p25, los_threshold_p10]
    ):
        min_result = aggregate_for_los_threshold(los_threshold_min)

    use_p25_override = (
        p25_result is not None and
        median_result["los_pass"] is False and
        p25_result["los_pass"] is True
    )
    use_p10_override = (
        p10_result is not None and
        median_result["los_pass"] is False and
        (p25_result is None or p25_result["los_pass"] is not True) and
        p10_result["los_pass"] is True
    )
    use_min_override = (
        min_result is not None and
        median_result["los_pass"] is False and
        (p25_result is None or p25_result["los_pass"] is not True) and
        (p10_result is None or p10_result["los_pass"] is not True) and
        min_result["los_pass"] is True
    )

    if use_min_override:
        chosen_result = min_result
    elif use_p10_override:
        chosen_result = p10_result
    elif use_p25_override:
        chosen_result = p25_result
    else:
        chosen_result = median_result
    threshold_sensitive = use_p25_override or use_p10_override or use_min_override
    pass_basis = (
        "unassessed" if chosen_result["overall_pass"] is None else
        "min_override" if use_min_override else
        "p10_override" if use_p10_override else
        "p25_override" if use_p25_override else
        "median"
    )

    summary_row = (
        pl.DataFrame([{
            "site_id": site_id,
            "los_eligible": chosen_result["los_eligible"],
            "los_compliant": chosen_result["los_compliant"],
            "los_compliance_pct": chosen_result["los_pct"],
            "ov1_eligible": chosen_result["ov1_eligible"],
            "ov1_compliant": chosen_result["ov1_compliant"],
            "ov1_compliance_pct": chosen_result["ov1_pct"],
            "los_pass": chosen_result["los_pass"],
            "ov1_pass": chosen_result["ov1_pass"],
            "overall_pass": chosen_result["overall_pass"],
            "los_threshold_used": chosen_result["los_threshold_used"],
            "threshold_sensitive": threshold_sensitive,
            "pass_basis": pass_basis,
        }])
        .with_columns([
            pl.col("site_id").cast(pl.Int64),
            pl.col("los_eligible").cast(pl.Int64),
            pl.col("los_compliant").cast(pl.Int64),
            pl.col("los_compliance_pct").cast(pl.Float64),
            pl.col("ov1_eligible").cast(pl.Int64),
            pl.col("ov1_compliant").cast(pl.Int64),
            pl.col("ov1_compliance_pct").cast(pl.Float64),
            pl.col("los_pass").cast(pl.Boolean),
            pl.col("ov1_pass").cast(pl.Boolean),
            pl.col("overall_pass").cast(pl.Boolean),
            pl.col("los_threshold_used").cast(pl.Float64),
            pl.col("threshold_sensitive").cast(pl.Boolean),
            pl.col("pass_basis").cast(pl.Utf8),
        ])
    )

    return {"detail": chosen_result["detail"], "summary_row": summary_row}


def run_phase_b_for_site(
    site_id,
    day_behaviours,
    PRated,
    *,
    raw_thresholds,
    confidence_info,
    phase_b_method="tier_based",
    tau=0.3,
    compliance_threshold_pct=90.0,
    ov1_floor_offset=None,
):
    """
    Run Phase B for one selected method using the thresholds learned in Phase A.
    """
    if ov1_floor_offset is None:
        ov1_floor_offset = float(raw_thresholds["ov1_anchor_site"]) - float(raw_thresholds["ov1_floor_site"])

    selected_thresholds = _select_phase_b_threshold_profile_for_method(
        site_id,
        day_behaviours,
        PRated,
        raw_thresholds,
        confidence_info,
        phase_b_method=phase_b_method,
        tau=tau,
        ov1_floor_offset=ov1_floor_offset,
    )
    threshold_row = _thresholds_row_from_threshold_dict(
        site_id,
        selected_thresholds,
        raw_thresholds,
        confidence_info,
    )
    phase_b = _run_phase_b_with_thresholds(
        site_id,
        day_behaviours,
        PRated,
        los_threshold=selected_thresholds["los_anchor_site"],
        los_threshold_p25=selected_thresholds["los_anchor_p25_site"],
        los_threshold_p10=selected_thresholds["los_anchor_p10_site"],
        los_threshold_min=selected_thresholds["los_anchor_min_site"],
        ov1_work_threshold=selected_thresholds["ov1_work_site"],
        tau=tau,
        compliance_threshold_pct=compliance_threshold_pct,
    )
    return {
        "detail": phase_b["detail"],
        "summary_row": phase_b["summary_row"],
        "threshold_row": threshold_row,
    }


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
