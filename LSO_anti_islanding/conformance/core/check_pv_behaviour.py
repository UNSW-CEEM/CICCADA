"""Prepared-site-day conformance behaviour assessment."""

from typing import Any

import polars as pl


class CheckPVBehaviour:
    def __init__(
        self,
        circuitData,
        volCol=None,
        minSamplesPercentage=None,
        timeWindowForClassification=60,
        vDrop=None,
        powerMeasError=None,
        vMeasError=None,
    ):
        # mandatory params
        self.circuitData = circuitData
        # which voltage value to consider
        self.volCol = "vmean" if volCol is None else volCol

        self.minSamplesPercentage = (
            0.1 if minSamplesPercentage is None else minSamplesPercentage
        )
        self.timeWindowForClassification = (
            3 if timeWindowForClassification is None else timeWindowForClassification
        )

        # site->terminal mapping (used in KPI math)
        self.vDrop = 2.3 if vDrop is None else vDrop
        self.powerMeasError = 0.04 if powerMeasError is None else powerMeasError
        self.vMeasError = 0 if vMeasError is None else vMeasError
        self._site_day_signals_cache = {}

        # per-row time deltas & next values
        self.circuitData = self.circuitData.with_columns(
            (
                pl.col("local_tstamp").cast(pl.Datetime).shift(-1)
                - pl.col("local_tstamp").cast(pl.Datetime)
            )
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
            c
            for c in df.columns
            if c.startswith("power")
            and not c.endswith("_next")
            and not c.endswith("_logic")
        ]

    # ------------------------------ OV1 KPI (anti‑islanding) ------------------------------
    # (Kept as‑is for assessment; diagnostics are not emitted here)

    def build_site_day_signals(self, PRated):
        """
        Prepare the site-day frame used by Phase A and Phase B.
        The dataframe stays on the measured/switchboard basis; site-specific
        voltage shifts are learned later in Phase A.
        """
        cache_key = None if PRated is None else float(PRated)
        if cache_key in self._site_day_signals_cache:
            return self._site_day_signals_cache[cache_key]

        df = self.circuitData.clone()
        power_cols = self._power_columns(df)
        power_cols_next = [
            c
            for c in df.columns
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
                pl.when(pl.col(c) < 0)
                .then(pl.lit(0.0))
                .otherwise(pl.col(c))
                .alias(logic_name)
            )
            logic_current.append(logic_name)

            next_name = f"{c}_next"
            if next_name in power_cols_next:
                df = df.with_columns(
                    pl.when(pl.col(next_name) < 0)
                    .then(pl.lit(0.0))
                    .otherwise(pl.col(next_name))
                    .alias(logic_next_name)
                )
            else:
                df = df.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias(logic_next_name)
                )
            logic_next.append(logic_next_name)

        df = df.with_columns(
            [
                pl.when(
                    pl.all_horizontal([pl.col(c).is_not_null() for c in logic_current])
                )
                .then(pl.sum_horizontal([pl.col(c) for c in logic_current]))
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("site_power"),
                pl.when(
                    pl.all_horizontal([pl.col(c).is_not_null() for c in logic_next])
                )
                .then(pl.sum_horizontal([pl.col(c) for c in logic_next]))
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("site_power_next"),
            ]
        )

        df = df.with_columns(
            [
                pl.when(pl.col("site_power").is_not_null())
                .then(
                    pl.all_horizontal(
                        [pl.col(c) <= p_disconnect for c in logic_current]
                    )
                    & (pl.col("site_power") <= p_disconnect)
                )
                .otherwise(pl.lit(None, dtype=pl.Boolean))
                .alias("is_disc"),
                pl.when(pl.col("site_power_next").is_not_null())
                .then(
                    pl.all_horizontal([pl.col(c) <= p_disconnect for c in logic_next])
                    & (pl.col("site_power_next") <= p_disconnect)
                )
                .otherwise(pl.lit(None, dtype=pl.Boolean))
                .alias("is_disc_next"),
            ]
        )

        df = df.with_columns(
            [
                (
                    pl.col("is_disc").is_not_null()
                    & (
                        pl.col("is_disc").fill_null(False)
                        | pl.col("is_disc_next").is_not_null()
                    )
                ).alias("_power_assessable"),
                (pl.col("site_power").shift(1) - pl.col("site_power")).alias(
                    "site_power_drop"
                ),
                (pl.col("site_power") - pl.col("site_power").shift(1)).alias(
                    "site_power_rise"
                ),
            ]
        )

        df = df.with_columns(
            [
                (pl.col("v10m_avg").is_not_null() & pl.col("_power_assessable")).alias(
                    "eligible_los"
                ),
                (pl.col("vinst_max").is_not_null() & pl.col("_power_assessable")).alias(
                    "eligible_ov1"
                ),
            ]
        )

        df = df.with_columns(
            [
                (
                    (~pl.col("is_disc").shift(1))
                    & pl.col("is_disc")
                    & (pl.col("site_power_drop") >= p_step_strict)
                )
                .fill_null(False)
                .alias("disconnect_edge"),
                (
                    pl.col("is_disc").shift(1)
                    & (~pl.col("is_disc"))
                    & (pl.col("site_power_rise") >= p_step_strict)
                )
                .fill_null(False)
                .alias("reconnect_edge"),
            ]
        )

        df = df.with_columns(
            [
                (
                    (~pl.col("is_disc").shift(1))
                    & pl.col("is_disc")
                    & (pl.col("site_power_drop") >= p_step_fallback)
                    & (~pl.col("disconnect_edge"))
                )
                .fill_null(False)
                .alias("disconnect_edge_fallback"),
                (
                    pl.col("is_disc").shift(1)
                    & (~pl.col("is_disc"))
                    & (pl.col("site_power_rise") >= p_step_fallback)
                    & (~pl.col("reconnect_edge"))
                )
                .fill_null(False)
                .alias("reconnect_edge_fallback"),
            ]
        )

        self._site_day_signals_cache[cache_key] = df
        return self._site_day_signals_cache[cache_key]

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
        timestamp_dtype = df.schema["local_tstamp"]

        disc_rows = (
            df.filter(pl.col("disconnect_edge") | pl.col("disconnect_edge_fallback"))
            .select(
                [
                    "site_id",
                    "local_tstamp",
                    "v10m_avg",
                    "vinst_max",
                    "site_power_drop",
                    "disconnect_edge",
                    "disconnect_edge_fallback",
                ]
            )
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
            .select(
                [
                    "site_id",
                    "local_tstamp",
                    "v10m_avg",
                    "vinst_max",
                    "reconnect_edge",
                    "reconnect_edge_fallback",
                ]
            )
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
            site_power_drop_pct = (
                None
                if PRated in [None, 0] or site_power_drop_kw is None
                else (float(site_power_drop_kw) / float(PRated)) * 100.0
            )
            mech = None
            threshold_voltage = None
            grey_non_sustained = False
            reconnect = next(
                (r for r in reconnect_list if r["local_tstamp"] > tdisc), None
            )
            vinst_in_ov1_region = vinst is not None and (
                los_hi_strict <= vinst <= los_hi_cap
            )

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

            brackets.append(
                {
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
                }
            )

        rec_df = (
            pl.DataFrame(records)
            if records
            else pl.DataFrame(
                schema={
                    "site_id": pl.Int64,
                    "event_id": pl.Int64,
                    "ts_disc": timestamp_dtype,
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
        ).with_columns(
            [
                pl.col("site_id").cast(pl.Int64),
                pl.col("event_id").cast(pl.Int64),
                pl.col("ts_disc").cast(timestamp_dtype),
                pl.col("edge_source").cast(pl.Utf8),
                pl.col("mech").cast(pl.Utf8),
                pl.col("v_los_recorded").cast(pl.Float64),
                pl.col("v_ov1_recorded").cast(pl.Float64),
                pl.col("v10m_disc").cast(pl.Float64),
                pl.col("vinst_disc").cast(pl.Float64),
                pl.col("site_power_drop_kw").cast(pl.Float64),
                pl.col("site_power_drop_pct_rated").cast(pl.Float64),
                pl.col("grey_non_sustained").cast(pl.Boolean),
            ]
        )
        br_df = (
            pl.DataFrame(brackets)
            if brackets
            else pl.DataFrame(
                schema={
                    "site_id": pl.Int64,
                    "event_id": pl.Int64,
                    "edge_source": pl.Utf8,
                    "mech": pl.Utf8,
                    "ts_disc": timestamp_dtype,
                    "ts_rec": timestamp_dtype,
                    "L": pl.Float64,
                    "U": pl.Float64,
                    "midpoint": pl.Float64,
                    "width": pl.Float64,
                }
            )
        ).with_columns(
            [
                pl.col("site_id").cast(pl.Int64),
                pl.col("event_id").cast(pl.Int64),
                pl.col("edge_source").cast(pl.Utf8),
                pl.col("mech").cast(pl.Utf8),
                pl.col("ts_disc").cast(timestamp_dtype),
                pl.col("ts_rec").cast(timestamp_dtype),
                pl.col("L").cast(pl.Float64),
                pl.col("U").cast(pl.Float64),
                pl.col("midpoint").cast(pl.Float64),
                pl.col("width").cast(pl.Float64),
            ]
        )

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

        df = df.with_columns(
            [
                (
                    pl.col("eligible_ov1")
                    & (pl.col("vinst_max") >= (ov1_work_threshold - tau))
                ).alias("ov1_responsible"),
            ]
        )
        df = df.with_columns(
            [
                (
                    pl.col("eligible_los")
                    & (~pl.col("ov1_responsible"))
                    & (pl.col("v10m_avg") > los_threshold)
                ).alias("los_responsible"),
                (
                    pl.col("is_disc").fill_null(False)
                    | pl.col("is_disc_next").fill_null(False)
                ).alias("is_disc_current_or_next"),
            ]
        )
        df = df.with_columns(
            [
                (pl.col("los_responsible") & pl.col("is_disc_current_or_next")).alias(
                    "los_compliant"
                ),
                (pl.col("ov1_responsible") & pl.col("is_disc_current_or_next")).alias(
                    "ov1_compliant"
                ),
            ]
        )

        detail = df.select(
            [
                "site_id",
                "local_tstamp",
                "utc_tstamp",
                "v10m_avg",
                "vinst_max",
                "eligible_los",
                "eligible_ov1",
                "is_disc",
                "is_disc_next",
                "los_responsible",
                "ov1_responsible",
                "los_compliant",
                "ov1_compliant",
            ]
        )

        summary = {
            "los_eligible": int(detail.filter(pl.col("los_responsible")).height),
            "los_compliant": int(detail.filter(pl.col("los_compliant")).height),
            "ov1_eligible": int(detail.filter(pl.col("ov1_responsible")).height),
            "ov1_compliant": int(detail.filter(pl.col("ov1_compliant")).height),
        }

        return {"frame": df, "detail": detail, "summary": summary}
