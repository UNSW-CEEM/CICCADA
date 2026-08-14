"""
The GHI counterfactual: structured data, model, uncurtailed PV.
===============================================================

Deliverables D12b and D12c. Ports of ``build_structured_data.py``,
``build_ghi_model.py`` and ``build_all_uncurtailedpv.py`` to DuckDB.

Three steps, in order::

    build_structured()    se_interval + bom_solar -> se_structured
                          (GHI, empirical GHI_cs, normalised P)
    fit_ghi_model()       per-site, per-5-min-ToD regression -> se_ghi_model
    build_uncurtailedpv() apply the model -> se_uncurtailedpv

Why GHI_cs cannot be replaced by pvlib
--------------------------------------
``GHI_cs`` in the original is NOT a modelled clear-sky curve. It is derived from
the BOM data itself: pick the clearest day per month per grid point (lowest
``cloud_sum``, ``max_GHI > 200``), then take a percentile of GHI over a window.

The model's regressor is the RATIO ``GHI / GHI_cs``. Substituting a modelled
clear-sky irradiance changes what that ratio means, and the fitted coefficients
would no longer be comparable to the Solar Analytics ones. That is why D12
insists on the BOM extract rather than falling back to pvlib.

The postcode compromise, restated because it governs coverage
-------------------------------------------------------------
Solar Analytics had per-site coordinates. This has a postcode. Following
``BOM_NCI/process_bom.ipynb``, irradiance is AVERAGED over all BOM nodes inside
the postcode polygon -- with the site's true location unknown, an average is a
better estimator than snapping to one node.

Expect the MAPE quality gate to reject large-area postcodes preferentially. That
is correct behaviour and NOT random attrition: coverage will be biased toward
dense urban postcodes. ``postcode_area_km2`` travels through to the gate report
so the bias is auditable.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from solar_edge.config import se_config as C
from solar_edge.lib import se_params

__all__ = [
    "build_structured",
    "fit_ghi_model",
    "build_uncurtailedpv",
    "mape_quality_gate",
    "counterfactual_coverage",
    "check_ghi_alignment",
    "CLEAR_SKY_MIN_MAX_GHI",
    "CLEAR_SKY_PROFILE_PERCENTILE",
    "TIME_BIN_MIN",
    "MAPE_MAX",
]

#: Constants carried over from build_structured_data.py / build_ghi_model.py.
#: Modelling choices, not AS/NZS 4777.2 or BOM requirements.
CLEAR_SKY_MIN_MAX_GHI = 200        # a clear-sky reference day must peak above this
CLEAR_SKY_PROFILE_PERCENTILE = 0.60
CLEAR_SKY_MAX_DAY_DISTANCE = 45    # days
TIME_BIN_MIN = 5                   # time-of-day bin for the model
MAPE_MAX = 0.50                    # site quality gate
MIN_TRAIN_POINTS = 5


def _require(table: str, hint: str):
    if not C.store_path(table).exists():
        raise FileNotFoundError(
            f"{table} not found at {C.store_path(table)}.\n  {hint}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# D12b. STRUCTURED DATA
# ═══════════════════════════════════════════════════════════════════════════

def build_structured(
    con: duckdb.DuckDBPyConnection, config=None, write: bool = True
) -> pd.DataFrame:
    """
    Join irradiance to telemetry and build the clear-sky references.

    Steps, mirroring ``build_structured_data.build_sql``:

    1. **Postcode-average the BOM grid** (per ``process_bom.ipynb``).
    2. **10-min -> 5-min by DUPLICATION**, not interpolation: each reading is
       emitted at ``t`` and again at ``t + 5 min``. The practical resolution stays
       10 minutes and must be described that way.
    3. **Nearest-timestamp join** to ``se_interval``. Note SolarEdge timestamps sit
       on per-site 5-minute offsets rather than a common grid, so this rounds to
       the nearest 5-minute slot -- a step the Solar Analytics pipeline did not
       need, and a small extra source of misalignment.
    4. **Clear-sky day selection**: per postcode-month, the day with the lowest
       summed ``cloud_type`` whose ``max_GHI`` exceeds 200.
    5. **``GHI_cs`` and ``P_kw_norm_cs``**: the 60th percentile over a window on
       the nearest clear-sky day.
    """
    config = (config or se_params.CONFIG).validate()
    _require("bom_solar", "Run notebook 04 (BOM extract) first.")

    out = C.store_path("se_structured")
    # bom_solar arrives ALREADY averaged to postcode -- se_bom.extract_bom does the
    # `groupby(['time','postcode']).mean()` inside Athena rather than shipping
    # per-node rows and aggregating here. Nothing further to collapse.
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW _bom_postcode AS
        SELECT postcode, time, GHI, cloud_type FROM bom_solar
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW _bom_5min AS
        -- 10-min -> 5-min by duplication (t and t+5), exactly as the original.
        SELECT postcode, time AS time_5min, GHI, cloud_type FROM _bom_postcode
        UNION ALL
        SELECT postcode, time + INTERVAL '5' MINUTE, GHI, cloud_type FROM _bom_postcode
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW _clear_sky_days AS
        WITH daily AS (
            SELECT postcode,
                   CAST(time_5min + INTERVAL '10' HOUR AS DATE)          AS day_aest,
                   date_trunc('month', time_5min + INTERVAL '10' HOUR)   AS month_aest,
                   sum(cloud_type)                                       AS cloud_sum,
                   max(GHI)                                              AS max_GHI
            FROM _bom_5min GROUP BY 1, 2, 3
        )
        SELECT postcode, day_aest, month_aest FROM (
            SELECT *, row_number() OVER (
                       PARTITION BY postcode, month_aest ORDER BY cloud_sum
                   ) AS rn
            FROM daily WHERE max_GHI > {CLEAR_SKY_MIN_MAX_GHI}
        ) WHERE rn = 1
        """
    )

    frame_sql = f"""
        WITH site_norm AS (
            SELECT i.site_alias, i.ts_aest, i.ts_utc, i.postcode,
                   i.P_kW, i.Q_kvar, i.V_mean, i.V_max, i.derating_active,
                   c.s_99                                   AS normalization_capacity,
                   i.P_kW / nullif(c.s_99, 0)               AS P_kw_norm,
                   i.Q_kvar / nullif(c.s_99, 0)             AS Q_kvar_norm,
                   -- Round to the nearest 5-min slot: SolarEdge timestamps carry
                   -- per-site offsets, so an exact join would match almost nothing.
                   date_trunc('minute', i.ts_aest)
                     - INTERVAL '1' MINUTE * (minute(i.ts_aest) % {TIME_BIN_MIN})
                                                            AS slot_aest
            FROM se_interval i
            JOIN se_site_capacity c USING (site_alias)
            WHERE c.s_99 > 0
        ),
        with_ghi AS (
            SELECT s.*, b.GHI, b.cloud_type
            FROM site_norm s
            LEFT JOIN _bom_5min b
              ON b.postcode = s.postcode
             AND b.time_5min + INTERVAL '10' HOUR = s.slot_aest
        ),
        cs_ref AS (
            SELECT w.postcode,
                   CAST(w.slot_aest AS TIME)                 AS tod,
                   quantile_cont(w.GHI, {CLEAR_SKY_PROFILE_PERCENTILE})   AS GHI_cs,
                   quantile_cont(w.P_kw_norm, {CLEAR_SKY_PROFILE_PERCENTILE}) AS P_kw_norm_cs
            FROM with_ghi w
            JOIN _clear_sky_days d
              ON d.postcode = w.postcode
             AND CAST(w.slot_aest AS DATE) = d.day_aest
            GROUP BY 1, 2
        )
        SELECT w.site_alias, w.ts_aest, w.ts_utc, w.postcode,
               w.P_kW, w.Q_kvar, w.V_mean, w.V_max, w.derating_active,
               w.normalization_capacity, w.P_kw_norm, w.Q_kvar_norm,
               w.GHI, w.cloud_type,
               r.GHI_cs, r.P_kw_norm_cs,
               CAST(w.slot_aest AS TIME)                     AS tod_bin,
               CAST(w.slot_aest AS DATE)                     AS actual_day,
               strftime(w.ts_aest, '%Y-%m')                  AS {C.PARTITION_KEY}
        FROM with_ghi w
        LEFT JOIN cs_ref r
          ON r.postcode = w.postcode AND CAST(w.slot_aest AS TIME) = r.tod
        WHERE w.GHI IS NOT NULL
    """

    if write:
        out.mkdir(parents=True, exist_ok=True)
        con.execute(
            f"""
            COPY ({frame_sql} ORDER BY site_alias, ts_utc) TO '{out.as_posix()}'
            (FORMAT PARQUET, PARTITION_BY ({C.PARTITION_KEY}),
             COMPRESSION '{C.PARQUET_COMPRESSION}',
             ROW_GROUP_SIZE {C.PARQUET_ROW_GROUP_SIZE}, OVERWRITE_OR_IGNORE true)
            """
        )
        from solar_edge.lib import se_store
        se_store.register_store_views(con)
        return con.execute("SELECT count(*) AS n_rows FROM se_structured").df()
    return con.execute(frame_sql + " LIMIT 1000").df()


# ═══════════════════════════════════════════════════════════════════════════
# D12c. GHI MODEL AND COUNTERFACTUAL
# ═══════════════════════════════════════════════════════════════════════════

def fit_ghi_model(
    con: duckdb.DuckDBPyConnection, train_fraction: float = 0.7, write: bool = True
) -> pd.DataFrame:
    """
    Per-site, per-time-of-day regression of normalised power on the clear-sky index.

        P_norm / P_norm_cs = a + b * (GHI / GHI_cs),   with a = 1 - b

    The constraint ``a = 1 - b`` forces the line through (1, 1): on a clear-sky
    interval both ratios are 1 by construction. That reduces the fit to a single
    parameter, which is what makes it stable on the small per-ToD-bin samples.

    Training filters are carried over verbatim from ``build_ghi_model.py``::

        P_kw_norm_cs > 0.2          exclude dawn/dusk (reference too small)
        GHI > 50                    exclude low-light noise
        P_kw_norm > 0.05            exclude near-zero generation
        P_kw_norm <= P_kw_norm_cs   quality gate
        V <= 253                    exclude the Volt-Watt active zone
        (P_kw_norm >= 1 OR S_norm < 1.001)
                                    exclude Volt-VAr-curtailed intervals

    That last filter is the important one, and it is imperfect. It tries to keep
    curtailed intervals out of the training set, but cannot remove them all --
    which is precisely why the resulting counterfactual is conservative and
    Method B is a LOWER bound.
    """
    _require("se_structured", "Run build_structured() first.")
    out = C.store_path("se_ghi_model")

    sql = f"""
        WITH train AS (
            SELECT site_alias, tod_bin,
                   GHI / nullif(GHI_cs, 0)                     AS x,
                   P_kw_norm / nullif(P_kw_norm_cs, 0)         AS y
            FROM se_structured
            WHERE GHI_cs > 0 AND P_kw_norm_cs > 0.2
              AND GHI > 50
              AND P_kw_norm > 0.05
              AND P_kw_norm <= P_kw_norm_cs
              AND V_mean <= {C.as4777()['VW']['V1']}
              AND (P_kw_norm >= 1
                   OR sqrt(P_kw_norm * P_kw_norm + Q_kvar_norm * Q_kvar_norm) < 1.001)
              -- deterministic train split: hash, not random(), so a rerun
              -- reproduces the same model. The legacy ORDER BY random() split was
              -- flagged as non-reproducible.
              AND (hash(site_alias || CAST(tod_bin AS VARCHAR)) % 100) < {int(train_fraction * 100)}
        ),
        centred AS (
            -- a = 1 - b  =>  (y - 1) = b * (x - 1). Slope through the origin on
            -- the centred variables, so b = sum((x-1)(y-1)) / sum((x-1)^2).
            SELECT site_alias, tod_bin, x - 1.0 AS xc, y - 1.0 AS yc FROM train
        )
        SELECT site_alias, tod_bin,
               sum(xc * yc) / nullif(sum(xc * xc), 0)  AS b,
               1.0 - sum(xc * yc) / nullif(sum(xc * xc), 0) AS a,
               count(*)                                AS n
        FROM centred
        GROUP BY site_alias, tod_bin
        HAVING count(*) >= {MIN_TRAIN_POINTS}
    """
    frame = con.execute(sql).df()
    if write:
        out.parent.mkdir(parents=True, exist_ok=True)
        con.register("_model", frame)
        con.execute(f"COPY _model TO '{out.as_posix()}' (FORMAT PARQUET)")
        con.unregister("_model")
        from solar_edge.lib import se_store
        se_store.register_store_views(con)
    return frame


def mape_quality_gate(
    con: duckdb.DuckDBPyConnection, mape_max: float = MAPE_MAX
) -> pd.DataFrame:
    """
    Per-site MAPE of the fitted model, and the pass/fail gate.

    Sites above ``mape_max`` are excluded from the counterfactual entirely. Their
    intervals then have NO ``uncurtailed_P``, which Method B must report as
    *missing coverage* rather than as zero curtailment. Substituting zero is the
    defect that made the legacy counterfactual score sites conservatively.

    ``postcode_area_km2`` is carried through so the expected bias -- large rural
    postcodes failing more often, because their averaged irradiance is a worse
    proxy for the site -- is visible rather than inferred.
    """
    _require("se_ghi_model", "Run fit_ghi_model() first.")
    return con.execute(
        f"""
        WITH pred AS (
            SELECT s.site_alias,
                   s.P_kw_norm,
                   s.P_kw_norm_cs * (m.a + m.b * (s.GHI / nullif(s.GHI_cs, 0)))
                       AS P_kw_norm_est
            FROM se_structured s
            JOIN se_ghi_model m
              ON s.site_alias = m.site_alias AND s.tod_bin = m.tod_bin
            WHERE s.GHI_cs > 0 AND s.P_kw_norm_cs > 0.2 AND s.P_kw_norm > 0.05
        )
        SELECT p.site_alias,
               count(*)                                                AS n_eval,
               round(median(abs(P_kw_norm_est - P_kw_norm)
                            / nullif(P_kw_norm, 0)), 4)                AS mape,
               (median(abs(P_kw_norm_est - P_kw_norm)
                       / nullif(P_kw_norm, 0)) <= {mape_max})          AS passes_gate,
               round(any_value(st.postcode_area_km2), 1)               AS postcode_area_km2
        FROM pred p
        LEFT JOIN se_site st USING (site_alias)
        GROUP BY p.site_alias
        HAVING count(*) >= {MIN_TRAIN_POINTS}
        """
    ).df()


def build_uncurtailedpv(
    con: duckdb.DuckDBPyConnection, gate: pd.DataFrame | None = None, write: bool = True
) -> pd.DataFrame:
    """
    Apply the model to every eligible interval -> ``se_uncurtailedpv``.

        P_norm_est    = P_norm_cs * (a + b * GHI/GHI_cs)
        uncurtailed_P = greatest(P_norm_est, P_measured_norm) * capacity

    The FLOOR at observed power matters: the counterfactual must never claim the
    site would have produced *less* than it actually did. It is applied before
    any capacity cap, matching the original ordering, and both the floored and
    capped flags are written so either can be audited.

    Only sites passing the MAPE gate get a counterfactual. Everything else is
    absent, not zero.
    """
    _require("se_ghi_model", "Run fit_ghi_model() first.")
    gate = mape_quality_gate(con) if gate is None else gate
    passing = gate.loc[gate.passes_gate, "site_alias"]
    if len(passing) == 0:
        # Without this the COPY writes no files, the view is never created, and
        # the caller gets an opaque "Table se_uncurtailedpv does not exist".
        raise ValueError(
            f"No site passed the MAPE <= {MAPE_MAX:.0%} quality gate "
            f"({len(gate):,} evaluated), so there is nothing to write.\n"
            "  Check check_ghi_alignment() first -- a timezone mismatch between\n"
            "  bom_solar.time and se_interval.ts_aest produces a model that fits\n"
            "  noise and fails the gate everywhere."
        )
    con.register("_gate", pd.DataFrame({"site_alias": passing}))

    out = C.store_path("se_uncurtailedpv")
    sql = f"""
        SELECT s.site_alias, s.ts_aest, s.ts_utc,
               s.P_kW,
               s.GHI, s.GHI_cs,
               m.n                                              AS n_train,
               s.P_kw_norm_cs * (m.a + m.b * (s.GHI / nullif(s.GHI_cs, 0)))
                   * s.normalization_capacity                   AS model_prediction_raw,
               greatest(
                   s.P_kw_norm_cs * (m.a + m.b * (s.GHI / nullif(s.GHI_cs, 0)))
                       * s.normalization_capacity,
                   s.P_kW
               )                                                AS uncurtailed_P,
               (s.P_kw_norm_cs * (m.a + m.b * (s.GHI / nullif(s.GHI_cs, 0)))
                    * s.normalization_capacity < s.P_kW)        AS floor_applied,
               s.normalization_capacity                         AS capacity_limit,
               strftime(s.ts_aest, '%Y-%m')                     AS {C.PARTITION_KEY}
        FROM se_structured s
        JOIN se_ghi_model m
          ON s.site_alias = m.site_alias AND s.tod_bin = m.tod_bin
        JOIN _gate g ON s.site_alias = g.site_alias
        WHERE s.GHI_cs > 0 AND s.P_kw_norm_cs > 0
    """
    try:
        if write:
            out.mkdir(parents=True, exist_ok=True)
            con.execute(
                f"""
                COPY ({sql} ORDER BY site_alias, ts_utc) TO '{out.as_posix()}'
                (FORMAT PARQUET, PARTITION_BY ({C.PARTITION_KEY}),
                 COMPRESSION '{C.PARQUET_COMPRESSION}',
                 ROW_GROUP_SIZE {C.PARQUET_ROW_GROUP_SIZE}, OVERWRITE_OR_IGNORE true)
                """
            )
            from solar_edge.lib import se_store
            se_store.register_store_views(con)
            return con.execute(
                "SELECT count(*) AS n_rows, count(DISTINCT site_alias) AS n_sites "
                "FROM se_uncurtailedpv"
            ).df()
        return con.execute(sql + " LIMIT 1000").df()
    finally:
        con.unregister("_gate")


def check_ghi_alignment(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Does irradiance actually line up with generation in time? Run before fitting.

    ``build_structured`` assumes ``bom_solar.time`` is **UTC** and converts with a
    fixed ``+ INTERVAL '10' HOUR`` to the AEST analysis frame. If that assumption
    is wrong -- BOM ships local time, or the extract already shifted it -- every
    step downstream still *runs*. ``fit_ghi_model`` returns coefficients,
    ``build_uncurtailedpv`` produces a counterfactual, and Method B produces a
    curtailment number. All of it is meaningless, and nothing raises.

    The signature of a misalignment is that GHI peaks when generation does not.
    This bins both by hour of the AEST day and reports the correlation and the
    gap between their peak hours. On correctly aligned data the two peak within
    an hour of each other and correlate above ~0.8; a ten-hour offset shows up as
    a near-zero or negative correlation and a peak_hour_gap around 10.

    Encountered for real while smoke-testing this pipeline: with the frames
    mismatched, ``GHI_cs > 0`` and ``P_kw_norm_cs > 0.2`` never co-occurred and
    the training set came out empty. That is the loud failure. The quiet one --
    a one- or two-hour offset -- would not empty the training set, it would just
    bias every counterfactual, so check the peak gap, not only the row count.
    """
    _require("se_structured", "Run build_structured() first.")
    profile = con.execute(
        """
        SELECT hour(ts_aest)      AS hour_aest,
               avg(GHI)           AS mean_GHI,
               avg(P_kw_norm)     AS mean_P_norm,
               count(*)           AS n
        FROM se_structured
        WHERE GHI IS NOT NULL AND P_kw_norm IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    if profile.empty:
        raise ValueError("se_structured has no rows with both GHI and P_kw_norm.")

    corr = profile.mean_GHI.corr(profile.mean_P_norm)
    ghi_peak = int(profile.loc[profile.mean_GHI.idxmax(), "hour_aest"])
    p_peak = int(profile.loc[profile.mean_P_norm.idxmax(), "hour_aest"])
    gap = min(abs(ghi_peak - p_peak), 24 - abs(ghi_peak - p_peak))

    verdict = ("ALIGNED" if corr > 0.8 and gap <= 1 else
               "SUSPECT" if corr > 0.5 and gap <= 2 else "MISALIGNED")
    print(f"GHI vs generation diurnal alignment: {verdict}")
    print(f"  correlation over hour-of-day : {corr:.3f}   (want > 0.8)")
    print(f"  GHI peaks at {ghi_peak:02d}:00 AEST, generation at {p_peak:02d}:00 AEST"
          f"  -> gap {gap} h   (want <= 1)")
    if verdict != "ALIGNED":
        print("  bom_solar.time is assumed UTC and shifted +10 h in build_structured.\n"
              "  If BOM ships AEST or local time, that shift is wrong and every\n"
              "  downstream number is invalid despite running cleanly.")
    profile.attrs["verdict"] = verdict
    profile.attrs["correlation"] = corr
    profile.attrs["peak_hour_gap"] = gap
    return profile


def counterfactual_coverage(con: duckdb.DuckDBPyConnection, config=None) -> pd.DataFrame:
    """
    How much of the fleet actually got a counterfactual, and how that correlates
    with postcode size.

    The headline number for D12c. If coverage is low and concentrated in small
    urban postcodes, Method B is measuring a biased subset and must say so.
    """
    _require("se_uncurtailedpv", "Run build_uncurtailedpv() first.")
    return con.execute(
        """
        WITH per_site AS (
            SELECT st.site_alias,
                   any_value(st.postcode_area_km2) AS postcode_area_km2,
                   count(u.ts_utc)                 AS n_counterfactual
            FROM se_site st
            LEFT JOIN se_uncurtailedpv u USING (site_alias)
            GROUP BY st.site_alias
        )
        SELECT CASE WHEN postcode_area_km2 < 25   THEN 'a. < 25 km2'
                    WHEN postcode_area_km2 < 100  THEN 'b. 25-100'
                    WHEN postcode_area_km2 < 1000 THEN 'c. 100-1000'
                    ELSE 'd. > 1000 km2' END              AS postcode_size,
               count(*)                                   AS n_sites,
               count(*) FILTER (WHERE n_counterfactual > 0) AS n_with_counterfactual,
               round(100.0 * count(*) FILTER (WHERE n_counterfactual > 0) / count(*), 2)
                                                          AS pct_covered
        FROM per_site
        WHERE postcode_area_km2 IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """
    ).df()
