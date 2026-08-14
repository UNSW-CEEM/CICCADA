"""
Single-site exploration, with explicit sign provenance.
=======================================================

For interrogating individual sites and their daily curves, ported from
``ausgrid_analysis/notebooks/voltvar_voltwatt_lib.plot_operational_ausgrid``.

Sign provenance is the point of this module
-------------------------------------------
Every function here returns BOTH orientations side by side and labels them, so it
is never ambiguous which one you are looking at:

``Q_raw_var``
    Straight from the delivered Parquet, byte for byte. Nothing applied. This is
    SolarEdge's ``reactive_power_1 + _2 + _3`` in var.

``Q_stored_kvar``
    What ``se_interval`` holds, and what every conformance number in D9-D10 was
    computed from. By construction::

        Q_stored_kvar = REACTIVE_POWER_SIGN * Q_raw_var / 1000

    ``REACTIVE_POWER_SIGN`` is **+1 since 13 Aug** -- the store now holds the value
    AS DELIVERED, so stored and raw are identical. It was -1 for one day; anything
    computed then is on the other convention.

``verify_sign_transform()`` asserts that identity against the raw files rather
than trusting it, because the whole sign question turns on it.

Active power is NOT transformed: ``ACTIVE_POWER_SIGN = +1``. SolarEdge reports a
production magnitude with no negative values anywhere in the delivery, so
``P_kW = (p1 + p2 + p3) / 1000`` and there is no orientation question for P.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from solar_edge.config import se_config as C
from solar_edge.lib import se_contract as contract
from solar_edge.lib import se_params

C.bootstrap_sys_path()
from bms_sa_review.shared.as4777_curves import vvar_required_q  # noqa: E402

__all__ = [
    "site_day",
    "site_days_available",
    "site_voltvar_curve",
    "plot_operational",
    "site_voltwatt_day",
    "plot_voltwatt_day",
    "plot_voltvar_scatter",
    "site_voltwatt_categories",
    "voltwatt_day_summary",
    "voltwatt_counterfactual_gaps",
    "counterfactual_training_attrition",
    "VOLTWATT_CATEGORIES",
    "CATEGORY_COLORS",
    "plot_site_voltvar_curve",
    "verify_sign_transform",
    "site_profile",
    "site_interval_categories",
    "site_category_calendar",
    "plot_category_heatmap",
    "INTERVAL_CATEGORIES",
]

_A = C.as4777()

# Shared with se_plots so the explorer and the fleet figures use one palette.
# (Kept as literals rather than importing se_plots, which would make the two
# modules mutually dependent.)
C_V, C_P, C_Q = "#b45309", "#2a7f62", "#4709b2"
C_ACCENT, C_GREY = "#e8702a", "#8a8a8a"
C_REQ, C_BAD = "#d946ef", "#c62828"

#: Labels stamped onto every figure and table, so a plot can never be mistaken
#: for the other orientation.
#: Built from the live config so a plot can never claim a transform the ingest
#: is not applying. These labels were hard-coded to "-1 x raw" and went stale the
#: moment REACTIVE_POWER_SIGN changed on 13 Aug.
_SIGN = C.REACTIVE_POWER_SIGN
ORIENTATIONS = {
    "stored": (
        f"STORED (se_interval)  Q = {_SIGN:+.0f} x raw / 1000  "
        + ("[sign UNCHANGED from delivery]" if _SIGN > 0 else "[flip IS applied]")
    ),
    "raw": (
        "RAW (as delivered)  Q = raw / 1000, sign untouched"
        + ("  [identical to stored]" if _SIGN > 0 else "  [no flip applied]")
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# FETCH
# ═══════════════════════════════════════════════════════════════════════════

def _done(fig):
    """
    Close the figure so ``display(fig)`` renders it exactly once.

    With the inline backend an OPEN figure is also auto-rendered at the end of the
    cell, so every plot appeared twice. Closing suppresses that; display() still
    works because it draws from the figure object, not the pyplot state.
    """
    plt.close(fig)
    return fig


def _raw_site_sql(site_alias: str) -> str:
    """
    The raw delivery for one site, timezone-resolved the same way the ingest does,
    so it can be joined to the store on ``ts_utc``.

    Reuses ``se_ingest.tz_case_sql`` / ``ts_utc_sql`` rather than restating the
    conversion -- if the ingest's timezone logic ever changes, this follows it
    instead of silently disagreeing.
    """
    from solar_edge.lib import se_ingest as ing

    tz_case = ing.tz_case_sql("a.state")
    ts_utc = ing.ts_utc_sql("tz.timestamp", "tz.tz_name")
    return f"""
        WITH tz AS (
            SELECT r.*, {tz_case} AS tz_name
            FROM se_raw r
            JOIN se_alias a ON r.site_alias = a.alias
            WHERE r.site_alias = '{site_alias}'
        )
        SELECT {ts_utc} AS ts_utc,
               CASE WHEN reactive_power_1 IS NOT NULL
                      OR reactive_power_2 IS NOT NULL
                      OR reactive_power_3 IS NOT NULL
                    THEN coalesce(reactive_power_1, 0)
                       + coalesce(reactive_power_2, 0)
                       + coalesce(reactive_power_3, 0) END AS Q_raw_var,
               CASE WHEN active_power_1 IS NOT NULL
                      OR active_power_2 IS NOT NULL
                      OR active_power_3 IS NOT NULL
                    THEN coalesce(active_power_1, 0)
                       + coalesce(active_power_2, 0)
                       + coalesce(active_power_3, 0) END AS P_raw_w
        FROM tz
    """


def verify_sign_transform(con, site_alias: str) -> pd.DataFrame:
    """
    Prove, against the raw Parquet, that ``Q_stored_kvar == -Q_raw_var / 1000``.

    Re-reads the delivered file, re-applies the timezone conversion independently,
    joins on the UTC instant, and compares. The whole sign question turns on this
    one transform, so it is asserted rather than assumed -- any mismatch means the
    store is not what this module claims and every downstream conclusion is void.

    Also checks that active power was NOT transformed, since ``ACTIVE_POWER_SIGN``
    is +1 and any drift there would be just as consequential.
    """
    return con.execute(
        f"""
        WITH raw AS ({_raw_site_sql(site_alias)}),
        joined AS (
            SELECT r.Q_raw_var, r.P_raw_w, i.Q_kvar AS Q_stored_kvar, i.P_kW
            FROM raw r
            JOIN se_interval i
              ON i.site_alias = '{site_alias}' AND i.ts_utc = r.ts_utc
            WHERE r.Q_raw_var IS NOT NULL
        )
        SELECT count(*)                                                   AS n_compared,
               '{C.REACTIVE_POWER_SIGN:+.0f} x raw / 1000'                AS expected_Q_rule,
               sum(CASE WHEN abs(Q_stored_kvar
                                 - ({C.REACTIVE_POWER_SIGN} * Q_raw_var / 1000.0)) > 1e-6
                        THEN 1 ELSE 0 END)                                AS n_Q_mismatched,
               round(max(abs(Q_stored_kvar
                             - ({C.REACTIVE_POWER_SIGN} * Q_raw_var / 1000.0))), 9)
                                                                          AS max_Q_abs_diff,
               '{C.ACTIVE_POWER_SIGN:+.0f} x raw / 1000'                  AS expected_P_rule,
               sum(CASE WHEN abs(P_kW
                                 - ({C.ACTIVE_POWER_SIGN} * P_raw_w / 1000.0)) > 1e-6
                        THEN 1 ELSE 0 END)                                AS n_P_mismatched
        FROM joined
        """
    ).df()


def site_days_available(con, site_alias: str, min_intervals: int = 100,
                        config=None) -> pd.DataFrame:
    """
    Days available for a site, ranked by how interesting they are to look at.

    ``n_above_240`` and ``n_above_253`` are the counts of intervals in the
    Volt-VAr and Volt-Watt active bands. A day with none of either shows a flat
    reactive trace and tells you nothing about the sign.

    Voltage follows ``config.voltage_aggregation`` (default ``mean``) so the
    explorer and the conformance tables describe the same site the same way.
    These functions previously hard-coded ``V_max``, which meant a site could
    look different here than in D9.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "se_interval")
    return con.execute(
        f"""
        SELECT CAST(ts_aest AS DATE)                        AS day_aest,
               count(*)                                     AS n_intervals,
               count(*) FILTER (WHERE {v} > {_A['VVAR']['V3']}) AS n_above_240,
               count(*) FILTER (WHERE {v} > {_A['VW']['V1']})   AS n_above_253,
               round(max({v}), 1)                           AS max_V,
               round(max(P_kW), 2)                          AS max_P_kW,
               round(max(abs(Q_kvar)), 3)                   AS max_abs_Q_kvar
        FROM se_interval
        WHERE site_alias = '{site_alias}'
        GROUP BY 1 HAVING count(*) >= {min_intervals}
        ORDER BY n_above_240 DESC, max_abs_Q_kvar DESC
        """
    ).df()


def site_day(con, site_alias: str, day: str, config=None) -> pd.DataFrame:
    """
    One AEST day for one site, carrying BOTH reactive orientations.

    Columns:
        ts_aest, ts_utc, V, P_kW,
        Q_raw_var        -- as delivered, untouched
        Q_raw_kvar       -- raw / 1000, still untouched sign
        Q_stored_kvar    -- what se_interval holds (= -raw/1000)
        derating_active, s_99

    Both are carried so a plot can be drawn either way and labelled honestly.

    ``V`` follows ``config.voltage_aggregation``, matching the conformance layer.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    return con.execute(
        f"""
        WITH raw AS ({_raw_site_sql(site_alias)})
        SELECT i.ts_aest, i.ts_utc,
               {v}                      AS V,
               i.P_kW,
               r.Q_raw_var,
               r.Q_raw_var / 1000.0     AS Q_raw_kvar,
               i.Q_kvar                 AS Q_stored_kvar,
               i.derating_active,
               c.s_99,
               s.is_three_phase, s.state, s.postcode
        FROM se_interval i
        JOIN se_site s USING (site_alias)
        LEFT JOIN se_site_capacity c USING (site_alias)
        LEFT JOIN raw r ON r.ts_utc = i.ts_utc
        WHERE i.site_alias = '{site_alias}'
          AND CAST(i.ts_aest AS DATE) = DATE '{day}'
        ORDER BY i.ts_aest
        """
    ).df()


def site_voltvar_curve(con, site_alias: str, bin_v: float = 1.0,
                       min_p_kw: float = 0.2, min_intervals: int = 20,
                       config=None) -> pd.DataFrame:
    """
    Median reactive power against voltage for one site, both orientations, with
    the AS/NZS 4777.2 requirement alongside.

    This is the per-site version of the fleet diagnosis: it shows whether the
    site's |Q| tracks the required magnitude, and which orientation puts it on
    the correct side of zero.

    Voltage follows ``config.voltage_aggregation``.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    return con.execute(
        f"""
        SELECT floor({v} / {bin_v}) * {bin_v}          AS v_bin,
               count(*)                                 AS n_intervals,
               round(median(i.Q_kvar), 4)               AS Q_stored_kvar,
               round(median(-i.Q_kvar), 4)              AS Q_flipped_kvar,
               round(median(-1000.0 * i.Q_kvar), 1)     AS Q_raw_var,
               round(median(i.P_kW), 3)                 AS P_kW,
               any_value(c.s_99)                        AS s_99
        FROM se_interval i
        LEFT JOIN se_site_capacity c USING (site_alias)
        WHERE i.site_alias = '{site_alias}'
          AND i.P_kW > {min_p_kw} AND {v} BETWEEN 200 AND 270
        GROUP BY 1 HAVING count(*) >= {min_intervals}
        ORDER BY 1
        """
    ).df()


#: Per-interval category, in the order used for tables and the heatmap.
#: The five Q_impact buckets only fire when Q is OUTSIDE the permitted band, so two
#: more states are needed to account for every interval:
#:   not_assessable  -- |P| < 0.2 x s_99; the standard sets no requirement
#:   within_band     -- assessable and inside the band; this is conformance
INTERVAL_CATEGORIES = (
    "not_assessable",
    "within_band",
    "Q_adverse",
    "Q_inactive",
    "Q_significant_shortfall",
    "Q_near_conformant",
    "Q_major_surplus",
)


def _category_expr() -> str:
    """SQL assigning exactly one category per interval."""
    buckets = " ".join(
        f"WHEN {c} > 0 THEN '{c}'"
        for c in ("Q_adverse", "Q_inactive", "Q_significant_shortfall",
                  "Q_near_conformant", "Q_major_surplus")
    )
    return (
        "CASE WHEN capability_assessable = 0 THEN 'not_assessable' "
        f"{buckets} ELSE 'within_band' END"
    )


def site_interval_categories(con, site_alias: str, day: str | None = None,
                             config=None, params=None) -> pd.DataFrame:
    """
    Per-timestamp conformance categorisation for one site.

    Runs the D9 interval scoring restricted to a single site, so the table and the
    daily plot are guaranteed to describe the same thing. ``day=None`` returns the
    whole year.

    ``Q_impact`` is the signed ratio of measured to required reactive power at the
    nearest permitted band edge: 1.0 = exactly on it, 0 = no response, negative =
    wrong direction. ``dist_to_band_kvar`` is how far outside the band the interval
    sits, and is zero for anything within it.
    """
    from solar_edge.lib import se_conformance as cf

    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()
    day_filter = f"AND CAST(ts_aest AS DATE) = DATE '{day}'" if day else ""

    return con.execute(
        f"""
        WITH scored AS ({cf.voltvar_interval_sql(config, params)})
        SELECT ts_aest,
               round(V, 2)                          AS V,
               round(P_kW, 3)                       AS P_kW,
               round(Q_kvar, 4)                     AS Q_kvar,
               round(Q_min_final, 4)                AS Q_min_permitted,
               round(Q_max_final, 4)                AS Q_max_permitted,
               round(Q_impact, 4)                   AS Q_impact,
               (capability_assessable = 1)          AS assessable,
               {_category_expr()}                   AS category,
               round(greatest(Q_adverse, Q_inactive, Q_significant_shortfall,
                              Q_near_conformant, Q_major_surplus), 4)
                                                    AS dist_to_band_kvar,
               derating_active
        FROM scored
        WHERE site_alias = '{site_alias}' {day_filter}
        ORDER BY ts_aest
        """
    ).df()



# ═══════════════════════════════════════════════════════════════════════════
# VOLT-WATT: THE DAY PLOT THAT SHOWS *WHY* A SITE CONFORMS
# ═══════════════════════════════════════════════════════════════════════════

def site_voltwatt_day(con, site_alias: str, day: str, config=None) -> pd.DataFrame:
    """
    One AEST day with everything the Volt-Watt verdict depends on.

    ``plot_operational`` cannot show Volt-Watt conformance, and that is not a
    defect -- the two questions need different evidence. Volt-VAr asks "is Q on the
    curve", which is visible from Q and V alone. Volt-Watt asks "is P below the
    ceiling", and a site sitting far below the ceiling looks identical whether it
    curtailed correctly or simply had no sun. Distinguishing those needs a third
    trace: what the site COULD have produced.

    Columns: ``ts_aest, V, P_kW, rating_kva, P_ceiling_kW, uncurtailed_P,
    exposed, nonconformant, curtailed``.

    ``P_ceiling_kW`` is evaluated by the same ``vw_max_p_sql`` expression the
    conformance scoring uses, not a re-implementation, so the line drawn here is
    the line the verdict was measured against.

    ``uncurtailed_P`` requires ``se_uncurtailedpv`` (notebook 04 section 5). Absent
    it the column is all-NULL and the plot degrades to ceiling-versus-measured --
    still useful for spotting violations, useless for confirming curtailment.
    """
    from solar_edge.lib import se_conformance as cf

    config = (config or se_params.CONFIG).validate()
    rating = contract.capacity_column(config.rating_basis)
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    v_scored = "round(V, 6)"
    max_p = cf.vw_max_p_sql(v_scored, "rating_kva")
    tol = cf.tol_kw_sql("rating_kva", config.tolerance_fraction)

    have_cf = C.store_path("se_uncurtailedpv").exists()
    if have_cf:
        from solar_edge.lib import se_store

        se_store.register_store_views(con)
        cf_select = "u.uncurtailed_P"
        cf_join = ("LEFT JOIN se_uncurtailedpv u "
                   "ON u.site_alias = i.site_alias AND u.ts_aest = i.ts_aest")
    else:
        cf_select, cf_join = "CAST(NULL AS DOUBLE)", ""

    frame = con.execute(
        f"""
        WITH base AS (
            SELECT i.ts_aest, {v} AS V, i.P_kW, {rating} AS rating_kva,
                   i.derating_active, {cf_select} AS uncurtailed_P
            FROM se_interval i
            {contract.cohort_join_sql('i')}
            {cf_join}
            WHERE i.site_alias = '{site_alias}'
              AND CAST(i.ts_aest AS DATE) = DATE '{day}'
        ),
        ceil AS (
            SELECT *, ({max_p}) + {tol} AS P_ceiling_kW FROM base
        )
        SELECT *,
               ({v_scored} > {_A['VW']['V1']})                       AS exposed,
               ({v_scored} > {_A['VW']['V1']} AND P_kW > P_ceiling_kW) AS nonconformant,
               ({v_scored} > {_A['VW']['V1']}
                 AND P_kW <= P_ceiling_kW
                 AND uncurtailed_P > P_ceiling_kW)                   AS curtailed
        FROM ceil
        ORDER BY ts_aest
        """
    ).df()
    if not have_cf:
        print("No se_uncurtailedpv — plotting ceiling vs measured only.\n"
              "  Build it with notebook 04 section 5 to see whether power was "
              "available.")
    return frame


def plot_voltwatt_day(frame, site_alias: str, day: str, config=None,
                      figsize=(13, 8.0)):
    """
    Volt-Watt day: voltage above, power below, with the ceiling and what the
    site could have produced.

    Ported from ``bms_sa_review.data_query.lib.voltwatt_curtailment
    .plot_voltwatt_curtailment_day``.

    The two shaded regions answer different questions and must not be conflated:

    * **orange -- curtailment.** Available power exceeded the ceiling and the site
      stayed below it. This is Volt-Watt *working*, observed rather than assumed,
      and it is the evidence a "conformant" verdict rests on.
    * **red -- non-conformance.** Measured P above the ceiling while exposed.

    A day with neither shading and a flat green trace well under the ceiling is
    conformant in the sense that nothing was breached, but demonstrates nothing:
    the inverter was never asked to respond. That distinction is exactly what a
    site like a "conformant" Volt-Watt result can hide, and why this plot exists.
    """
    config = (config or se_params.CONFIG).validate()
    if frame.empty:
        print(f"No data for {site_alias} on {day}.")
        return None

    d = frame.copy()
    t = pd.to_datetime(d.ts_aest)
    S = float(pd.to_numeric(d.rating_kva, errors="coerce").median())
    V1 = _A["VW"]["V1"]
    exposed = d.exposed.fillna(False).to_numpy(bool)
    nonconf = d.nonconformant.fillna(False).to_numpy(bool)
    curt = d.curtailed.fillna(False).to_numpy(bool)
    has_cf = d.uncurtailed_P.notna().any()

    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=130, sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.8]})
    for ax in axes:
        ax.fill_between(t, 0, 1, where=exposed,
                        transform=ax.get_xaxis_transform(),
                        color=C_Q, alpha=0.07, lw=0, zorder=0)

    ax = axes[0]
    ax.plot(t, d.V, color=C_V, lw=1.4, label="Measured voltage")
    ax.axhline(V1, color=C_ACCENT, ls="--", lw=1.1,
               label=f"{V1:.0f} V — Volt-Watt starts")
    ax.set_ylabel("Voltage (V)", color=C_V)
    ax.tick_params(axis="y", colors=C_V)
    ax.legend(handles=ax.get_legend_handles_labels()[0]
              + [Patch(facecolor=C_Q, alpha=0.18,
                       label=f"Volt-Watt active (V > {V1:.0f} V)")],
              fontsize=7.5, loc="upper left", framealpha=0.92)
    ax.grid(color="#ebebeb", lw=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    if has_cf:
        ax.fill_between(t, d.P_kW, d.uncurtailed_P, where=curt, interpolate=True,
                        color=C_ACCENT, alpha=0.32, zorder=2,
                        label="Curtailment — power was available, site held below "
                              "the ceiling")
    ax.fill_between(t, d.P_ceiling_kW, d.P_kW, where=nonconf, interpolate=True,
                    color="#c62828", alpha=0.38, zorder=2,
                    label="Non-conformant — P above the ceiling")
    if has_cf:
        ax.plot(t, d.uncurtailed_P, color=C_ACCENT, lw=1.6, ls="--", alpha=0.9,
                zorder=4, label="Uncurtailed P (GHI counterfactual)")
    ax.plot(t, d.P_ceiling_kW, color="#1a1a1a", lw=1.7, zorder=5,
            label=f"Volt-Watt ceiling (+{config.tolerance_fraction:.0%} tol)")
    ax.plot(t, d.P_kW, color=C_P, lw=1.7, zorder=4, label="Measured P")
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time (AEST)")
    ax.legend(fontsize=7.3, loc="upper left", framealpha=0.92)
    ax.grid(color="#ebebeb", lw=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    h = C.INTERVAL_H
    curt_kwh = float(np.where(curt, (d.uncurtailed_P - d.P_kW).fillna(0), 0).sum()) * h
    exc_kwh = float(np.where(nonconf, (d.P_kW - d.P_ceiling_kW).fillna(0), 0).sum()) * h
    verdict = ("no exposure — nothing was tested" if not exposed.any() else
               f"{int(nonconf.sum())} non-conformant, {int(curt.sum())} curtailed "
               f"of {int(exposed.sum())} exposed intervals")
    fig.suptitle(
        f"{site_alias} — Volt-Watt, {day}   |   {verdict}\n"
        f"{curt_kwh:.2f} kWh curtailed  ·  {exc_kwh:.2f} kWh over the ceiling  ·  "
        f"rating {S:.1f} kVA ({config.rating_basis})"
        + ("" if has_cf else "   ·   NO counterfactual — curtailment not assessable"),
        fontsize=10.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    return _done(fig)



#: Category palette, matching ``voltvar_voltwatt_lib.STATUS_COLORS`` so an
#: Ausgrid figure and a SolarEdge figure can sit side by side unchanged.
CATEGORY_COLORS = {
    "within_band": "#1565c0",
    "Q_near_conformant": "#2e7d32",
    "Q_significant_shortfall": "#ef6c00",
    "Q_inactive": "#c62828",
    "Q_adverse": "#7f1d1d",
    "Q_major_surplus": "#6a1b9a",
    "not_assessable": "#9e9e9e",
}
CATEGORY_LABELS = {
    "within_band": "Conforming (inside the band)",
    "Q_near_conformant": "Near-conformant (90-110% of required)",
    "Q_significant_shortfall": "Significant shortfall (10-90%)",
    "Q_inactive": "Inactive (no response)",
    "Q_adverse": "Adverse (wrong direction)",
    "Q_major_surplus": "Major surplus (>110%)",
    "not_assessable": "Not assessable (|P| < 20% of s_99)",
}
#: Draw order: the big conforming cloud first, so the rare failures land on top
#: instead of being buried under it.
CATEGORY_ORDER = (
    "not_assessable", "within_band", "Q_near_conformant",
    "Q_major_surplus", "Q_significant_shortfall", "Q_inactive", "Q_adverse",
)


def plot_voltvar_scatter(categories: pd.DataFrame, site_alias: str,
                         capacity_kva: float, period_label: str = "all data",
                         config=None, zoom=(230, 262, -60, 40),
                         max_points: int = 60_000, seed: int = 0,
                         figsize=(12, 8)):
    """
    Q-vs-V scatter for one site, every interval, coloured by its verdict.

    Ported from ``voltvar_voltwatt_lib.plot_vvar_month_scatter_ausgrid``, with one
    change: the colours come from ``site_interval_categories``, which runs the
    **same D9 SQL that produced the fleet numbers**, rather than a separate scalar
    classifier. A point coloured "adverse" here is one of the intervals counted as
    adverse in notebook 03 -- the explorer and the headline cannot drift apart.

    Where the daily plot shows *when* a site misbehaves, this shows *where on the
    curve*. The two failure shapes are visually distinct and mean different things:

    * a horizontal band of points near Q = 0 across the whole voltage range is an
      inverter with Volt-VAr **disabled** -- it never responds at any voltage;
    * points that track the required curve but sit consistently short of it are an
      inverter responding with the **wrong settings** or a smaller capability.

    Both produce similar non-conformance rates. Only this view separates them, and
    they call for different follow-up.

    ``capacity_kva`` is ``s_99``, an empirical p99 of apparent power, **not** a
    verified nameplate rating -- so every percentage on this axis is relative to a
    proxy. That is stamped on the axis label rather than left in a caption.
    """
    config = (config or se_params.CONFIG).validate()
    if categories.empty:
        print(f"No intervals for {site_alias}.")
        return None

    S = float(capacity_kva)
    tol = config.tolerance_fraction
    d = categories.copy()

    n_total = len(d)
    if n_total > max_points:
        # Deterministic subsample: a scatter of 100k points is unreadable and slow,
        # and the seed keeps the figure reproducible between runs.
        d = d.sample(max_points, random_state=seed)
    d["Q_pct"] = d.Q_kvar / S * 100.0

    v_grid = np.linspace(zoom[0], zoom[1], 600)
    q_req_pct = np.array([vvar_required_q(v, S) for v in v_grid]) / S * 100.0

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    vv = _A["VVAR"]
    ax.axvspan(vv["V2"], vv["V3"], color=C_GREY, alpha=0.09, lw=0, zorder=0,
               label=f"Deadband ({vv['V2']:.0f}-{vv['V3']:.0f} V)")

    counts = categories.category.value_counts()
    for cat in CATEGORY_ORDER:
        sub = d[d.category == cat]
        if sub.empty:
            continue
        n = int(counts.get(cat, 0))
        ax.scatter(sub.V, sub.Q_pct, s=5,
                   alpha=0.25 if cat in ("within_band", "not_assessable") else 0.55,
                   color=CATEGORY_COLORS[cat], lw=0, zorder=3,
                   label=f"{CATEGORY_LABELS[cat]} ({n:,}, {100 * n / n_total:.1f}%)")

    ax.fill_between(v_grid, q_req_pct - tol * 100, q_req_pct + tol * 100,
                    color="#f59e0b", alpha=0.18, lw=0, zorder=2,
                    label=f"Permitted band (+/-{tol:.0%} of s_99)")
    ax.plot(v_grid, q_req_pct, color="#f59e0b", lw=1.8, zorder=5,
            label="Required Q (AS/NZS 4777.2 Australia A)")
    ax.axhline(0, color="black", lw=0.6, zorder=1)

    for vx, lbl in [(vv["V1"], "V1"), (vv["V2"], "V2"), (vv["V3"], "V3"),
                    (vv["V4"], "V4"), (_A["VW"]["V1"], "V-Watt")]:
        if zoom[0] <= vx <= zoom[1]:
            ax.axvline(vx, color=C_GREY, lw=0.7, ls=":", zorder=1)
            ax.text(vx + 0.3, zoom[3] * 0.96, f"{lbl} {vx:.0f}", fontsize=6.5,
                    color=C_GREY, va="top")

    ax.set_xlim(zoom[0], zoom[1])
    ax.set_ylim(zoom[2], zoom[3])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
    ax.set_xlabel("Site voltage (V)")
    ax.set_ylabel("Reactive power (% of s_99 PROXY, not verified S_rated)\n"
                  "+ = supplying, - = absorbing")
    shown = (f"  ·  showing {len(d):,} of {n_total:,} intervals"
             if len(d) < n_total else f"  ·  {n_total:,} intervals")
    ax.set_title(
        f"{site_alias}  ·  {period_label}  ·  s_99 = {S:.2f} kVA{shown}\n"
        f"Volt-VAr response vs AS/NZS 4777.2:2020, coloured by the D9 verdict",
        fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.93, ncol=1)
    ax.grid(color="#ebebeb", lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    print(f"Category breakdown for {site_alias} ({n_total:,} intervals):")
    for cat in CATEGORY_ORDER:
        n = int(counts.get(cat, 0))
        if n:
            print(f"  {cat:<26} {n:>8,}  ({100 * n / n_total:5.1f}%)")
    return _done(fig)



#: Volt-Watt per-interval outcomes. Five, not two, because "P below the ceiling"
#: covers three genuinely different situations and collapsing them is what makes
#: a "conformant" verdict hard to interpret.
VOLTWATT_CATEGORIES = (
    "not_exposed",          # V <= 253 -- Volt-Watt imposes no limit
    "nonconformant",        # exposed, P above the ceiling
    "curtailed",            # exposed, below ceiling, counterfactual says sun WAS there
    "sun_limited",          # exposed, below ceiling, counterfactual says it was not
    "no_counterfactual",    # exposed, below ceiling, no prediction available
)
VOLTWATT_CATEGORY_COLORS = {
    "not_exposed": "#c9c9c9",
    "nonconformant": "#c62828",
    "curtailed": "#e8702a",
    "sun_limited": "#2a7f62",
    "no_counterfactual": "#9e9e9e",
}


def site_voltwatt_categories(con, site_alias: str, day: str | None = None,
                             config=None) -> pd.DataFrame:
    """
    Per-timestamp Volt-Watt classification, the analogue of
    ``site_interval_categories``.

    The five-way split matters because "conformant" hides three different states.
    An exposed interval below the ceiling can mean the inverter curtailed
    correctly (``curtailed``), that there was never enough sun to reach the
    ceiling so nothing was tested (``sun_limited``), or that the counterfactual
    has no prediction and the question is simply unanswerable
    (``no_counterfactual``).

    Only ``curtailed`` is positive evidence that Volt-Watt works. Only
    ``nonconformant`` is evidence that it does not. The other three are silence,
    and a site-level rate built on the first two alone can be resting on very few
    intervals -- ``excess_kW`` and ``available_kW`` are carried so that is visible.
    """
    frame = site_voltwatt_day(con, site_alias, day, config) if day else None
    if frame is None:
        raise ValueError("site_voltwatt_categories needs a day; "
                         "pass day='YYYY-MM-DD'")

    d = frame.copy()
    exposed = d.exposed.fillna(False).to_numpy(bool)
    over = (d.P_kW > d.P_ceiling_kW).fillna(False).to_numpy(bool)
    has_cf = d.uncurtailed_P.notna().to_numpy(bool)
    available = (d.uncurtailed_P > d.P_ceiling_kW).fillna(False).to_numpy(bool)

    category = np.select(
        [~exposed, exposed & over, exposed & ~over & has_cf & available,
         exposed & ~over & has_cf & ~available],
        ["not_exposed", "nonconformant", "curtailed", "sun_limited"],
        default="no_counterfactual",
    )
    d["category"] = category
    d["excess_kW"] = np.where(exposed & over,
                              (d.P_kW - d.P_ceiling_kW).fillna(0), 0.0).round(4)
    d["available_kW"] = np.where(category == "curtailed",
                                 (d.uncurtailed_P - d.P_kW).fillna(0), 0.0).round(4)
    d["headroom_kW"] = (d.P_ceiling_kW - d.P_kW).round(4)

    cols = ["ts_aest", "V", "P_kW", "P_ceiling_kW", "uncurtailed_P", "headroom_kW",
            "category", "excess_kW", "available_kW", "derating_active"]
    return d[cols]


def voltwatt_day_summary(categories: pd.DataFrame) -> pd.DataFrame:
    """Interval counts and energy per Volt-Watt category for one day."""
    h = C.INTERVAL_H
    rows = []
    n = len(categories)
    for cat in VOLTWATT_CATEGORIES:
        sub = categories[categories.category == cat]
        rows.append({
            "category": cat,
            "n_intervals": len(sub),
            "pct_of_day": round(100.0 * len(sub) / n, 1) if n else 0.0,
            "kWh_over_ceiling": round(float(sub.excess_kW.sum()) * h, 3),
            "kWh_available_not_taken": round(float(sub.available_kW.sum()) * h, 3),
        })
    return pd.DataFrame(rows)


def voltwatt_counterfactual_gaps(con, site_alias: str, day: str,
                                 config=None) -> pd.DataFrame:
    """
    Why is the counterfactual patchy? Attribute the missing predictions.

    ``uncurtailed_P`` is produced only where a per-(site, time-of-day-bin)
    regression exists AND the clear-sky references are positive::

        FROM se_structured s
        JOIN se_ghi_model m ON s.site_alias = m.site_alias AND s.tod_bin = m.tod_bin
        JOIN _gate g        ON s.site_alias = g.site_alias
        WHERE s.GHI_cs > 0 AND s.P_kw_norm_cs > 0

    and the model itself only exists for bins with at least ``MIN_TRAIN_POINTS``
    surviving training rows.

    So the gaps are per-TIME-OF-DAY-BIN, not per-interval: if a site's 11:25 bin
    never accumulated five usable training rows across the whole record, every
    11:25 in the year is missing a prediction. That is what makes the dashed line
    on the day plot patchy rather than merely truncated.

    Measured on one site (13,948 structured intervals), the training filters cut
    in this order::

        GHI_cs > 0                     12,591   (-10%)
        P_kw_norm_cs > 0.2              8,922   (-29%)  dawn/dusk reference too small
        GHI > 50                        8,922     (0%)
        P_kw_norm > 0.05                8,715    (-2%)
        P_kw_norm <= P_kw_norm_cs       5,328   (-39%)  <- the largest single cut
        V_mean <= 253                   4,556   (-14%)

    then 70% of those go to training and each bin needs 5. The dominant loss is
    the ``P_kw_norm <= P_kw_norm_cs`` quality gate, not -- as one might assume --
    the ``V_mean <= 253`` exclusion. The 253 V filter is real and does contribute
    (it is the reason you cannot learn from curtailed intervals), but attributing
    the patchiness to it alone would be wrong. Use
    ``counterfactual_training_attrition`` to get the equivalent breakdown for the
    site in front of you rather than assuming these proportions transfer.

    Coverage improves with record length: bins fill up, fewer fall below the
    minimum. A three-month store is far patchier than a twelve-month one.

    Returns one row per time-of-day bin present on the day, showing whether a
    model exists and how many training rows it had.
    """
    from solar_edge.lib import se_counterfactual as ctf

    config = (config or se_params.CONFIG).validate()
    if not C.store_path("se_ghi_model").exists():
        raise FileNotFoundError(
            "se_ghi_model not found — run notebook 04 section 5 first."
        )
    from solar_edge.lib import se_store

    se_store.register_store_views(con)

    return con.execute(
        f"""
        WITH day_bins AS (
            SELECT ts_aest,
                   CAST(date_trunc('minute', ts_aest)
                        - INTERVAL '1' MINUTE
                          * (minute(ts_aest) % {ctf.TIME_BIN_MIN}) AS TIME) AS tod_bin,
                   {contract.voltage_sql(config.voltage_aggregation, 'i')} AS V
            FROM se_interval i
            WHERE site_alias = '{site_alias}'
              AND CAST(ts_aest AS DATE) = DATE '{day}'
        )
        SELECT d.tod_bin,
               count(*)                                    AS intervals_on_day,
               round(avg(d.V), 1)                          AS mean_V,
               count(*) FILTER (WHERE d.V > {_A['VW']['V1']}) AS n_exposed,
               (m.site_alias IS NOT NULL)                  AS model_exists,
               any_value(m.n)                              AS model_train_points
        FROM day_bins d
        LEFT JOIN se_ghi_model m
               ON m.site_alias = '{site_alias}' AND m.tod_bin = d.tod_bin
        GROUP BY d.tod_bin, (m.site_alias IS NOT NULL)
        ORDER BY d.tod_bin
        """
    ).df()



def counterfactual_training_attrition(con, site_alias: str) -> pd.DataFrame:
    """
    Where a site's counterfactual training rows go, filter by filter.

    Answers "why is my uncurtailed_P patchy" with numbers instead of a guess. Each
    row is cumulative: the count surviving that filter AND every filter above it,
    in the order ``fit_ghi_model`` applies them.

    Read ``pct_lost_here`` -- the biggest number is the binding constraint for this
    site, and it is not the same one for every site. A site with a well-behaved
    clear-sky reference loses most rows to ``V_mean <= 253``; a site whose
    reference day was itself cloudy loses them to ``P_kw_norm <= P_kw_norm_cs``.
    """
    from solar_edge.lib import se_counterfactual as ctf
    from solar_edge.lib import se_store

    if not C.store_path("se_structured").exists():
        raise FileNotFoundError(
            "se_structured not found — run notebook 04 section 5 first.")
    se_store.register_store_views(con)

    v1 = _A["VW"]["V1"]
    steps = [
        ("all structured intervals", "TRUE"),
        ("GHI_cs > 0", "GHI_cs > 0"),
        ("P_kw_norm_cs > 0.2  (dawn/dusk cut)", "P_kw_norm_cs > 0.2"),
        ("GHI > 50", "GHI > 50"),
        ("P_kw_norm > 0.05", "P_kw_norm > 0.05"),
        ("P_kw_norm <= P_kw_norm_cs  (quality gate)", "P_kw_norm <= P_kw_norm_cs"),
        (f"V_mean <= {v1:.0f}  (exclude Volt-Watt zone)", f"V_mean <= {v1}"),
    ]
    cumulative, rows, prev = [], [], None
    for label, predicate in steps:
        cumulative.append(predicate)
        n = con.execute(
            f"SELECT count(*) FROM se_structured WHERE site_alias = '{site_alias}' "
            f"AND {' AND '.join(cumulative)}"
        ).fetchone()[0]
        rows.append({
            "filter": label,
            "surviving": int(n),
            "lost_here": 0 if prev is None else int(prev - n),
            "pct_lost_here": 0.0 if prev in (None, 0) else round(100 * (prev - n) / prev, 1),
        })
        prev = n

    frame = pd.DataFrame(rows)
    total = frame.surviving.iloc[0]
    frame["pct_of_original"] = (100 * frame.surviving / total).round(1) if total else 0.0
    print(f"{site_alias}: {frame.surviving.iloc[-1]:,} of {total:,} intervals "
          f"({frame.pct_of_original.iloc[-1]:.1f}%) are eligible to train on.")
    print(f"  Then 70% go to training, and each 5-minute time-of-day bin needs "
          f">= {ctf.MIN_TRAIN_POINTS} of them\n  to get a model at all — bins below "
          f"that produce no prediction, all year.")
    return frame


def site_category_calendar(con, site_alias: str, config=None, params=None,
                           normalise: bool = True) -> pd.DataFrame:
    """
    Daily counts per category for one site, for the whole year.

    Aggregated IN DUCKDB to (day x category) before anything reaches pandas. A
    site-year is ~100,000 intervals; this returns at most 365 x 7 = 2,555 rows, so
    the heatmap plots from a few kilobytes instead of scanning a hundred thousand
    points. That is the whole optimisation -- do not fetch the intervals and pivot
    them client-side.

    ``normalise=True`` returns each category as a % of that day's intervals, which
    is what makes days with different daylight lengths comparable.
    """
    from solar_edge.lib import se_conformance as cf

    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()

    frame = con.execute(
        f"""
        WITH scored AS ({cf.voltvar_interval_sql(config, params)}),
        tagged AS (
            SELECT CAST(ts_aest AS DATE) AS day_aest,
                   {_category_expr()}    AS category
            FROM scored WHERE site_alias = '{site_alias}'
        )
        SELECT day_aest, category, count(*) AS n_intervals
        FROM tagged GROUP BY 1, 2
        """
    ).df()

    if frame.empty:
        return frame

    wide = (frame.pivot(index="day_aest", columns="category", values="n_intervals")
            .reindex(columns=list(INTERVAL_CATEGORIES)).fillna(0))
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    if normalise:
        wide = 100 * wide.div(wide.sum(axis=1).replace(0, pd.NA), axis=0)
    return wide


def site_profile(con, site_alias: str, config=None) -> pd.Series:
    """One-line summary of a site: cohort, capacity, coverage, response strength."""
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    return con.execute(
        f"""
        SELECT s.site_alias, s.state, s.postcode,
               s.n_phases, s.is_three_phase,
               s.n_intervals, s.n_days_observed,
               s.has_night_generation_anomaly,
               round(c.s_99, 2) AS s_99, round(c.p_99, 2) AS p_99,
               round(s.pct_derating, 3) AS pct_derating,
               round((SELECT median(Q_kvar) FROM se_interval i
                      WHERE i.site_alias = s.site_alias AND {v} > 250
                        AND i.P_kW > 0.5), 4) AS med_Q_stored_above_250,
               round((SELECT median(Q_kvar) FROM se_interval i
                      WHERE i.site_alias = s.site_alias
                        AND {v} BETWEEN 225 AND 238 AND i.P_kW > 0.5), 4)
                   AS med_Q_stored_deadband
        FROM se_site s
        LEFT JOIN se_site_capacity c USING (site_alias)
        WHERE s.site_alias = '{site_alias}'
        """
    ).df().iloc[0]


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_site_voltvar_curve(curve: pd.DataFrame, site_alias: str,
                            profile: pd.Series | None = None, figsize=(11, 5)):
    """
    Median Q vs voltage for one site, both orientations, against the requirement.

    Read the MAGENTA required band first. Then ask which of the two measured
    traces sits on the same side of zero as it. That is the whole sign question,
    for this site, in one picture.
    """
    S = float(curve.s_99.dropna().iloc[0]) if curve.s_99.notna().any() else 5.0
    v = curve.v_bin.to_numpy()
    required = np.array([vvar_required_q(x, S) for x in v])
    tol = _A["TOL_FRAC"] * S

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    ax.axvspan(_A["VVAR"]["V2"], _A["VVAR"]["V3"], color="#8a8a8a", alpha=0.10, lw=0,
               label="Volt-VAr deadband (220-240 V)")
    ax.axvline(_A["VW"]["V1"], color="#e8702a", ls="--", lw=1.0, label="253 V (Volt-Watt starts)")
    ax.axhline(0, color="black", lw=0.8)

    ax.fill_between(v, required - tol, required + tol, color="#d946ef", alpha=0.22,
                    lw=0, label=f"Required Q +/-{_A['TOL_FRAC']:.0%} of s_99")
    ax.plot(v, required, color="#d946ef", lw=1.8, label="Required Q (AS/NZS 4777.2)")

    ax.plot(curve.v_bin, curve.Q_stored_kvar, color="#1565c0", lw=2.0, marker="o", ms=3,
            label=f"Q as STORED  ({_SIGN:+.0f} x raw / 1000)")
    ax.plot(curve.v_bin, curve.Q_flipped_kvar, color="#2e7d32", lw=1.6, ls="--",
            marker="s", ms=2.5, label=f"Q NEGATED  ({-_SIGN:+.0f} x raw / 1000)")

    ax.set_xlabel("Site voltage (V)")
    ax.set_ylabel("Median Q (kvar)   —   negative = absorbing (generator convention)")
    cohort = ""
    if profile is not None:
        cohort = (" | three-phase" if profile.is_three_phase else " | single-phase")
    ax.set_title(f"{site_alias}{cohort} | s_99 = {S:.2f} kVA\n"
                 "Which trace sits on the same side of zero as the requirement?",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0.92, loc="best")
    ax.grid(color="#ebebeb", lw=0.5)
    fig.tight_layout()
    return _done(fig)


def plot_category_heatmap(calendar: pd.DataFrame, site_alias: str,
                          normalise: bool = True, figsize=(15, 3.6)):
    """
    A site's whole year: dates across, conformance categories down.

    Built from ``site_category_calendar`` -- already aggregated to at most 365 x 7
    cells, so this is a ``pcolormesh`` over a few kilobytes rather than a scatter
    over a hundred thousand points.

    Reading it: a site that is genuinely fine shows a solid ``within_band`` row
    with everything below it empty. Non-conformance that appears as a vertical
    stripe is episodic -- a few bad days -- and is a different finding from a
    horizontal band, which is persistent behaviour. A rate alone cannot tell the
    two apart, which is exactly why this plot is worth the space.
    """
    import matplotlib.colors as mcolors

    if calendar.empty:
        raise ValueError(f"no data for {site_alias}")

    # ALL seven rows are drawn, even empty ones. An empty `Q_adverse` row is a
    # finding in its own right, and keeping the axis fixed lets two sites be
    # compared by eye without re-reading the labels.
    keep = [c for c in INTERVAL_CATEGORIES if c in calendar.columns]
    data = calendar[keep].T.astype(float).fillna(0)

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    # shading="nearest": X and Y are cell CENTRES matching the data dimensions.
    # "flat" would need edge arrays one longer than the data on each axis.
    mesh = ax.pcolormesh(
        calendar.index, np.arange(len(keep)), data.to_numpy(),
        cmap="YlOrRd", shading="nearest",
        norm=mcolors.Normalize(vmin=0, vmax=100 if normalise else data.to_numpy().max()),
    )
    ax.set_yticks(np.arange(len(keep)))
    ax.set_yticklabels([k.replace("Q_", "").replace("_", " ") for k in keep], fontsize=8.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_xlabel("Date (AEST)")

    total_days = len(calendar)
    bad = [c for c in ("Q_adverse", "Q_inactive", "Q_significant_shortfall")
           if c in calendar.columns]
    reduced = calendar[bad].sum(axis=1) if bad else pd.Series(0.0, index=calendar.index)
    n_bad_days = int((reduced > 0).sum())
    ax.set_title(
        f"{site_alias} — conformance category by day  |  {total_days} days  |  "
        f"{n_bad_days} day(s) with any reduced non-conformance",
        fontsize=10.5, fontweight="bold")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label("% of that day's intervals" if normalise else "intervals", fontsize=8.5)
    fig.tight_layout()
    return _done(fig)


def plot_operational(df_day, site_alias, capacity_kva, zoom_date,
                     orientation: str = "stored", figsize=(13, 13.5)):
    """
    Five-panel daily plot: voltage, active power, Volt-Watt non-conformance,
    reactive power, Volt-VAr non-conformance.

    Ported from ``voltvar_voltwatt_lib.plot_operational_ausgrid``.

    ``orientation`` selects which reactive trace is plotted and is stamped on the
    figure title:

      ``"stored"``  the store as-is -- what D9/D10 scored
      ``"raw"``     raw/1000, exactly as SolarEdge delivered it

    Since 13 Aug ``REACTIVE_POWER_SIGN = +1``, so the two are IDENTICAL. The
    distinction is retained because the orientation is now an analysis parameter
    (``SEAnalysisConfig.reactive_orientation``) and can be swept.

    ``capacity_kva`` is ``s_99``, an EMPIRICAL p99 of apparent power, not a
    verified rating. Every percentage on this figure is relative to that proxy.
    """
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {list(ORIENTATIONS)}")

    vw, vv, tol = _A["VW"], _A["VVAR"], _A["TOL_FRAC"]
    S = float(capacity_kva)

    t = pd.to_datetime(df_day["ts_aest"])
    V = df_day["V"]
    P = df_day["P_kW"]
    Q = df_day["Q_stored_kvar"] if orientation == "stored" else df_day["Q_raw_kvar"]

    def _vw_ceil_pct(v):
        if v < vw["V1"]:
            return 100.0
        if v > vw["V2"]:
            return vw["P2"] * 100.0
        return (1.0 - vw["P2"]) / (vw["V1"] - vw["V2"]) * (v - vw["V2"]) * 100.0 + vw["P2"] * 100.0

    Q_req = V.map(lambda v: vvar_required_q(v, S))
    Q_req_max, Q_req_min = Q_req + tol * S, Q_req - tol * S

    P_pct = (P / S) * 100.0
    P_ceil_pct = V.map(_vw_ceil_pct) + tol * 100.0
    P_nc_pct = np.where(V.values > vw["V1"], np.maximum(0.0, P_pct.values - P_ceil_pct.values), 0.0)
    Q_pct = (Q / S) * 100.0
    Q_req_pct, Q_req_max_pct, Q_req_min_pct = (Q_req / S) * 100, (Q_req_max / S) * 100, (Q_req_min / S) * 100

    nc_signed_pct = np.where(
        Q_pct.values > Q_req_max_pct.values, Q_pct.values - Q_req_max_pct.values,
        np.where(Q_pct.values < Q_req_min_pct.values, -(Q_req_min_pct.values - Q_pct.values), 0.0))

    capability_assessable = (P.abs() >= _A["QCAP"]["P_MIN"] * S).to_numpy()
    nc_kvar = np.maximum(0, Q - Q_req_max) + np.maximum(0, Q_req_min - Q)
    nc_kvarh = float(nc_kvar[capability_assessable].sum() * _A["INTERVAL_H"])

    vvar_active, vw_active = V.values > vv["V3"], V.values > vw["V1"]
    # Palette matches ausgrid_analysis/notebooks/voltvar_voltwatt_lib.py exactly,
    # so the two studies' figures can sit side by side without recolouring.
    Cv, Cp, Cc, Cq, Cn = "#b45309", "#2e7d32", "#1a1a1a", "#1565c0", "#c62828"
    C_REF, C_VVAR, C_VW, C_GRID = "#f59e0b", "#7c3aed", "#4709b2", "#ebebeb"
    C_NA = "#9e9e9e"   # not-assessable grey

    fig, axes = plt.subplots(5, 1, figsize=figsize, dpi=130, sharex=True,
                             gridspec_kw={"height_ratios": [1.8, 2.2, 1.0, 2.2, 1.0]})
    fig.subplots_adjust(hspace=0.05, left=0.10, right=0.90, top=0.92, bottom=0.05)
    ax_v, ax_p, ax_pnc, ax_q, ax_qnc = axes

    for ax in axes:
        ax.fill_between(t, 0, 1, where=vvar_active, transform=ax.get_xaxis_transform(),
                        color=C_VVAR, alpha=0.07, lw=0, zorder=0)
        ax.fill_between(t, 0, 1, where=vw_active, transform=ax.get_xaxis_transform(),
                        color=C_VW, alpha=0.08, lw=0, zorder=0)
        ax.grid(color=C_GRID, lw=0.5)
        ax.set_facecolor("white")

    # --- voltage -----------------------------------------------------------
    ax_v.plot(t, V, color=Cv, lw=1.3, zorder=4)
    for vref, ls, al, lbl in [
        (vv["V3"], ":", 0.55, "240 V (V-VAr deadband hi)"),
        (vw["V1"], "--", 0.85, "253 V (V-Watt start)"),
        (vv["V4"], "--", 0.85, "258 V (V-VAr max absorb)"),
        (vw["V2"], "--", 0.85, "260 V (V-Watt full curtail)"),
    ]:
        ax_v.axhline(vref, color=Cv, lw=0.8, ls=ls, alpha=al, zorder=3)
        ax_v.text(t.iloc[-1], vref + 0.25, lbl, va="bottom", ha="right",
                  fontsize=6, color=Cv, alpha=min(al + 0.15, 1.0))
    ax_v.set_ylabel("Voltage (V)", fontsize=8.5, color=Cv)
    ax_v.tick_params(axis="y", colors=Cv, labelsize=8)
    ax_v.set_ylim(min(228, V.min() - 2), max(262, V.max() + 2))
    ax_v.legend(handles=[Patch(color=C_VVAR, alpha=0.30, label=f"V > {vv['V3']:.0f} V — V-VAr required"),
                         Patch(color=C_VW, alpha=0.30, label=f"V >= {vw['V1']:.0f} V — V-Watt active")],
                fontsize=7, loc="upper left", framealpha=0.92)

    # --- active power ------------------------------------------------------
    ax_p.plot(t, P_ceil_pct, color=Cc, lw=1.6, zorder=4, label=f"V-Watt ceiling (+{tol*100:.0f}% tol)")
    ax_p.plot(t, P_pct, color=Cp, lw=1.3, zorder=5, label="Measured P (% of s_99)")
    ax_p.axhline(100, color=Cp, lw=0.6, ls=":", alpha=0.45)
    ax_p.set_ylabel("Active power\n(% of s_99)", fontsize=8.5, color=Cp)
    ax_p.tick_params(axis="y", colors=Cp, labelsize=8)
    ax_p.set_ylim(-10, 115)
    ax_p.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax_pkw = ax_p.twinx()
    ax_pkw.set_ylim(-0.10 * S, 1.15 * S)
    ax_pkw.set_ylabel("Active power (kW)", fontsize=8.5, color=Cp)
    ax_pkw.tick_params(axis="y", colors=Cp, labelsize=8)

    # --- Volt-Watt non-conformance ----------------------------------------
    ax_pnc.bar(t, P_nc_pct, width=pd.Timedelta(minutes=4.5), color=Cn, alpha=0.80, zorder=4,
               label="V-Watt NC (pp above ceiling)")
    ax_pnc.axhline(0, color="k", lw=0.5)
    ax_pnc.set_ylabel("V-W NC\n(pp)", fontsize=8.5, color=Cn)
    ax_pnc.tick_params(axis="y", colors=Cn, labelsize=8)
    ax_pnc.set_ylim(0, max(P_nc_pct.max() * 1.15, 2.0))
    ax_pnc.legend(fontsize=7, loc="upper left", framealpha=0.9)

    # --- reactive power ----------------------------------------------------
    ax_q.fill_between(t, Q_req_min_pct, Q_req_max_pct, color=C_REF, alpha=0.25, lw=0, zorder=1)
    ax_q.plot(t, Q_req_min_pct, color=C_REF, lw=0.8, ls="--", zorder=2)
    ax_q.plot(t, Q_req_max_pct, color=C_REF, lw=0.8, ls="--", zorder=2)
    ax_q.plot(t, Q_req_pct, color=C_REF, lw=1.0, ls="-", alpha=0.6, zorder=2)
    ax_q.plot(t, Q_pct, color=Cq, lw=1.4, zorder=4)
    ax_q.axhline(0, color="k", lw=0.5, zorder=3)
    ax_q.set_ylabel("Reactive power\n(% of s_99, + supply / - absorb)", fontsize=8.5, color=Cq)
    ax_q.tick_params(axis="y", colors=Cq, labelsize=8)
    _qlim = max(abs(vv["Q1"]), abs(vv["Q4"])) * 100 + 10
    ax_q.set_ylim(-_qlim, _qlim)
    ax_q.legend(handles=[
        Patch(color=C_REF, alpha=0.40,
              label=f"Required Q band (+/-{tol*100:.0f}% of s_99)"),
        plt.Line2D([0], [0], color=Cq, lw=1.4,
                   label=f"Measured Q — {orientation.upper()}"),
    ], fontsize=7, loc="upper left", framealpha=0.9)
    ax_qkvar = ax_q.twinx()
    ax_qkvar.set_ylim(-_qlim / 100 * S, _qlim / 100 * S)
    ax_qkvar.set_ylabel("Reactive power (kvar)", fontsize=8.5, color=Cq)
    ax_qkvar.tick_params(axis="y", colors=Cq, labelsize=8)

    # --- Volt-VAr non-conformance -----------------------------------------
    # Non-assessable intervals are drawn GREY and are NOT in the kvarh tally.
    # Below 0.2 x S_rated, Figure 2.1 sets no quantified minimum capability, so a
    # deviation there is not evidence of non-conformance -- it is an interval the
    # standard declines to judge. Colouring them the same red as real breaches
    # would invite the reader to count them.
    qnc_colors = np.where(capability_assessable, Cn, C_NA)
    ax_qnc.bar(t, nc_signed_pct, width=pd.Timedelta(minutes=4.5), color=qnc_colors,
               alpha=0.80, align="center", zorder=4)
    ax_qnc.axhline(0, color="k", lw=0.5)
    ax_qnc.set_ylabel("V-VAr NC\n(pp, signed)", fontsize=8.5, color=Cn)
    ax_qnc.tick_params(axis="y", colors=Cn, labelsize=8)
    ax_qnc.legend(handles=[
        plt.Line2D([0], [0], color=Cn, lw=4, alpha=0.80,
                   label="V-VAr NC (pp outside permitted band, signed)"),
        plt.Line2D([0], [0], color=C_NA, lw=4, alpha=0.80,
                   label=f"Not assessable (|P| < {_A['QCAP']['P_MIN']*100:.0f}% of s_99)"),
    ], fontsize=7, loc="upper left", framealpha=0.9)
    n_assessable = int(capability_assessable.sum())
    ax_qnc.text(0.99, 0.95,
                f"Daily V-VAr NC (assessable only): {nc_kvarh:.3f} kvarh"
                f"   ({nc_kvarh / S:.4f} kvarh/kVA)"
                f"   |  {n_assessable}/{len(t)} intervals assessable",
                transform=ax_qnc.transAxes, ha="right", va="top", fontsize=7.5,
                color=Cn,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=Cn, alpha=0.85, lw=0.7))
    ax_qnc.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_qnc.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_qnc.set_xlabel("Time (AEST, fixed UTC+10)")

    fig.suptitle(
        f"{site_alias}  |  {zoom_date}  |  s_99 = {S:.2f} kVA (empirical proxy, NOT nameplate)\n"
        f"REACTIVE SIGN: {ORIENTATIONS[orientation]}\n"
        f"V-VAr non-conformance this day: {nc_kvarh:.2f} kvarh",
        fontsize=10, fontweight="bold", y=0.985,
    )
    return _done(fig)
