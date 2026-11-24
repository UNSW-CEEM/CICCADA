"""
Family A: site-day time series.

    A.0  plot_operational_day  Volt-VAr + Volt-Watt in one figure
    A.1  plot_voltvar_day      V -> P -> Q vs permitted band -> Volt-VAr NC
    A.2  plot_voltwatt_day     V -> P vs ceiling -> Volt-Watt NC
    A.3  plot_protective_day   TO BE DONE (sustained operation, anti-islanding)

Input contract (any data source; see sources.py)
-------------------------------------------------
One tidy DataFrame for ONE AEST day:

    t_aest     naive datetime, AEST (fixed UTC+10)
    V          volts
    P_kw       active power, generation positive
    Q_kvar     reactive power, generator convention (+ supplying, - absorbing)
    phase      optional, 'L1'/'L2'/'L3'. Absent or a single value = single-phase
    P_pot_kw   optional, PV potential / counterfactual (site total; A.2 only)

plus `rating_kva`, the rated apparent power of the whole site (or the documented
proxy, e.g. ac_capacity_kw). Every figure function takes (df, info) with the
loader's info dict, or (df, rating_kva, meta).

Single-phase: one line per quantity, in the quantity colour.
Three-phase:  one thin line per phase (palette.PHASE) plus one thick line in the
              quantity colour.
              V:    per-phase volts; thick DASHED line = average of the phases
                    (the AS/NZS 4777.2 Cl. 3.3.2.1 reference voltage).
              P, Q: per-phase lines and the thick SUM of the phases, all as %
                    of the site rating (each phase sits near a third of the sum).

Volt-VAr is assessed on the thick line (mean V, total P, total Q against the
total rating) with bms_sa_review/shared (as4777_curves zones, band and
categories; vvar_metrics for the day's metrics) -- the same rules as the
Stage 2 and OEM pipelines:
  - |P| < 20% S_rated : no minimum; fails only on wrong direction or going
                        beyond the curve, each by more than tol (wide band)
  - 20-60%            : accepted from min(curve, 44%) up to the curve
  - 60-80%            : from min(curve, 0.75 P) up to the curve
  - > 80%             : full curve (reactive power priority)
  - 220-240 V         : voltage deadband, Q within +/-tol
Zones: A response required | B sufficient power, voltage deadband |
C low power, voltage exposed | D low power, voltage deadband.
`metric=` picks which intervals count (the band never changes):
  remote_conformance_rate (default)  zone A + failing zone C
  nonconformance_frequency           every valid interval
  legacy_conformance_rate            P >= 20% at any voltage (pre-rework headline)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# shared AS/NZS 4777.2 curves live in bms_sa_review/shared ("from shared.x import ...")
_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "bms_sa_review"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.as4777_curves import (  # noqa: E402
    ZONE_LABELS,
    ZONES,
    vvar_band,
    vvar_required_q_np,
    vw_max_p,
)
from shared.ciccada_config import AS4777, CONFORMANCE  # noqa: E402
from shared.vvar_metrics import (  # noqa: E402
    LEGACY_FAILURES,
    conformance_metrics,
    failure_categories,
    metric_columns,
    score_intervals,
    zone_counts,
)

from . import palette as C  # noqa: E402

VV, VW, TOL, QCAP = AS4777["VVAR"], AS4777["VW"], AS4777["TOL_FRAC"], AS4777["QCAP"]

# Fixed y ranges so every site-day figure is directly comparable.
V_RANGE = (215.0, 265.0)  # widened automatically if the data go outside
P_RANGE = (-5.0, 115.0)  # % of rating
Q_RANGE = (-70.0, 55.0)  # % of rating (Q4 = 60% absorbing, Q1 = 44% supplying)

# Set-point lines drawn on the voltage panel, per figure.
_V_LINES = {
    "voltvar": [
        (VV["V1"], "V1 207 V"),
        (VV["V2"], "V2 220 V"),
        (VV["V3"], "V3 240 V"),
        (VV["V4"], "V4 258 V"),
    ],
    "voltwatt": [
        (VW["V1"], "Volt-Watt start 253 V"),
        (VW["V2"], "Volt-Watt 20% floor 260 V"),
    ],
    "operational": [
        (VV["V3"], "240 V"),
        (VW["V1"], "253 V"),
        (VV["V4"], "258 V"),
        (VW["V2"], "260 V"),
    ],
}


# =============================================================================
# 1. Data preparation
# =============================================================================
@dataclass
class SiteDay:
    """One site-day in plotting units. Built by `prepare`."""

    t: pd.DatetimeIndex
    rating_kva: float
    phases: list[str]
    v_phase: pd.DataFrame  # volts, one column per phase
    p_phase_pct: pd.DataFrame  # each phase, % of the site rating
    q_phase_pct: pd.DataFrame
    v: pd.Series  # mean voltage
    p_kw: pd.Series  # site total
    q_kvar: pd.Series
    p_pct: pd.Series  # site total, % of rating
    q_pct: pd.Series
    p_pot_pct: pd.Series | None = None
    interval_h: float = 5 / 60
    meta: dict = field(default_factory=dict)

    @property
    def three_phase(self) -> bool:
        return len(self.phases) > 1


def prepare(df: pd.DataFrame, rating_kva: float, meta: dict | None = None) -> SiteDay:
    """Validate the input contract and convert to plotting units."""
    missing = {"t_aest", "V", "P_kw", "Q_kvar"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not rating_kva or rating_kva <= 0:
        raise ValueError("rating_kva must be a positive number")

    d = df.copy()
    d["t_aest"] = pd.to_datetime(d["t_aest"])
    if "phase" not in d.columns or d["phase"].nunique(dropna=True) <= 1:
        d["phase"] = "L1"
    phases = sorted(d["phase"].dropna().unique())
    n = len(phases)

    wide = d.pivot_table(
        index="t_aest", columns="phase", values=["V", "P_kw", "Q_kvar"], aggfunc="mean"
    )
    t = wide.index
    v_ph, p_ph, q_ph = wide["V"][phases], wide["P_kw"][phases], wide["Q_kvar"][phases]

    p_tot = p_ph.sum(axis=1, min_count=1)
    q_tot = q_ph.sum(axis=1, min_count=1)
    step = pd.Series(t).diff().median()
    interval_h = step / pd.Timedelta(hours=1) if pd.notna(step) else 5 / 60

    pot = None
    if "P_pot_kw" in d.columns and d["P_pot_kw"].notna().any():
        pot = d.groupby("t_aest")["P_pot_kw"].first().reindex(t) / rating_kva * 100

    return SiteDay(
        t=t,
        rating_kva=float(rating_kva),
        phases=phases,
        v_phase=v_ph,
        p_phase_pct=p_ph / rating_kva * 100,
        q_phase_pct=q_ph / rating_kva * 100,
        v=v_ph.mean(axis=1),
        p_kw=p_tot,
        q_kvar=q_tot,
        p_pct=p_tot / rating_kva * 100,
        q_pct=q_tot / rating_kva * 100,
        p_pot_pct=pot,
        interval_h=interval_h,
        meta=meta or {},
    )


# =============================================================================
# 2. Assessment (shared rules: bms_sa_review/shared)
# =============================================================================
#: Display names of the three metrics (explicit, as in ciccada_config.CONFORMANCE).
METRIC_LABELS = {
    "nonconformance_frequency": "Non-conformance frequency",
    "remote_conformance_rate": "Remote conformance rate",
    "legacy_conformance_rate": "Legacy conformance rate",
}
#: What each metric leaves out (shaded "not counted" in the Q panel).
NOT_COUNTED_REASON = {
    "nonconformance_frequency": None,
    "remote_conformance_rate": "voltage deadband (zones B, D) and low-power passes (zone C)",
    "legacy_conformance_rate": "low power, P < 20% (zones C, D)",
}


def _options(metric=None, tol=None, failure_definition=None):
    metric = metric or CONFORMANCE["METRIC_DEFAULT"]
    if metric not in METRIC_LABELS:
        raise ValueError(f"metric must be one of {sorted(METRIC_LABELS)}, got {metric!r}")
    tol = CONFORMANCE["TOL_DEFAULT"] if tol is None else tol
    fdef = failure_definition or CONFORMANCE["FAILURE_DEFINITION_DEFAULT"]
    return metric, tol, fdef


def assess_voltvar(
    day: SiteDay, metric: str | None = None, tol: float | None = None,
    failure_definition: str | None = None,
) -> pd.DataFrame:
    """
    Per interval, with the shared rules (bms_sa_review/shared): required Q, the
    accepted band, zone, category, the legacy category, whether the interval is
    counted by `metric`, whether it is a counted failure, and the distance
    outside the band (nc_pct). Q values in % of rating. The day's metric values
    (all three, from shared.vvar_metrics) are in .attrs["metrics"].

    metric: 'remote_conformance_rate' (default), 'nonconformance_frequency' or
    'legacy_conformance_rate'. The band drawn is the same for every metric; the
    metric only decides which intervals count.
    """
    metric, tol, fdef = _options(metric, tol, failure_definition)
    S = day.rating_kva
    frame = pd.DataFrame(
        {
            "V": day.v.to_numpy(),
            "P_kW": day.p_kw.to_numpy(),
            "Q_kvar": day.q_kvar.to_numpy(),
            "s_rated": S,
            "day": str(day.meta.get("date", "day")),
            "site": "site",
        },
        index=day.t,
    )
    sc = score_intervals(frame, tol=tol)
    q_lo, q_hi = vvar_band(sc.V, sc.P_kW, S, tol=tol)
    a = pd.DataFrame(index=day.t)
    a["q_req"] = vvar_required_q_np(sc.V, S) / S * 100
    a["q_min_final"] = q_lo / S * 100
    a["q_max_final"] = q_hi / S * 100
    a["q"] = sc.Q_kvar / S * 100
    for c in ("valid", "zone", "category", "legacy_category", "legacy_assessable"):
        a[c] = sc[c].to_numpy()
    new_fail = a.category.isin(failure_categories(fdef)).to_numpy()
    legacy_fail = a.legacy_category.isin(LEGACY_FAILURES).to_numpy()
    valid = a.valid.to_numpy(bool)
    zone = a.zone.to_numpy(object)
    if metric == "nonconformance_frequency":
        counted, fail = valid, new_fail
    elif metric == "remote_conformance_rate":
        counted = valid & ((zone == "A") | ((zone == "C") & new_fail))
        fail = new_fail
    else:
        counted, fail = valid & a.legacy_assessable.to_numpy(bool), legacy_fail
    a["counted"] = counted
    a["failure"] = counted & fail
    above, below = a.q > a.q_max_final, a.q < a.q_min_final
    a["nc_pct"] = np.where(above, a.q - a.q_max_final, np.where(below, a.q - a.q_min_final, 0.0))
    a["nc_kvar"] = a.nc_pct.abs() / 100 * S
    counts = zone_counts(sc, by=["site"], day_col="day")
    a.attrs.update(
        metric=metric,
        tol=tol,
        failure_definition=fdef,
        metrics=conformance_metrics(counts, failure_definition=fdef).iloc[0],
    )
    return a


def metric_summary(vv: pd.DataFrame) -> str:
    """One line for the selected metric, one for voltage exposure and zone counts."""
    m, metric = vv.attrs["metrics"], vv.attrs["metric"]
    pct = lambda x: "–" if pd.isna(x) else f"{100 * x:.1f}%"
    if metric == "nonconformance_frequency":
        fails = int(sum(m[f"fail_{z}"] for z in ZONES))
        line = (
            f"{METRIC_LABELS[metric]} {pct(m.nonconformance_frequency)} "
            f"({fails} of {int(m.n_valid)} valid intervals fail; by zone "
            + "  ".join(f"{z} {int(m[f'fail_{z}'])}" for z in ZONES)
            + ")"
        )
    elif metric == "remote_conformance_rate":
        if not m.remote_conformance_rate_assessable:
            line = (
                f"{METRIC_LABELS[metric]}: not assessable "
                f"({int(m.n_A)} zone A intervals on {int(m.n_days_A)} day(s); "
                f"minimum {int(m.min_zone_a_intervals)} on {int(m.min_zone_a_days)})"
            )
        else:
            passes = int(m.remote_denominator - m.fail_A - m.fail_C)
            line = (
                f"{METRIC_LABELS[metric]} {pct(m.remote_conformance_rate)} "
                f"({passes} of {int(m.remote_denominator)} pass: zone A {int(m.n_A)} "
                f"+ failing zone C {int(m.fail_C)})"
            )
    else:
        line = (
            f"{METRIC_LABELS[metric]} {pct(m.legacy_conformance_rate)} "
            f"({int(m.n_legacy_total - m.n_legacy_fail)} of {int(m.n_legacy_total)} "
            f"intervals with P ≥ 20% pass)"
        )
    verdict = m[metric_columns(metric)["verdict"]]
    line2 = (
        f"site tag (10% rule): {verdict}  ·  voltage exposure {pct(m.voltage_exposure)} "
        f"(zone A {int(m.n_A)} of {int(m.n_valid)})  ·  tol ±{100 * vv.attrs['tol']:.0f}%  ·  "
        + (
            "failures: legacy (adverse + inactive + significant shortfall)"
            if metric == "legacy_conformance_rate"
            else f"failures: {vv.attrs['failure_definition']}"
        )
    )
    return line + "\n" + line2



def assess_voltwatt(day: SiteDay) -> pd.DataFrame:
    """
    Volt-Watt curve, ceiling (= curve + tolerance), exposure (V > 253 V) and NC above the
    ceiling, in % of rating. Volt-Watt is a cap: any P at or below the ceiling is permitted,
    so the "band" drawn is the tolerance between the curve and the ceiling.
    """
    S = day.rating_kva
    curve = (
        pd.Series(
            [vw_max_p(v, S) if pd.notna(v) else np.nan for v in day.v], index=day.t
        )
        / S
        * 100
    )
    ceil = curve + TOL * 100
    exposed = (day.v > VW["V1"]).fillna(False)
    nc = np.where(exposed, np.maximum(0.0, day.p_pct - ceil), 0.0)
    a = pd.DataFrame(
        {"curve": curve, "ceiling": ceil, "exposed": exposed, "nc_pct": nc}, index=day.t
    )
    if day.p_pot_pct is not None:
        a["curtailed_pct"] = np.where(
            exposed, np.maximum(0.0, day.p_pot_pct - day.p_pct), 0.0
        )
    return a


# =============================================================================
# 3. Shared drawing helpers
# =============================================================================
def _shade(ax, t, where, color, alpha, hatch=None, zorder=0):
    """Vertical bands across the full panel height wherever `where` is True."""
    ax.fill_between(
        t,
        0,
        1,
        where=np.asarray(where, bool),
        transform=ax.get_xaxis_transform(),
        step="mid",
        facecolor=color,
        alpha=alpha,
        hatch=hatch,
        edgecolor=C.STATUS["not_assessable"],
        linewidth=0,
        zorder=zorder,
    )


def _night(day: SiteDay) -> np.ndarray:
    """Before the first and after the last interval with P above 1% of rating."""
    gen = (day.p_pct > 1).to_numpy()
    out = np.ones(len(gen), bool)
    if gen.any():
        i0, i1 = np.argmax(gen), len(gen) - np.argmax(gen[::-1])
        out[i0:i1] = False
    return out


def _lines(
    ax,
    day: SiteDay,
    phase_frame: pd.DataFrame,
    total: pd.Series,
    color: str,
    total_label: str,
    ls: str = "-",
):
    """Per-phase thin lines (3-phase only) plus the thick quantity line (average or sum)."""
    if day.three_phase:
        for ph in day.phases:
            ax.plot(
                day.t,
                phase_frame[ph],
                color=C.PHASE.get(ph, C.OTHER),
                lw=0.9,
                alpha=0.9,
                label=ph,
                zorder=3,
            )
    ax.plot(
        day.t,
        total,
        color=color,
        lw=1.9 if day.three_phase else 1.4,
        ls=ls if day.three_phase else "-",
        label=total_label,
        zorder=4,
    )


def _pct_axis(ax, day: SiteDay, unit: str, label: str):
    """% of rating on the left, physical unit on the right (pure relabel of the same scale)."""
    S = day.rating_kva
    sec = ax.secondary_yaxis(
        "right", functions=(lambda x: x / 100 * S, lambda y: y / S * 100)
    )
    sec.set_ylabel(f"{label} ({unit})")


def _bar_width(day: SiteDay):
    return pd.Timedelta(hours=day.interval_h) * 0.9


def _time_axis(ax):
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 25, 3)))
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time (AEST, UTC+10)")


def _legend(ax, handles=None, **kw):
    kw.setdefault("loc", "upper left")
    kw.setdefault("ncol", 3)
    ax.legend(handles=handles, **kw) if handles else ax.legend(**kw)


# =============================================================================
# 4. Panels (each draws on one axis; figures below compose them)
# =============================================================================
def panel_voltage(ax, day: SiteDay, lines: str = "operational"):
    _shade(ax, day.t, _night(day), C.REFERENCE["night"], 1.0)
    _lines(
        ax,
        day,
        day.v_phase,
        day.v,
        C.QUANTITY["V"],
        "Average V" if day.three_phase else "V",
        ls=(0, (5, 2)),
    )
    lo = min(V_RANGE[0], np.nanmin(day.v_phase.to_numpy()) - 2)
    hi = max(V_RANGE[1], np.nanmax(day.v_phase.to_numpy()) + 2)
    ax.set_ylim(lo, hi)
    for v, lab in _V_LINES[lines]:
        if not lo < v < hi:
            continue
        ax.axhline(v, color=C.REFERENCE["threshold"], lw=0.8, ls=":", zorder=2)
        ax.annotate(
            lab,
            (1.0, v),
            xycoords=("axes fraction", "data"),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=C.REFERENCE["threshold"],
        )
    ax.set_ylabel("Voltage (V)")
    _legend(ax, ncol=4)


def panel_active_power(
    ax, day: SiteDay, vw: pd.DataFrame | None = None, gate: bool = False
):
    _shade(ax, day.t, _night(day), C.REFERENCE["night"], 1.0)
    handles = []
    if vw is not None:
        _shade(ax, day.t, vw.exposed, C.QUANTITY["V"], 0.08)
        handles.append(
            Patch(
                facecolor=C.QUANTITY["V"],
                alpha=0.25,
                label=f"V > {VW['V1']:.0f} V (Volt-Watt active)",
            )
        )
        ax.fill_between(
            day.t,
            vw.curve,
            vw.ceiling,
            color=C.REFERENCE["band"],
            alpha=C.BAND_ALPHA,
            lw=0,
            zorder=1,
        )
        ax.plot(
            day.t, vw.ceiling, color=C.REFERENCE["required"], lw=0.9, ls="--", zorder=5
        )
        ax.plot(
            day.t,
            vw.curve,
            color=C.REFERENCE["required"],
            lw=0.8,
            ls=":",
            alpha=0.6,
            zorder=5,
        )
        handles += [
            Patch(
                facecolor=C.REFERENCE["band"],
                alpha=0.25,
                edgecolor=C.REFERENCE["required"],
                ls="--",
                label=f"Permitted: P ≤ ceiling (curve + {TOL*100:.0f}% tol.)",
            ),
            Line2D(
                [],
                [],
                color=C.REFERENCE["required"],
                ls=":",
                alpha=0.6,
                label="Volt-Watt curve",
            ),
        ]
        if day.p_pot_pct is not None:
            ax.plot(
                day.t,
                day.p_pot_pct,
                color=C.QUANTITY["potential"],
                lw=1.2,
                ls="--",
                zorder=3,
                label="PV potential",
            )
            ax.fill_between(
                day.t,
                day.p_pct,
                day.p_pot_pct,
                where=vw.curtailed_pct > 0,
                step="mid",
                color=C.QUANTITY["potential"],
                alpha=0.35,
                lw=0,
                zorder=2,
            )
            handles.append(
                Patch(
                    facecolor=C.QUANTITY["potential"],
                    alpha=0.35,
                    label="Curtailed while exposed",
                )
            )
    if gate:
        ax.axhline(
            QCAP["P_MIN"] * 100,
            color=C.REFERENCE["threshold"],
            lw=0.9,
            ls=":",
            zorder=2,
            label=f"{QCAP['P_MIN']*100:.0f}% gate (Fig. 2.1)",
        )
    ax.axhline(100, color=C.QUANTITY["S"], lw=0.7, zorder=2, label="100% rating")
    _lines(
        ax,
        day,
        day.p_phase_pct,
        day.p_pct,
        C.QUANTITY["P"],
        "Sum of phases" if day.three_phase else "P",
    )
    ax.set_ylim(*P_RANGE)
    ax.set_ylabel("Active power\n(% of rating)")
    _pct_axis(ax, day, "kW", "P")
    h, _ = ax.get_legend_handles_labels()
    _legend(ax, handles=h + handles)


def _zone_strip(ax, day: SiteDay, vv: pd.DataFrame):
    """Thin strip along the top of the panel: the zone of every interval."""
    for z in ZONES:
        ax.fill_between(
            day.t, 0.965, 1.0, where=(vv.zone == z).to_numpy(), step="mid",
            transform=ax.get_xaxis_transform(), facecolor=C.ZONE[z], lw=0, zorder=5,
        )
    return [
        Patch(facecolor=C.ZONE[z], label=ZONE_LABELS[z])
        for z in ZONES
        if (vv.zone == z).any()
    ]


def panel_reactive_power(ax, day: SiteDay, vv: pd.DataFrame):
    metric, tol = vv.attrs["metric"], vv.attrs["tol"]
    not_counted = vv.valid & ~vv.counted
    if NOT_COUNTED_REASON[metric]:
        _shade(ax, day.t, not_counted, C.NOT_COUNTED, 0.22, hatch="///")
    ax.fill_between(
        day.t,
        vv.q_min_final,
        vv.q_max_final,
        color=C.REFERENCE["band"],
        alpha=C.BAND_ALPHA,
        lw=0,
        zorder=1,
    )
    ax.plot(
        day.t, vv.q_max_final, color=C.REFERENCE["required"], lw=0.9, ls="--", zorder=2
    )
    ax.plot(
        day.t, vv.q_min_final, color=C.REFERENCE["required"], lw=0.9, ls="--", zorder=2
    )
    ax.plot(
        day.t,
        vv.q_req,
        color=C.REFERENCE["required"],
        lw=0.8,
        ls=":",
        alpha=0.6,
        zorder=2,
    )
    ax.axhline(0, color=C.REFERENCE["zero"], lw=0.5, zorder=2)
    _lines(
        ax,
        day,
        day.q_phase_pct,
        day.q_pct,
        C.QUANTITY["Q"],
        "Sum of phases" if day.three_phase else "Q",
    )
    zone_handles = _zone_strip(ax, day, vv)
    ax.set_ylim(
        min(Q_RANGE[0], np.nanmin(day.q_phase_pct.to_numpy()) - 5),
        max(Q_RANGE[1], np.nanmax(day.q_phase_pct.to_numpy()) + 8),
    )
    ax.set_ylabel("Reactive power\n(% of rating, + supply / − absorb)")
    _pct_axis(ax, day, "kvar", "Q")
    h, _ = ax.get_legend_handles_labels()
    extra = [
        Patch(
            facecolor=C.REFERENCE["band"],
            alpha=0.25,
            edgecolor=C.REFERENCE["required"],
            ls="--",
            label=f"Accepted band (curve ±{tol*100:.0f}%; Fig. 2.1 at measured P; "
            "wide band below 20% P)",
        ),
        Line2D(
            [],
            [],
            color=C.REFERENCE["required"],
            ls=":",
            alpha=0.6,
            label="Volt-VAr curve",
        ),
    ]
    if NOT_COUNTED_REASON[metric]:
        extra.append(
            Patch(
                facecolor=C.NOT_COUNTED,
                alpha=0.4,
                hatch="///",
                label=f"Not counted in {METRIC_LABELS[metric].lower()}: {NOT_COUNTED_REASON[metric]}",
            )
        )
    _legend(ax, handles=h + extra + zone_handles, fontsize=7, loc="lower left", ncol=2)


def panel_nc_voltvar(ax, day: SiteDay, vv: pd.DataFrame):
    outside = vv.nc_pct != 0
    colors = np.where(
        vv.failure, C.STATUS["nonconformant"], C.NOT_COUNTED
    )
    ax.bar(day.t, vv.nc_pct.where(outside), width=_bar_width(day), color=colors, zorder=3)
    ax.axhline(0, color=C.REFERENCE["zero"], lw=0.5)
    ax.set_ylabel("Volt-VAr\n(pp outside band)")
    ax.text(
        0.99,
        0.95,
        metric_summary(vv),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC"),
    )
    _legend(
        ax,
        handles=[
            Patch(color=C.STATUS["nonconformant"], label="Outside band: counted failure"),
            Patch(
                color=C.NOT_COUNTED,
                label="Outside band: not a failure under this metric / definition",
            ),
        ],
        ncol=2,
        fontsize=7,
        loc="lower left",
    )
    lim = max(2.0, float(np.nanmax(np.abs(vv.nc_pct))) * 1.6)
    ax.set_ylim(-lim, lim)



def panel_nc_voltwatt(ax, day: SiteDay, vw: pd.DataFrame):
    ax.bar(
        day.t,
        vw.nc_pct,
        width=_bar_width(day),
        color=C.STATUS["nonconformant"],
        zorder=3,
        label="Above ceiling (exposed)",
    )
    ax.axhline(0, color=C.REFERENCE["zero"], lw=0.5)
    ax.set_ylabel("Volt-Watt NC\n(pp above ceiling)")
    kwh = float((vw.nc_pct / 100 * day.rating_kva).sum() * day.interval_h)
    ax.text(
        0.99,
        0.95,
        f"Exposed {int(vw.exposed.sum())}/{len(vw)} intervals  ·  NC {int((vw.nc_pct > 0).sum())}"
        f"  ·  {kwh:.2f} kWh above ceiling",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC"),
    )
    _legend(ax, ncol=1)
    ax.set_ylim(0, max(2.0, float(np.nanmax(vw.nc_pct)) * 1.15))


# =============================================================================
# 5. Figures
# =============================================================================
STANDARD_MANDATORY = pd.Timestamp(
    "2021-12-18"
)  # AS/NZS 4777.2:2020 mandatory for new installations


def _fmt(v, unit="", nd=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") + unit
    return f"{v}{unit}"


def metadata_lines(day: SiteDay) -> list[str]:
    """
    Context printed above each figure (not drawn on it): two lines about the site (from the loader's
    info["site"]) and two about the day (computed from the data). Empty fields are skipped.
    """
    m, site = day.meta, day.meta.get("site", {}) or {}
    j = lambda xs: "  ·  ".join(x for x in xs if x)

    loc = " ".join(
        str(x) for x in (site.get("state"), site.get("postcode")) if x is not None
    )
    oem = " ".join(str(x) for x in (site.get("manufacturer"), site.get("model")) if x)
    inv = site.get("inverter_count")
    if oem and inv and inv > 1:
        oem += f" (×{int(inv)})"
    cap = (
        f"AC {_fmt(site['ac_capacity_kw'], ' kW')} / DC {_fmt(site.get('dc_capacity_kw'), ' kW')}"
        if site.get("ac_capacity_kw") is not None
        else ""
    )
    exp = ""
    if "export_limit_kw" in site:
        exp = "export limit " + (
            _fmt(site["export_limit_kw"], " kW")
            if site["export_limit_kw"] is not None
            else "none"
        )
    line1 = j([loc, site.get("dnsp_name"), oem, cap, exp])

    inst = site.get("pv_install_date")
    inst_s = ""
    if inst is not None and pd.notna(inst):
        inst = pd.Timestamp(inst)
        inst_s = f"installed {inst.date()} ({'after' if inst >= STANDARD_MANDATORY else 'before'} AS/NZS 4777.2:2020 became mandatory)"
    flex = site.get("flex_export_detected")
    line2 = j(
        [
            inst_s,
            f"voltage class {site['voltage_class']}"
            if site.get("voltage_class")
            else "",
            f"flexible export {'yes' if flex else 'no'}" if flex is not None else "",
            f"monitor {site['device_type']}" if site.get("device_type") else "",
            f"{site['n_pv_circuits']} PV circuit(s)"
            if site.get("n_pv_circuits")
            else "",
        ]
    )

    expected = int(round(24 / day.interval_h))
    pv_kwh = float(day.p_kw.clip(lower=0).sum() * day.interval_h)
    v = day.v
    line3 = j(
        [
            f"{v.notna().sum()}/{expected} intervals",
            f"PV {pv_kwh:.1f} kWh",
            f"peak P {day.p_pct.max():.0f}% of rating",
            f"V{' (avg)' if day.three_phase else ''} {v.min():.1f}–{v.max():.1f} V",
            f"{int((v > 240).sum())} intervals > 240 V, {int((v > 253).sum())} > 253 V",
        ]
    )
    line4 = j(
        [
            f"source: {m['source']}" if m.get("source") else "",
            f"rating: {m['rating_basis']}" if m.get("rating_basis") else "",
        ]
    )
    return [ln for ln in (line1, line2, line3, line4) if ln]


def _figure(
    day: SiteDay, ratios: list[float], title: str, height: float, show_meta: bool = True
):
    fig, axes = plt.subplots(
        len(ratios),
        1,
        figsize=(12, height),
        sharex=True,
        gridspec_kw={"height_ratios": ratios, "hspace": 0.08},
    )
    label = day.meta.get("label", "")
    n = f"{len(day.phases)}-phase" if day.three_phase else "single-phase"
    fig.suptitle(
        f"{title}\n{label}  ·  {day.meta.get('date', '')}  ·  rating {day.rating_kva:g} kVA  ·  {n}",
        x=0.07,
        ha="left",
        y=0.995,
        fontsize=10.5,
    )
    if show_meta:  # printed in the notebook output, not drawn on the figure
        print("\n".join(metadata_lines(day)))
    fig.subplots_adjust(
        top=1 - 0.75 / height, bottom=0.45 / height, left=0.09, right=0.9
    )
    _time_axis(axes[-1])
    return fig, axes


def _as_day(df_or_day, rating_kva, meta) -> SiteDay:
    """Accept a SiteDay, or (df, rating_kva, meta), or (df, info) where info is a loader's dict."""
    if isinstance(df_or_day, SiteDay):
        return df_or_day
    if isinstance(rating_kva, dict):
        meta, rating_kva = rating_kva, rating_kva["rating_kva"]
    return prepare(df_or_day, rating_kva, meta)


def plot_voltvar_day(
    df,
    rating_kva=None,
    meta: dict | None = None,
    show_meta: bool = True,
    metric: str | None = None,
    tol: float | None = None,
    failure_definition: str | None = None,
):
    """
    A.1 Volt-VAr: V -> P (20% gate) -> Q vs accepted band -> outside-band bars.
    metric: 'remote_conformance_rate' (default), 'nonconformance_frequency' or
    'legacy_conformance_rate'; it changes only which intervals count (shading
    and summary), not the band. tol / failure_definition: ciccada_config defaults.
    """
    day = _as_day(df, rating_kva, meta)
    vv = assess_voltvar(day, metric, tol, failure_definition)
    fig, (a_v, a_p, a_q, a_nc) = _figure(
        day,
        [1.6, 1.6, 2.4, 1.0],
        f"Volt-VAr response  ·  {METRIC_LABELS[vv.attrs['metric']]}",
        12.5,
        show_meta,
    )
    panel_voltage(a_v, day, "voltvar")
    panel_active_power(a_p, day, gate=True)
    panel_reactive_power(a_q, day, vv)
    panel_nc_voltvar(a_nc, day, vv)
    return fig


def plot_voltwatt_day(
    df, rating_kva=None, meta: dict | None = None, show_meta: bool = True
):
    """A.2 Volt-Watt: V -> P vs ceiling (and PV potential if given) -> Volt-Watt NC."""
    day = _as_day(df, rating_kva, meta)
    vw = assess_voltwatt(day)
    fig, (a_v, a_p, a_nc) = _figure(
        day, [1.6, 2.2, 0.9], "Volt-Watt response", 9, show_meta
    )
    panel_voltage(a_v, day, "voltwatt")
    panel_active_power(a_p, day, vw=vw)
    panel_nc_voltwatt(a_nc, day, vw)
    return fig


def plot_operational_day(
    df,
    rating_kva=None,
    meta: dict | None = None,
    show_meta: bool = True,
    metric: str | None = None,
    tol: float | None = None,
    failure_definition: str | None = None,
):
    """A.0 Both operational modes: V -> P vs ceiling -> Volt-Watt NC -> Q vs band -> Volt-VAr bars."""
    day = _as_day(df, rating_kva, meta)
    vv, vw = assess_voltvar(day, metric, tol, failure_definition), assess_voltwatt(day)
    fig, (a_v, a_p, a_pnc, a_q, a_qnc) = _figure(
        day,
        [1.6, 2.0, 0.8, 2.4, 1.0],
        f"Operational modes: Volt-Watt and Volt-VAr  ·  {METRIC_LABELS[vv.attrs['metric']]}",
        15.5,
        show_meta,
    )
    panel_voltage(a_v, day, "operational")
    panel_active_power(a_p, day, vw=vw, gate=True)
    panel_nc_voltwatt(a_pnc, day, vw)
    panel_reactive_power(a_q, day, vv)
    panel_nc_voltvar(a_qnc, day, vv)
    return fig



def plot_protective_day(df, rating_kva=None, meta: dict | None = None):
    """A.3 Protective functions (sustained operation, anti-islanding). TO BE DONE."""
    raise NotImplementedError("A.3 protective-function day plot is not built yet.")
