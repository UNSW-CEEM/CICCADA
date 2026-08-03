"""Plot functions over ``result_views.py``'s aggregated frames.

Every function here consumes an already-aggregated ``pandas.DataFrame`` (or a
small context dict) produced by ``result_views.py``. None of them queries raw
data, accepts a ``config``/``scope``, or interpolates a column name from
notebook input -- that boundary belongs entirely to ``result_views.py``.
This module only renders what it is handed.

Rules enforced throughout:

- every subtitle states measurement basis, voltage-measurement location,
  voltage basis, capacity basis and both sign-review states;
- every panel displays its own denominator (``n=...``) or an explicit
  "nothing to classify" / low-denominator panel;
- a group with a zero denominator is never ranked or plotted as if it had a
  defined rate -- excluded rows are called out, not silently dropped;
- ``pass``/``fail``/``conforming`` is never used for a proxy or
  observability label; the source table's own status string is always
  preserved verbatim in axis/legend labels;
- Volt-Watt's ``proxy_does_not_exceed_curve_ceiling`` is never relabelled
  conformance;
- the curtailment panel renders no number at all.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .as4777_curves import VOLT_VAR, VOLT_WATT
from .result_views import (
    CURTAILMENT_UNAVAILABLE_CONTEXT,
    VOLTVAR_DENOMINATOR_COLUMNS,
    VOLTVAR_OBSERVABILITY_STATUSES,
    VOLTVAR_STATUS_COLUMNS,
    VOLTWATT_DENOMINATOR_COLUMNS,
    VOLTWATT_OBSERVABILITY_STATUSES,
    VOLTWATT_STATUS_COLUMNS,
)

DENOMINATOR_LABELS: dict[str, str] = {
    "n_ineligible_site": "ineligible site",
    "n_missing_input": "missing input",
    "n_not_activated": "not activated",
    "n_sign_unverified": "sign unverified",
    "n_not_exporting": "not exporting",
    "n_capacity_unavailable": "capacity unavailable",
    "n_below_minimum_active_power": "below min active power",
    "n_assessable": "assessable",
}

STATUS_LABELS: dict[str, str] = {
    "n_proxy_within_curve_band": "proxy_within_curve_band",
    "n_proxy_q_adverse": "proxy_q_adverse",
    "n_proxy_q_inactive": "proxy_q_inactive",
    "n_proxy_q_significant_shortfall": "proxy_q_significant_shortfall",
    "n_proxy_q_near_conformant": "proxy_q_near_conformant",
    "n_proxy_q_major_surplus": "proxy_q_major_surplus",
    "n_proxy_exceeds_curve_ceiling": "proxy_exceeds_curve_ceiling",
    "n_proxy_does_not_exceed_curve_ceiling": "proxy_does_not_exceed_curve_ceiling",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _subtitle(context: Mapping[str, object]) -> str:
    return (
        f"basis={context.get('measurement_basis', 'n/a')} | "
        f"voltage_loc={context.get('voltage_measurement_location', 'n/a')} | "
        f"voltage_basis={context.get('voltage_basis', 'n/a')} | "
        f"capacity_basis={context.get('capacity_basis', 'n/a')} | "
        f"active_sign={context.get('active_sign_review_state', 'n/a')} | "
        f"reactive_sign={context.get('reactive_sign_review_state', 'n/a')}"
    )


def _apply_subtitle(ax: Axes, context: Mapping[str, object]) -> None:
    ax.text(
        0.0,
        -0.32,
        _subtitle(context),
        transform=ax.transAxes,
        fontsize=7,
        color="dimgray",
        ha="left",
        va="top",
    )


def _new_axes(ax: Axes | None, figsize: tuple[float, float] = (8.0, 4.0)) -> Axes:
    if ax is not None:
        return ax
    _, created = plt.subplots(figsize=figsize)
    return created


def _empty_panel(ax: Axes, message: str, *, context: Mapping[str, object] | None = None) -> Axes:
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="firebrick",
        fontsize=9,
        wrap=True,
    )
    ax.set_axis_off()
    if context is not None:
        ax.text(
            0.0,
            0.02,
            _subtitle(context),
            transform=ax.transAxes,
            fontsize=7,
            color="dimgray",
        )
    return ax


def _fraction_breakdown(
    frame: pd.DataFrame,
    fraction_columns: Mapping[str, str],
    *,
    denominator_column: str,
    title: str,
    context: Mapping[str, object],
    index_dimension: str | None = None,
    minimum_denominator: int = 0,
    ax: Axes | None = None,
) -> Axes:
    """Generic fraction-of-denominator bar chart.

    ``fraction_columns`` maps a fraction column name that ``result_views.py``
    already computed (e.g. ``'assessable_fraction_of_source'``) to a display
    label. This function never invents a fraction; it only plots columns
    that were computed with explicit null-on-zero handling upstream.
    """

    if frame is None or frame.empty:
        raise ValueError(f"{title}: frame is empty -- nothing to plot")
    ax = _new_axes(ax)
    n_total = int(frame[denominator_column].sum())

    if index_dimension and index_dimension in frame.columns and len(frame) > 1:
        plot_source = frame
        if minimum_denominator:
            plot_source = frame[frame[denominator_column] >= minimum_denominator]
        n_excluded = len(frame) - len(plot_source)
        if plot_source.empty:
            return _empty_panel(
                ax,
                f"No rows meet minimum_denominator={minimum_denominator:,} "
                f"for {denominator_column} -- nothing plotted, not zero.",
                context=context,
            )
        plot_frame = plot_source.set_index(index_dimension)[
            list(fraction_columns)
        ].astype(float)
        plot_frame.columns = list(fraction_columns.values())
        plot_frame.plot.bar(stacked=True, ax=ax, width=0.85)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        if n_excluded:
            ax.annotate(
                f"{n_excluded} row(s) below minimum_denominator excluded from ranking",
                xy=(0.5, 1.05),
                xycoords="axes fraction",
                ha="center",
                fontsize=7,
                color="firebrick",
            )
    else:
        totals = frame[list(fraction_columns)].iloc[0].astype(float)
        totals.index = list(fraction_columns.values())
        totals.plot.bar(ax=ax, color="#4472c4")

    ax.set_ylabel("fraction of denominator")
    ax.set_title(f"{title} (n={n_total:,})")
    if n_total == 0:
        ax.text(
            0.5,
            0.5,
            f"{denominator_column} = 0 -- nothing to classify here.\n"
            "This is not a zero rate; see the denominator breakdown for why.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="firebrick",
            fontsize=9,
        )
    _apply_subtitle(ax, context)
    return ax


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------


def plot_denominator_breakdown(
    frame: pd.DataFrame,
    *,
    mechanism_name: str,
    context: Mapping[str, object],
    denominator_columns: Sequence[str] | None = None,
    index_dimension: str | None = None,
    minimum_denominator: int = 0,
    ax: Axes | None = None,
) -> Axes:
    """Denominator-bucket breakdown for a Volt-VAr or Volt-Watt view frame.

    ``mechanism_name`` should be ``'Volt-VAr'`` or ``'Volt-Watt'`` and
    determines the default column set if ``denominator_columns`` is omitted.
    """

    if denominator_columns is None:
        denominator_columns = (
            VOLTVAR_DENOMINATOR_COLUMNS
            if mechanism_name.lower().startswith("volt-var")
            or mechanism_name.lower().startswith("voltvar")
            else VOLTWATT_DENOMINATOR_COLUMNS
        )
    fraction_columns = {
        f"{(c[2:] if c.startswith('n_') else c)}_fraction_of_source": DENOMINATOR_LABELS.get(c, c)
        for c in denominator_columns
    }
    return _fraction_breakdown(
        frame,
        fraction_columns,
        denominator_column="n_source_intervals",
        title=f"{mechanism_name} denominator breakdown",
        context=context,
        index_dimension=index_dimension,
        minimum_denominator=minimum_denominator,
        ax=ax,
    )


def plot_status_breakdown(
    frame: pd.DataFrame,
    *,
    mechanism_name: str,
    context: Mapping[str, object],
    status_columns: Sequence[str] | None = None,
    index_dimension: str | None = None,
    minimum_denominator: int = 0,
    ax: Axes | None = None,
) -> Axes:
    """Proxy-curve status breakdown, always expressed as a fraction of
    ``n_assessable`` -- never labelled pass/fail/conforming.
    """

    is_voltwatt = mechanism_name.lower().startswith("volt-watt") or mechanism_name.lower().startswith("voltwatt")
    if status_columns is None:
        status_columns = VOLTWATT_STATUS_COLUMNS if is_voltwatt else VOLTVAR_STATUS_COLUMNS
    fraction_columns = {
        f"{(c[2:] if c.startswith('n_') else c)}_fraction_of_assessable": STATUS_LABELS.get(c, c)
        for c in status_columns
    }
    return _fraction_breakdown(
        frame,
        fraction_columns,
        denominator_column="n_assessable",
        title=f"{mechanism_name} proxy-curve status",
        context=context,
        index_dimension=index_dimension,
        minimum_denominator=minimum_denominator,
        ax=ax,
    )


def plot_monthly_coverage(
    frame: pd.DataFrame,
    *,
    mechanism_name: str,
    context: Mapping[str, object],
    denominator_column: str = "n_source_intervals",
    assessable_column: str = "n_assessable",
    ax: Axes | None = None,
) -> Axes:
    """Source-interval and assessable-interval counts by UTC year/month.

    ``frame`` should come from a denominator or status view grouped by
    ``dimensions=("year_utc", "month_utc")``.
    """

    if frame is None or frame.empty:
        raise ValueError("monthly coverage frame is empty -- nothing to plot")
    if "year_utc" not in frame.columns or "month_utc" not in frame.columns:
        raise ValueError(
            "plot_monthly_coverage requires a frame grouped by "
            "dimensions=('year_utc', 'month_utc')"
        )
    ax = _new_axes(ax)
    ordered = frame.sort_values(["year_utc", "month_utc"])
    period = ordered["year_utc"].astype(str) + "-" + ordered["month_utc"].astype(str).str.zfill(2)
    ax.bar(period, ordered[denominator_column], color="#a5a5a5", label=denominator_column)
    if assessable_column in ordered.columns:
        ax.bar(period, ordered[assessable_column], color="#4472c4", label=assessable_column)
    ax.set_ylabel("intervals")
    ax.set_title(f"{mechanism_name} monthly coverage (UTC year/month)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    _apply_subtitle(ax, context)
    return ax


def plot_voltage_bin_status(
    frame: pd.DataFrame,
    *,
    mechanism_name: str,
    context: Mapping[str, object],
    status_columns: Sequence[str] | None = None,
    minimum_denominator: int = 0,
    ax: Axes | None = None,
) -> Axes:
    """Proxy-curve status by ``voltage_bin_lower_v``, with AS/NZS 4777.2
    breakpoints marked for reference.

    ``frame`` should come from ``voltvar_status_view``/``voltwatt_status_view``
    grouped by ``dimensions=("voltage_bin_lower_v",)``.
    """

    if "voltage_bin_lower_v" not in frame.columns:
        raise ValueError(
            "plot_voltage_bin_status requires a frame grouped by "
            "dimensions=('voltage_bin_lower_v',)"
        )
    ax = plot_status_breakdown(
        frame,
        mechanism_name=mechanism_name,
        context=context,
        status_columns=status_columns,
        index_dimension="voltage_bin_lower_v",
        minimum_denominator=minimum_denominator,
        ax=ax,
    )
    is_voltwatt = mechanism_name.lower().startswith("volt-watt") or mechanism_name.lower().startswith("voltwatt")
    breakpoints = (
        (VOLT_WATT.v1, "V1"), (VOLT_WATT.v2, "V2")
    ) if is_voltwatt else (
        (VOLT_VAR.v1, "V1"), (VOLT_VAR.v2, "V2"), (VOLT_VAR.v3, "V3"), (VOLT_VAR.v4, "V4")
    )
    for voltage, label in breakpoints:
        ax.axvline(voltage, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("voltage bin lower bound (V) -- dashed lines mark curve breakpoints")
    return ax


def plot_cohort_comparison(
    frame: pd.DataFrame,
    *,
    mechanism_name: str,
    context: Mapping[str, object],
    denominator_column: str = "n_source_intervals",
    ax: Axes | None = None,
) -> Axes:
    """Denominator breakdown by ``analysis_cohort``.

    ``frame`` should come from a denominator view grouped by
    ``dimensions=("analysis_cohort",)``.
    """

    if "analysis_cohort" not in frame.columns:
        raise ValueError(
            "plot_cohort_comparison requires a frame grouped by "
            "dimensions=('analysis_cohort',)"
        )
    is_voltwatt = mechanism_name.lower().startswith("volt-watt") or mechanism_name.lower().startswith("voltwatt")
    denominator_columns = VOLTWATT_DENOMINATOR_COLUMNS if is_voltwatt else VOLTVAR_DENOMINATOR_COLUMNS
    return plot_denominator_breakdown(
        frame,
        mechanism_name=mechanism_name,
        context=context,
        denominator_columns=denominator_columns,
        index_dimension="analysis_cohort",
        ax=ax,
    )


def plot_site_profile(
    voltvar_frame: pd.DataFrame,
    voltwatt_frame: pd.DataFrame,
    serial: str,
    *,
    context: Mapping[str, object],
) -> Figure:
    """A single site's monthly Volt-VAr/Volt-Watt denominator profile.

    Both frames should come from denominator views grouped by
    ``dimensions=("serial", "year_utc", "month_utc")`` (or filtered/queried
    upstream to one serial); this function filters to ``serial`` itself so
    callers can pass an already fleet-wide-by-site-and-month frame directly.
    Denominator and methodology context are shown next to the plot so an
    unassessable site is never mistaken for a poor-performing one.
    """

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, frame, mechanism_name, denom_cols in (
        (axes[0], voltvar_frame, "Volt-VAr", VOLTVAR_DENOMINATOR_COLUMNS),
        (axes[1], voltwatt_frame, "Volt-Watt", VOLTWATT_DENOMINATOR_COLUMNS),
    ):
        if frame is None or "serial" not in frame.columns:
            _empty_panel(ax, f"{mechanism_name}: no per-site frame provided", context=context)
            continue
        site_frame = frame[frame["serial"] == serial]
        if site_frame.empty:
            _empty_panel(
                ax,
                f"{mechanism_name}: no rows for site {serial} in this scope",
                context=context,
            )
            continue
        plot_denominator_breakdown(
            site_frame,
            mechanism_name=mechanism_name,
            context=context,
            denominator_columns=denom_cols,
            index_dimension="month_utc" if "month_utc" in site_frame.columns else None,
            ax=ax,
        )
    fig.suptitle(f"Site profile -- serial {serial}")
    fig.tight_layout()
    return fig


def plot_observability_status(
    frame: pd.DataFrame,
    *,
    mechanism_name: str,
    context: Mapping[str, object],
    index_dimension: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Observability status breakdown for Volt-VAr or Volt-Watt.

    ``frame`` should come from ``observability_status_view``. Volt-VAr and
    Volt-Watt statuses live in separate ``n_voltvar_status_*`` /
    ``n_voltwatt_status_*`` columns upstream and are never merged here --
    pass the correct ``mechanism_name`` to select which set is plotted.
    """

    is_voltwatt = mechanism_name.lower().startswith("volt-watt") or mechanism_name.lower().startswith("voltwatt")
    statuses = VOLTWATT_OBSERVABILITY_STATUSES if is_voltwatt else VOLTVAR_OBSERVABILITY_STATUSES
    prefix = "voltwatt" if is_voltwatt else "voltvar"
    fraction_columns = {
        f"{prefix}_status_{status}_fraction_of_site_phase_months": status
        for status in statuses
    }
    return _fraction_breakdown(
        frame,
        fraction_columns,
        denominator_column="n_site_phase_months",
        title=f"{mechanism_name} response observability status",
        context=context,
        index_dimension=index_dimension,
        ax=ax,
    )


def plot_observability_scatter(
    frame: pd.DataFrame,
    *,
    mechanism_name: str,
    context: Mapping[str, object],
    ax: Axes | None = None,
) -> Axes:
    """Slope vs. correlation scatter across whatever rows ``frame`` contains
    (e.g. one point per site, or per site-month), sized by excitation count.

    ``frame`` should come from ``observability_metric_view``. This is
    direction/association evidence only -- axis labels avoid any causal or
    conformance language.
    """

    is_voltwatt = mechanism_name.lower().startswith("volt-watt") or mechanism_name.lower().startswith("voltwatt")
    if is_voltwatt:
        slope_col, corr_col, n_col = (
            "voltwatt_slope_w_per_v_weighted_mean",
            "voltwatt_voltage_correlation_mean",
            "n_voltwatt_excited_export_intervals",
        )
    else:
        slope_col, corr_col, n_col = (
            "voltvar_slope_var_per_v_weighted_mean",
            "voltvar_voltage_correlation_mean",
            "n_voltvar_excited_intervals",
        )
    if frame is None or frame.empty:
        raise ValueError("observability metric frame is empty -- nothing to plot")
    ax = _new_axes(ax)
    plottable = frame.dropna(subset=[slope_col, corr_col])
    if plottable.empty:
        return _empty_panel(
            ax,
            f"No rows have a defined {slope_col}/{corr_col} "
            "(zero-variance or fully unexcited groups only)",
            context=context,
        )
    sizes = 20 + 60 * (
        plottable[n_col] / plottable[n_col].max() if plottable[n_col].max() else 0
    )
    ax.scatter(plottable[corr_col], plottable[slope_col], s=sizes, alpha=0.6, color="#4472c4")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(f"{corr_col} (association only, not causal)")
    ax.set_ylabel(slope_col)
    ax.set_title(f"{mechanism_name} observability: slope vs. correlation (n rows={len(plottable)})")
    _apply_subtitle(ax, context)
    return ax


def plot_curtailment_unavailable(
    curtailment_context: Mapping[str, object] = CURTAILMENT_UNAVAILABLE_CONTEXT,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Render only the fixed unavailable-gate-7 panel.

    No curtailment energy, rate, frequency or blended score is ever
    computed or displayed here -- this function accepts no data frame at
    all, only the fixed context dict, precisely so it cannot be handed a
    fabricated number by mistake.
    """

    ax = _new_axes(ax, figsize=(8.0, 3.0))
    ax.text(
        0.5,
        0.6,
        f"unavailable — {curtailment_context.get('reason', 'methodology gate 7 unmet')}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color="firebrick",
    )
    ax.text(
        0.5,
        0.3,
        str(curtailment_context.get("detail", "")),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray",
        wrap=True,
    )
    ax.set_axis_off()
    return ax
