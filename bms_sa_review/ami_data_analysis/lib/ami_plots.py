"""
Phase 3 (and reused in Phase 6): example-circuit and example-site plots.
============================================================================

Every function here takes a prepared, already-queried DataFrame and explicit
parameters -- no globals, no querying -- mirroring the house convention in
`data_query/lib/explore_plots.py`. `to_aest` is this module's own copy of that
file's helper (not imported cross-package, to keep `ami_data_analysis`
self-contained per its own tests), applied because `ts.t_stamp` is UTC and a
human reading a daily curve wants to see it against a normal Australian day.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from bms_sa_review.ami_data_analysis.config import ami_config as C

__all__ = ["to_aest", "plot_circuit_day", "plot_aggregation_check", "aest_day_window_utc"]


def to_aest(t_stamp: pd.Series) -> pd.Series:
    """UTC timestamps -> AEST (fixed UTC+10, no DST). Pure.

    Coerces via `pd.to_datetime` first rather than assuming `t_stamp` already
    carries a datetime64 dtype: an Athena/awswrangler query that returns zero
    rows for its day/circuit window (real data gaps happen -- a circuit can
    exist in `meta_up23c` without having `ts` rows for every day) comes back
    with an empty, untyped column rather than one Glue's TIMESTAMP schema was
    applied to, and `.dt` raises on that. `pd.to_datetime` on an empty Series
    returns an empty datetime64 Series, so this handles both the empty-frame
    case and any other not-yet-parsed timestamp representation the same way.
    """
    if not pd.api.types.is_datetime64_any_dtype(t_stamp):
        t_stamp = pd.to_datetime(t_stamp)
    if t_stamp.dt.tz is None:
        t_stamp = t_stamp.dt.tz_localize("UTC")
    return t_stamp.dt.tz_convert(C.FIXED_OFFSET)


def aest_day_window_utc(day_start, *, n_days: int = 1, fixed_offset=None) -> dict:
    """
    Convert an AEST calendar-day start into the equivalent UTC query window,
    plus the `(year, month)` partition pairs that window spans. Pure.

    `ts.t_stamp` is UTC and `year`/`month` are partition columns derived
    from it. Picking "N full AEST days" by reusing a UTC calendar day's
    bounds (e.g. `t_stamp >= '2025-06-01 00:00' AND < '2025-06-02 00:00'`)
    actually selects 10:00 AEST to 10:00 AEST the next day, not midnight to
    midnight -- the same UTC/AEST mismatch `plot_circuit_day`'s tz= comment
    warns about for tick labels, but here it would silently shift which
    rows a query returns, not just how they're labelled. This makes the
    conversion explicit and testable rather than inline notebook
    arithmetic. `year_month_pairs`/`year_month_sql` matter because a window
    that is exactly `n_days` whole AEST days can still straddle a UTC
    calendar-month boundary (AEST is UTC+10, so the AEST day starts 10
    hours into the previous UTC day) -- a query that only filters on the
    AEST start date's own `(year, month)` would silently miss the tail end
    of the window.
    """
    if fixed_offset is None:
        fixed_offset = C.FIXED_OFFSET
    aest_start = pd.Timestamp(day_start).tz_localize(fixed_offset)
    aest_end = aest_start + pd.Timedelta(days=n_days)
    start_utc = aest_start.tz_convert("UTC").tz_localize(None)
    end_utc = aest_end.tz_convert("UTC").tz_localize(None)

    last_included_moment = end_utc - pd.Timedelta(microseconds=1)
    year_month_pairs = set()
    day_pointer = start_utc.normalize()
    while day_pointer <= last_included_moment:
        year_month_pairs.add((day_pointer.year, day_pointer.month))
        day_pointer += pd.Timedelta(days=1)
    year_month_pairs = sorted(year_month_pairs)

    return {
        "aest_start": aest_start,
        "aest_end": aest_end,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "year_month_pairs": year_month_pairs,
        "year_month_sql": " OR ".join(
            f"(year = {y} AND month = {m})" for y, m in year_month_pairs
        ),
    }


def plot_circuit_day(
    frame: pd.DataFrame, *,
    circuit_column: str = "circuit_id",
    time_column: str = "t_stamp",
    value_column: str = "power_signed",
    label_column: str | None = "circuit_type",
    title: str = "",
    ylabel: str = "power (W, sign-corrected)",
    ax=None,
):
    """
    One or more circuits' readings over a day, on one axis. Pure-ish: builds
    and returns a Figure/Axes, draws nothing to screen itself (the caller's
    notebook cell does that by not suppressing the return value, or by
    calling `plt.show()`).

    `frame` is expected already filtered to one day and one site -- this
    function does not do that filtering, so a caller who wants "one site, one
    day" says so explicitly at the query, not by trusting a default here.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.figure

    if frame is None or not len(frame):
        ax.set_title(title or "(no data)")
        return fig, ax

    plot_frame = frame.copy()
    plot_frame["_t_aest"] = to_aest(plot_frame[time_column])

    for circuit_id, group in plot_frame.groupby(circuit_column):
        # Athena/awswrangler gives no row-ordering guarantee -- drawing raw
        # query-return order connects points out of time sequence, which
        # looks like a zigzag/star-burst rather than a smooth daily curve.
        group = group.sort_values("_t_aest")
        label = str(circuit_id)
        if label_column and label_column in group.columns:
            values = group[label_column].dropna().unique()
            if len(values):
                label = f"{circuit_id} ({values[0]})"
        ax.plot(group["_t_aest"], group[value_column], label=label, linewidth=1.2)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time (AEST)")
    # tz= is required here, not optional: matplotlib's DateFormatter renders
    # tick labels in matplotlib's rcParam timezone (UTC by default) regardless
    # of the tzinfo the plotted datetimes themselves carry -- the AEST
    # conversion in `to_aest` has zero effect on the displayed axis without
    # this. Confirmed empirically: an AEST-tz-aware series plotted without
    # tz= here renders labels 10 hours behind the true AEST time (a UTC
    # midnight-aligned window showed tick labels starting at "23:00" instead
    # of the correct "09:00"), while the underlying plotted data and any
    # hour-based computation done on the pandas Series beforehand (e.g.
    # `ami_signal.night_window_stats`) were already correct -- only the
    # rendered labels were wrong.
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=C.FIXED_OFFSET))
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_aggregation_check(
    long: pd.DataFrame, *,
    candidate_circuit_ids, component_circuit_ids,
    time_column: str = "t_stamp",
    circuit_column: str = "circuit_id",
    power_column: str = "power_signed",
    title: str = "",
    ax=None,
):
    """
    Candidate vs sum(components), overlaid, for one aggregation check. Pure-ish.

    The visual companion to `ami_taxonomy.check_aggregation` -- two lines that
    sit on top of each other are the picture of "this candidate IS the sum of
    these components"; visible daylight between them is the picture of "it
    is not". Makes the arithmetic verdict legible at a glance rather than
    trusted from a single tolerance number.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.figure

    if long is None or not len(long):
        ax.set_title(title or "(no data)")
        return fig, ax

    frame = long.copy()
    frame["_t_aest"] = to_aest(frame[time_column])

    candidate = (
        frame[frame[circuit_column].isin(candidate_circuit_ids)]
        .groupby("_t_aest")[power_column].sum()
    )
    components = (
        frame[frame[circuit_column].isin(component_circuit_ids)]
        .groupby("_t_aest")[power_column].sum()
    )
    ax.plot(candidate.index, candidate.values, label="candidate (own reading)",
            linewidth=1.6)
    ax.plot(components.index, components.values, label="sum(components)",
            linewidth=1.2, linestyle="--")

    ax.set_title(title)
    ax.set_ylabel("power (W, sign-corrected)")
    ax.set_xlabel("time (AEST)")
    # tz= is required here, not optional: matplotlib's DateFormatter renders
    # tick labels in matplotlib's rcParam timezone (UTC by default) regardless
    # of the tzinfo the plotted datetimes themselves carry -- the AEST
    # conversion in `to_aest` has zero effect on the displayed axis without
    # this. Confirmed empirically: an AEST-tz-aware series plotted without
    # tz= here renders labels 10 hours behind the true AEST time (a UTC
    # midnight-aligned window showed tick labels starting at "23:00" instead
    # of the correct "09:00"), while the underlying plotted data and any
    # hour-based computation done on the pandas Series beforehand (e.g.
    # `ami_signal.night_window_stats`) were already correct -- only the
    # rendered labels were wrong.
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=C.FIXED_OFFSET))
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax
