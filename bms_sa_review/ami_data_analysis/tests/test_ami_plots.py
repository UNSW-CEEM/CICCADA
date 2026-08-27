"""
Smoke tests for the example-circuit plots: they must build a Figure without
raising, with the number of lines a human would expect, on both real and
edge-case (empty) input. Not a check on pixels -- just "did it crash, and did
it draw what it was given."
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless -- no display in CI/sandbox

import pandas as pd
import pytest

from bms_sa_review.ami_data_analysis.lib import ami_plots as P


def _sample_frame():
    times = pd.date_range("2025-06-01", periods=10, freq="5min", tz="UTC")
    rows = []
    for circuit_id, ctype in ((1, "load_pool"), (2, "load_hot_water")):
        for t in times:
            rows.append({
                "circuit_id": circuit_id, "circuit_type": ctype,
                "t_stamp": t, "power_signed": 100.0,
            })
    return pd.DataFrame(rows)


def test_to_aest_shifts_by_ten_hours():
    utc = pd.Series(pd.to_datetime(["2025-06-01 00:00:00"])).dt.tz_localize("UTC")
    aest = P.to_aest(utc)
    assert aest.dt.hour.iloc[0] == 10


def test_to_aest_is_a_noop_reapplication_safe():
    already_utc = pd.Series(pd.to_datetime(["2025-06-01 00:00:00"]))
    once = P.to_aest(already_utc)
    assert once.dt.tz is not None


def test_to_aest_handles_empty_series_without_raising():
    """
    A day/circuit window with zero `ts` rows (a real data gap -- a circuit
    can exist in `meta_up23c` without having rows for every day) comes back
    as an empty, untyped column rather than one Glue's TIMESTAMP schema was
    applied to. `.dt.tz` on that used to raise
    "Can only use .dt accessor with datetimelike values" -- this must not.
    """
    empty = pd.Series([], dtype=object)
    out = P.to_aest(empty)
    assert len(out) == 0
    assert pd.api.types.is_datetime64_any_dtype(out)


def test_to_aest_coerces_non_datetime_dtype():
    # A column that never got typed as datetime64 (e.g. from an empty/odd
    # query result) but holds valid timestamp strings must still convert.
    untyped = pd.Series(["2025-06-01 00:00:00", "2025-06-01 01:00:00"], dtype=object)
    out = P.to_aest(untyped)
    assert out.dt.hour.iloc[0] == 10  # UTC 00:00 -> AEST 10:00
    assert out.dt.tz is not None


def test_aest_day_window_utc_one_day_shifts_ten_hours_earlier():
    # AEST 2025-06-01 00:00 -> UTC 2025-05-31 14:00 (AEST is UTC+10)
    window = P.aest_day_window_utc("2025-06-01", n_days=1)
    assert window["start_utc"] == pd.Timestamp("2025-05-31 14:00:00")
    assert window["end_utc"] == pd.Timestamp("2025-06-01 14:00:00")


def test_aest_day_window_utc_two_days_spans_expected_range():
    window = P.aest_day_window_utc("2025-06-01", n_days=2)
    assert window["end_utc"] - window["start_utc"] == pd.Timedelta(days=2)
    assert window["start_utc"] == pd.Timestamp("2025-05-31 14:00:00")
    assert window["end_utc"] == pd.Timestamp("2025-06-02 14:00:00")


def test_aest_day_window_utc_crosses_a_month_boundary_in_utc():
    # 2 AEST days from 2025-06-01 span UTC 2025-05-31 .. 2025-06-02 --
    # both May and June partitions must be included even though the AEST
    # start date is entirely in June.
    window = P.aest_day_window_utc("2025-06-01", n_days=2)
    assert window["year_month_pairs"] == [(2025, 5), (2025, 6)]
    assert "(year = 2025 AND month = 5)" in window["year_month_sql"]
    assert "(year = 2025 AND month = 6)" in window["year_month_sql"]


def test_aest_day_window_utc_single_month_when_no_boundary_crossed():
    # AEST 2025-06-15 + 1 day stays well inside UTC June.
    window = P.aest_day_window_utc("2025-06-15", n_days=1)
    assert window["year_month_pairs"] == [(2025, 6)]


def test_aest_day_window_utc_does_not_include_trailing_month_at_exact_boundary():
    # AEST 2025-06-29 + 2 days = AEST 2025-07-01 00:00, i.e. UTC
    # 2025-06-30 14:00 (exclusive) -- the window never touches July at all.
    window = P.aest_day_window_utc("2025-06-29", n_days=2)
    assert window["end_utc"] == pd.Timestamp("2025-06-30 14:00:00")
    assert window["year_month_pairs"] == [(2025, 6)]


def test_plot_circuit_day_draws_one_line_per_circuit():
    fig, ax = P.plot_circuit_day(_sample_frame(), title="test")
    assert len(ax.get_lines()) == 2


def test_plot_circuit_day_sorts_out_of_order_rows_by_time():
    """
    Athena/awswrangler makes no ordering guarantee. Rows arriving shuffled
    must still draw a time-ordered line, not connect points in query-return
    order (which draws a zigzag of long diagonals across the whole plot).
    """
    times = pd.date_range("2025-06-01", periods=10, freq="5min", tz="UTC")
    shuffled = [5, 0, 8, 2, 9, 1, 7, 3, 6, 4]
    frame = pd.DataFrame({
        "circuit_id": [1] * 10,
        "circuit_type": ["load_pool"] * 10,
        "t_stamp": [times[i] for i in shuffled],
        "power_signed": [float(i) for i in shuffled],
    })
    fig, ax = P.plot_circuit_day(frame)
    xdata = list(ax.get_lines()[0].get_xdata())
    assert xdata == sorted(xdata), "plotted x-data must be time-ordered, not query row order"


def test_plot_circuit_day_empty_frame_does_not_raise():
    fig, ax = P.plot_circuit_day(pd.DataFrame(columns=["circuit_id", "t_stamp", "power_signed"]))
    assert fig is not None


def test_plot_aggregation_check_draws_two_lines():
    frame = _sample_frame()
    fig, ax = P.plot_aggregation_check(
        frame, candidate_circuit_ids=[1], component_circuit_ids=[2], title="test"
    )
    assert len(ax.get_lines()) == 2


def test_plot_circuit_day_axis_labels_are_aest_not_utc():
    """
    matplotlib's DateFormatter renders tick labels in its rcParam timezone
    (UTC by default) regardless of the plotted datetimes' own tzinfo --
    confirmed empirically (a UTC-midnight-aligned AEST series plotted
    without `tz=` passed to DateFormatter renders a tick sequence starting
    at "23:00", i.e. 10 hours behind the true AEST time). With `tz=` set
    correctly, matplotlib's auto tick locator places 3-hour-aligned ticks --
    for a window starting at 10:00 AEST that means the first tick reads
    "09:00" (the nearest round boundary at/before the data start), then
    "12:00", "15:00", ... -- never a UTC-clock value like "00:00" or "23:00".
    """
    times = pd.date_range("2025-06-01 00:00:00", periods=24, freq="1h", tz="UTC")
    frame = pd.DataFrame({
        "circuit_id": [1] * 24, "circuit_type": ["pv_site"] * 24,
        "t_stamp": times, "power_signed": list(range(24)),
    })
    fig, ax = P.plot_circuit_day(frame)
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["09:00", "12:00", "15:00", "18:00", "21:00", "00:00", "03:00", "06:00", "09:00"], (
        f"got {labels!r} -- DateFormatter is likely missing tz=C.FIXED_OFFSET, which would "
        f"instead show the raw UTC clock time (a tick sequence starting at '23:00')"
    )


def test_plot_aggregation_check_axis_labels_are_aest_not_utc():
    times = pd.date_range("2025-06-01 00:00:00", periods=24, freq="1h", tz="UTC")
    frame = pd.DataFrame({
        "circuit_id": [1] * 24 + [2] * 24,
        "t_stamp": list(times) * 2,
        "power_signed": list(range(24)) * 2,
    })
    fig, ax = P.plot_aggregation_check(frame, candidate_circuit_ids=[1], component_circuit_ids=[2])
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[0] == "09:00", f"got {labels!r} -- DateFormatter is likely missing tz=C.FIXED_OFFSET"
