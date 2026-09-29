"""
Nighttime-baseline PV/inverter signal estimation from NET meter data --
Phase 7 (`bms_sa_review/ami_analysis`), notebook 05's Section 5.

`ami_meter_residential`'s `P_kw`/`Q_kvar` are the `ac_load_net` circuit's
own reading, "already net-of-solar as landed" (see
`ami_dataset_column_reference.xlsx`'s `ami_meter`/`ami_raw` sheets) -- NOT
a pure house-load-only measurement, and NOT the inverter's own generation/
absorption either. `ami_raw`'s own documented derivation is:

    gross_house_load = net_reading + inverter_own_reading      (both P and Q)
    ==>  inverter_own_reading = gross_house_load - net_reading

`gross_house_load` (the household's PV-independent consumption) isn't
directly observable from `ami_meter_residential` alone -- there is no
separate "load-only" circuit in that table by construction (see
`ami_pv_phase.py`'s module docstring). This module approximates it with
each circuit's own NIGHTTIME median, when PV output is ~0 and
net ~= gross, and uses that to back out an estimate of the inverter's own
P/Q from the net reading:

    pv_estimate = nighttime_baseline - net_reading

This is a real approximation, not a measurement -- it assumes a
household's own load pattern doesn't shift drastically between night and
day. That's the honest cost of staying genuinely "blind" (AMI-only, no
access to a separately-metered PV circuit); validate it against
`ami_raw_phaseseparate`'s real PV-only signal (notebook Section 6) before
trusting it past a first-pass check.

Why this matters for Q specifically, not just P: the SAME formula applies
to both, but the physical consequence differs. Active-power generation
(positive `P_pv`) SUBTRACTS from net import -- an intuitive "net dips
during solar" shape. Reactive-power ABSORPTION (negative `Q_pv`, exactly
what AS/NZS 4777.2 requires whenever voltage is high) makes net Q reading
MORE POSITIVE under this same formula, since absorbing behaves like an
extra inductive load on the meter rather than offsetting anything -- this
is why a raw net Q trace can look "flipped" relative to the AS/NZS 4777.2
generator-convention required-Q curve even though nothing is actually
wrong with the data. Baseline-subtracting recovers the inverter's own
(generator-convention-comparable) signal in both cases.

Two functions, same split as `ami_conformance_plot.py`: a SQL-string
builder (`build_nighttime_baseline_sql`, no DuckDB import, a notebook runs
it) plus a pure pandas function (`estimate_pv_signal`) that is directly
testable.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["build_nighttime_baseline_sql", "estimate_pv_signal"]


def build_nighttime_baseline_sql(
    source_table: str, *,
    where_clause: str = "TRUE",
    nighttime_hours: tuple[int, int] = (0, 4),
    timestamp_column: str = "t_stamp",
    power_column: str = "P_kw",
    reactive_column: str = "Q_kvar",
) -> str:
    """
    One row per (`site_id`, `circuit_id`) in `source_table` (filtered by
    `where_clause` -- e.g. a `site_id`/`circuit_id` match, a scope-months
    filter, or both combined with `AND`): the median `power_column`/
    `reactive_column` during the AEST nighttime window
    [`nighttime_hours[0]`, `nighttime_hours[1]`) (fixed +10h offset, no
    DST -- same convention as everywhere else in this notebook).

    Aggregating over as much of `where_clause`'s scope as practical (a
    whole month, not one night) gives a far more stable baseline than a
    single night's reading -- exactly like `ami_pv_phase.py`'s own
    `circuit_signature` build in notebook Section 2, which this
    duplicates the shape of (P only there; this adds Q, since that's the
    whole point here).
    """
    return f"""
        WITH local_hour AS (
            SELECT site_id, circuit_id,
                   {power_column} AS p_value,
                   {reactive_column} AS q_value,
                   CAST((EXTRACT(hour FROM {timestamp_column}) + 10) % 24 AS INTEGER) AS hour_aest
            FROM {source_table}
            WHERE {where_clause}
        )
        SELECT
            site_id,
            circuit_id,
            median(p_value) FILTER (
                WHERE hour_aest >= {nighttime_hours[0]} AND hour_aest < {nighttime_hours[1]}
            ) AS nighttime_median_p_kw,
            median(q_value) FILTER (
                WHERE hour_aest >= {nighttime_hours[0]} AND hour_aest < {nighttime_hours[1]}
            ) AS nighttime_median_q_kvar
        FROM local_hour
        GROUP BY site_id, circuit_id
    """.strip()


def estimate_pv_signal(
    frame: pd.DataFrame,
    baseline_p_kw: float,
    baseline_q_kvar: float, *,
    power_column: str = "P_kw",
    reactive_column: str = "Q_kvar",
    output_power_column: str = "P_kw_pv_est",
    output_reactive_column: str = "Q_kvar_pv_est",
) -> pd.DataFrame:
    """
    Adds `output_power_column`/`output_reactive_column` to a COPY of
    `frame` -- one circuit's estimated own P/Q, back-derived from its net
    reading and a single (already-resolved) nighttime baseline pair:

        pv_estimate = baseline - net_reading

    `baseline_p_kw`/`baseline_q_kvar` are scalars (this operates on one
    circuit's rows at a time -- e.g. the output of
    `build_nighttime_baseline_sql` filtered/looked up for the one
    `site_id`/`circuit_id` being estimated), not a per-row lookup -- see
    the module docstring for why nighttime medians are used as the
    baseline and what that assumes.
    """
    result = frame.copy()
    result[output_power_column] = baseline_p_kw - result[power_column]
    result[output_reactive_column] = baseline_q_kvar - result[reactive_column]
    return result
