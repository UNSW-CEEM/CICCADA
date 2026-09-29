"""
Per-circuit diagnostic queries -- Phase 7 (`bms_sa_review/ami_analysis`),
notebook 05's phase-inference sanity checks (and any later notebook that
wants to eyeball a site's raw signal).

Ported in SHAPE, not copied verbatim, from
`dnsp_analysis/notebooks/02a_explore_canonical.ipynb`'s own daily-profile-by-
-phase plot and `quality` table -- same idea (median V/P/Q by local hour,
per group; null/zero counts per group), computed directly in DuckDB against
this project's own `ami_meter_*` schema instead of a pandas sample, since
this fleet is bigger than the one-site sample that notebook worked from.

Both functions are pure SQL-string builders, same convention as
`ami_conformance.py`: no DuckDB import here, a notebook builds the relation
and runs the string.
"""

from __future__ import annotations

__all__ = ["build_circuit_daily_profile_sql", "build_circuit_quality_sql"]


def build_circuit_daily_profile_sql(
    source_relation: str, *,
    voltage_column: str = "V",
    power_column: str = "P_kw",
    reactive_column: str = "Q_kvar",
    timestamp_column: str = "t_stamp",
) -> str:
    """
    One row per (`site_id`, `circuit_id`, `local_hour`) -- `local_hour` is
    AEST as a fractional hour (fixed +10 offset, no DST, matching this
    project's own `ciccada_config.FIXED_OFFSET` convention), the same
    `hour + minute/60` shape as `02a_explore_canonical.ipynb`'s own
    `timestamp_local.dt.hour + timestamp_local.dt.minute/60`. Adds median
    `voltage_column`/`power_column`/`reactive_column` per group -- plot one
    line per `circuit_id` against `local_hour` to see whether a circuit's
    own daily shape looks PV-like (a midday dip/dent in net `P_kw` if that
    circuit carries PV, on this project's "positive = import" convention).

    `source_relation` must carry at least `site_id`, `circuit_id`,
    `timestamp_column`, `voltage_column`, `power_column`, `reactive_column`
    -- typically a single-site, scope-filtered subquery, not the whole
    `ami_meter_residential` table (grouping by `local_hour` alone, across
    every site, would blend unrelated households' profiles together).
    """
    return f"""
        WITH local_hour AS (
            SELECT site_id, circuit_id,
                   {voltage_column} AS v_value,
                   {power_column} AS p_value,
                   {reactive_column} AS q_value,
                   CAST((EXTRACT(hour FROM {timestamp_column}) + 10) % 24 AS INTEGER)
                       + EXTRACT(minute FROM {timestamp_column}) / 60.0 AS local_hour
            FROM {source_relation}
        )
        SELECT
            site_id,
            circuit_id,
            local_hour,
            median(v_value) AS median_v,
            median(p_value) AS median_p_kw,
            median(q_value) AS median_q_kvar
        FROM local_hour
        GROUP BY site_id, circuit_id, local_hour
    """.strip()


def build_circuit_quality_sql(
    source_relation: str, *,
    voltage_column: str = "V",
    power_column: str = "P_kw",
    reactive_column: str = "Q_kvar",
    current_column: str | None = "current_a",
) -> str:
    """
    One row per (`site_id`, `circuit_id`) -- the same data-quality counters
    as `02a_explore_canonical.ipynb`'s own `quality` table (`rows`, `p_null`,
    `q_null`, `current_null`, `voltage_zero`), computed directly in DuckDB.
    Pass `current_column=None` if `source_relation` doesn't carry a current
    column (`current_null` comes back NULL in that case, not 0 -- "not
    checked", not "no nulls found").
    """
    current_null_expr = (
        f"count(*) - count({current_column})" if current_column else "CAST(NULL AS BIGINT)"
    )
    return f"""
        SELECT
            site_id,
            circuit_id,
            count(*) AS rows,
            count(*) - count({power_column}) AS p_null,
            count(*) - count({reactive_column}) AS q_null,
            {current_null_expr} AS current_null,
            sum(CASE WHEN {voltage_column} <= 0 THEN 1 ELSE 0 END) AS voltage_zero
        FROM {source_relation}
        GROUP BY site_id, circuit_id
    """.strip()
