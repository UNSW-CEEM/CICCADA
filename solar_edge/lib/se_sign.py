"""
Reactive-power sign convention: evidence and consequences.
==========================================================

Deliverable D8.

The single-phase convention is settled: SolarEdge reports reactive power in the
LOAD convention (positive = absorbing), so ingest multiplies by -1 to reach the
CICCADA generator convention. That was established empirically and confirmed on
12 Aug 2026, and it is locked in ``se_config.REACTIVE_POWER_SIGN``.

The three-phase cohort is NOT settled, and this module exists to settle it.

The question
-----------
D6 found the two cohorts moving in opposite reactive directions, yet both showing
a |Q| minimum at 230-235 V -- almost exactly the AS/NZS 4777.2 deadband at
220-240 V. Two near mirror-image curves. Either

  (A) three-phase inverters report reactive power with the opposite polarity, and
      flipping the sign makes them broadly conformant; or
  (B) three-phase inverters genuinely respond in the wrong direction, which is a
      substantial conformance finding in its own right.

Why it cannot be waved through
------------------------------
Under the current convention the three-phase cohort scores 81.7% reduced
non-conformance, dominated by ``Q_adverse``. Getting this wrong therefore either
invents a fleet-wide non-conformance across 415 sites, or erases a real one. It is
not a rounding decision.

What this module provides
-------------------------
``site_response_classification``  per-site direction of response, so the question
                                  becomes "how many sites" rather than "what does
                                  the median do".
``deadband_shape``                where each cohort's |Q| minimum sits, and how
                                  sharply it rises either side.
``sign_flip_sensitivity``         conformance scored both ways, so the cost of
                                  being wrong is a number rather than a worry.

None of these can prove (A) or (B) on their own. They are assembled so that a
decision can be made on evidence, and so that whichever way it goes, the reasoning
is on the record.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from solar_edge.config import se_config as C
from solar_edge.lib import se_contract as contract
from solar_edge.lib import se_params

__all__ = [
    "site_response_classification",
    "deadband_shape",
    "sign_flip_sensitivity",
    "sign_evidence_summary",
]

_A = C.as4777()


def site_response_classification(
    con: duckdb.DuckDBPyConnection,
    config=None,
    low_v: float = 235.0,
    high_v: float = 250.0,
    min_samples: int = 50,
    min_delta_kvar: float = 0.02,
) -> pd.DataFrame:
    """
    Classify each site by the DIRECTION of its reactive response to voltage.

    For every site with enough observations either side, compare median Q below
    ``low_v`` against median Q above ``high_v``. In the CICCADA generator
    convention a conforming inverter absorbs more as voltage rises, so
    ``delta = Q_high - Q_low`` should be NEGATIVE.

    ``min_delta_kvar`` separates a real response from noise: sites moving less
    than this are classed inactive rather than being forced into a direction.

    Per-site classification matters because a fleet median can be dominated by a
    minority of strong responders. "How many sites move which way" is the question
    a reviewer will ask.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "i")

    return con.execute(
        f"""
        WITH d AS (
            SELECT i.site_alias, s.is_three_phase, s.state,
                   {v} AS V, i.Q_kvar, i.P_kW
            FROM se_interval i
            {contract.cohort_join_sql('i')}
            WHERE {contract.cohort_where_sql(config)}
              AND i.P_kW > 0.2 AND i.Q_kvar IS NOT NULL
              AND {v} BETWEEN 200 AND 270
        ),
        agg AS (
            SELECT site_alias,
                   any_value(is_three_phase)                       AS is_three_phase,
                   any_value(state)                               AS state,
                   median(Q_kvar) FILTER (WHERE V < {low_v})       AS q_low,
                   median(Q_kvar) FILTER (WHERE V > {high_v})      AS q_high,
                   count(*) FILTER (WHERE V < {low_v})             AS n_low,
                   count(*) FILTER (WHERE V > {high_v})            AS n_high
            FROM d GROUP BY site_alias
        )
        SELECT *,
               q_high - q_low AS delta_q_kvar,
               CASE
                   WHEN abs(q_high - q_low) < {min_delta_kvar} THEN 'inactive'
                   WHEN q_high < q_low THEN 'conforming direction (absorbs more)'
                   ELSE 'adverse direction (supplies more)'
               END AS response_class
        FROM agg
        WHERE n_low >= {min_samples} AND n_high >= {min_samples}
        """
    ).df()


def deadband_shape(
    con: duckdb.DuckDBPyConnection, config=None, bin_v: float = 2.5
) -> pd.DataFrame:
    """
    Where each cohort's |Q| minimum sits.

    The AS/NZS 4777.2 Australia A deadband runs 220-240 V, and an inverter
    implementing the curve should show minimum reactive magnitude inside it,
    rising on both sides.

    This is the evidence that distinguishes a sign-reporting difference from a
    genuinely adverse response. A cohort responding backwards has no reason to
    reproduce the standard's own deadband geometry; a cohort reporting the
    opposite polarity reproduces it exactly, inverted.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "i")

    return con.execute(
        f"""
        SELECT CASE WHEN s.is_three_phase THEN 'three-phase' ELSE 'single-phase' END AS cohort,
               floor({v} / {bin_v}) * {bin_v}          AS v_bin,
               count(*)                                 AS n_intervals,
               round(median(i.Q_kvar), 4)               AS median_Q_kvar,
               round(median(abs(i.Q_kvar)), 4)          AS median_abs_Q_kvar,
               round(median(i.P_kW), 3)                 AS median_P_kW
        FROM se_interval i
        {contract.cohort_join_sql('i')}
        WHERE {contract.cohort_where_sql(config)}
          AND i.P_kW > 0.5 AND i.Q_kvar IS NOT NULL
          AND {v} BETWEEN 210 AND 262
        GROUP BY 1, 2 HAVING count(*) >= 500
        ORDER BY 1, 2
        """
    ).df()


def sign_flip_sensitivity(
    con: duckdb.DuckDBPyConnection, config=None, params=None
) -> pd.DataFrame:
    """
    Score three-phase Volt-VAr conformance both ways.

    Runs the full D9 scoring on the three-phase cohort as stored, then again with
    ``Q_kvar`` negated, and reports both. The difference is the cost of getting the
    convention wrong, expressed in the categories that would actually be published.

    Reading it: if flipping moves the bulk of intervals out of ``Q_adverse`` and
    into the shortfall or near-conformant bands, hypothesis (A) is supported -- the
    cohort is behaving like the single-phase one, just reported inverted. If
    flipping merely swaps one implausible picture for another, (B) survives.
    """
    from solar_edge.lib import se_conformance as cf

    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()
    three_phase = config.with_changes(phase_cohort="three")

    rows = []
    for label, flip in (("as stored", False), ("sign flipped", True)):
        sql = cf.voltvar_interval_sql(three_phase, params)
        if flip:
            # Negate Q at source, before any scoring, so the whole CTE chain --
            # tolerance band, capability clamp, Q_impact -- sees the flipped value.
            sql = sql.replace("i.Q_kvar,", "-i.Q_kvar AS Q_kvar,", 1)

        counts = ",\n                   ".join(
            f"count(*) FILTER (WHERE {c} > 0) AS {c}" for c in cf.Q_CATEGORIES
        )
        frame = con.execute(
            f"""
            WITH scored AS ({sql})
            SELECT count(*) FILTER (WHERE capability_assessable = 1) AS assessable,
                   {counts},
                   count(*) FILTER (WHERE Q_kvar < 0) AS absorbing_intervals
            FROM scored
            """
        ).df().iloc[0]

        assessable = int(frame.assessable)
        reduced = sum(int(frame[c]) for c in cf.REDUCED_NONCONF)
        rows.append({
            "scenario": label,
            "assessable_intervals": assessable,
            **{c: int(frame[c]) for c in cf.Q_CATEGORIES},
            "reduced_nonconf_pct": round(100 * reduced / assessable, 2) if assessable else None,
            "absorbing_intervals": int(frame.absorbing_intervals),
        })
    return pd.DataFrame(rows)


def sign_evidence_summary(
    classification: pd.DataFrame, shape: pd.DataFrame, flip: pd.DataFrame
) -> pd.DataFrame:
    """
    Assemble the three strands into one table a reviewer can read in 30 seconds.

    Deliberately reports evidence rather than a verdict. The decision needs
    SolarEdge documentation or a site with known ground truth; this narrows what
    that documentation has to explain.
    """
    rows = []

    for cohort, group in classification.groupby("is_three_phase"):
        name = "three-phase" if cohort else "single-phase"
        total = len(group)
        for label, count in group.response_class.value_counts().items():
            rows.append({
                "evidence": "per-site response direction",
                "cohort": name,
                "finding": label,
                "value": f"{count} of {total} sites ({100 * count / total:.1f}%)",
            })

    for cohort, group in shape.groupby("cohort"):
        low = group.loc[group.median_abs_Q_kvar.idxmin()]
        rows.append({
            "evidence": "deadband location",
            "cohort": cohort,
            "finding": "voltage of minimum median |Q|",
            "value": f"{low.v_bin:.1f} V (|Q| = {low.median_abs_Q_kvar:.4f} kvar)",
        })
        rows.append({
            "evidence": "deadband location",
            "cohort": cohort,
            "finding": "AS/NZS 4777.2 deadband for reference",
            "value": f"{_A['VVAR']['V2']:.0f}-{_A['VVAR']['V3']:.0f} V",
        })

    for _, row in flip.iterrows():
        rows.append({
            "evidence": "sign-flip sensitivity",
            "cohort": "three-phase",
            "finding": f"reduced non-conformance, {row.scenario}",
            "value": f"{row.reduced_nonconf_pct}%  "
                     f"(Q_adverse {row.Q_adverse:,} intervals)",
        })

    return pd.DataFrame(rows)
