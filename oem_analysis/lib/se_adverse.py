"""
adverse-direction sites.
======================================
taxonomy
------------
An adverse site is classified by whether its reactive MAGNITUDE tracks the
AS/NZS 4777.2 requirement. Magnitude is orientation-independent, which is what
makes it usable while the sign is unresolved:

``polarity_suspect``
    Adverse in direction, but |Q| sits within the +/-4% tolerance of the required
    curve across the ramp. An inverter delivering exactly the required quantity
    of reactive power, with the sign reported the other way.

``genuinely_adverse``
    Adverse in direction AND |Q| materially exceeds what the curve requires. The
    inverter is doing something substantial in the wrong direction.

``adverse_but_inactive``
    Adverse in direction but |Q| is small relative to the requirement.

"""

from __future__ import annotations

import duckdb
import pandas as pd

from oem_analysis.config import se_config as C
from oem_analysis.lib import se_contract as contract
from oem_analysis.lib import se_params

__all__ = [
    "classify_adverse_sites",
    "adverse_summary",
    "adverse_conformance_impact",
    "ADVERSE_CLASSES",
]

_A = C.as4777()

ADVERSE_CLASSES = (
    "polarity_suspect",
    "genuinely_adverse",
    "adverse_but_inactive",
    "not_adverse",
)


def classify_adverse_sites(
    con: duckdb.DuckDBPyConnection,
    config=None,
    v_lo: float = 241.0,
    v_hi: float = 253.0,
    min_p_kw: float = 0.5,
    min_intervals: int = 200,
    inactive_ratio: float = 0.25,
) -> pd.DataFrame:
    """
    Classify every site by direction AND magnitude of its reactive response.
    """
    config = (config or se_params.CONFIG).validate()
    q = contract.q_expr(config, "i")
    # Route through config.voltage_aggregation like every other query in the pipeline (default "mean") rather than hardcoding max-of-phases
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    tol = config.tolerance_fraction
    required = (
        f"-{_A['VVAR']['Q4']} * c.s_99 * ({v} - {_A['VVAR']['V3']}) / "
        f"({_A['VVAR']['V4']} - {_A['VVAR']['V3']})"
    )

    return con.execute(
        f"""
        WITH per_interval AS (
            SELECT i.site_alias,
                   s.is_three_phase,
                   s.state,
                   c.s_99,
                   {q}                     AS q_oriented,
                   ({required})            AS q_required,
                   {tol} * c.s_99          AS tol_kvar
            FROM se_interval i
            JOIN se_site s USING (site_alias)
            JOIN se_site_capacity c USING (site_alias)
            WHERE i.P_kW > {min_p_kw}
              AND {v} BETWEEN {v_lo} AND {v_hi}
              AND c.s_99 > 0 AND i.Q_kvar IS NOT NULL
        ),
        per_site AS (
            SELECT site_alias,
                   any_value(is_three_phase)                            AS is_three_phase,
                   any_value(state)                                     AS state,
                   any_value(s_99)                                      AS s_99,
                   count(*)                                             AS n_intervals,
                   median(q_oriented)                                   AS median_q_kvar,
                   median(abs(q_oriented) / nullif(abs(q_required), 0)) AS ratio_to_required,
                   median(abs(abs(q_oriented) - abs(q_required)))       AS mad_magnitude,
                   any_value(tol_kvar)                                  AS tol_kvar
            FROM per_interval
            GROUP BY site_alias HAVING count(*) >= {min_intervals}
        )
        SELECT site_alias,
               CASE WHEN is_three_phase THEN 'three-phase' ELSE 'single-phase' END AS cohort,
               state,
               n_intervals,
               round(s_99, 2)              AS s_99,
               round(median_q_kvar, 4)     AS median_q_kvar,
               round(ratio_to_required, 3) AS ratio_to_required,
               round(mad_magnitude, 4)     AS mad_magnitude_kvar,
               round(tol_kvar, 4)          AS tol_kvar,
               CASE
                   -- Absorbing (or near zero) in the configured orientation:
                   -- not adverse, whatever the magnitude.
                   WHEN median_q_kvar <= 0 THEN 'not_adverse'
                   -- Adverse in direction, but barely responding at all.
                   WHEN ratio_to_required < {inactive_ratio} THEN 'adverse_but_inactive'
                   -- Adverse in direction, magnitude matches the curve within
                   -- tolerance: the signature of an inverter following the curve
                   -- with the sign reported the other way.
                   WHEN mad_magnitude <= tol_kvar THEN 'polarity_suspect'
                   -- Adverse in direction and magnitude does not match: doing
                   -- something substantial, in the wrong direction.
                   ELSE 'genuinely_adverse'
               END AS adverse_class
        FROM per_site
        ORDER BY adverse_class, ratio_to_required DESC
        """
    ).df()


def adverse_summary(classified: pd.DataFrame) -> pd.DataFrame:
    """Counts per class and cohort"""
    meaning = {
        "not_adverse": "absorbing (or near zero) in the configured orientation",
        "polarity_suspect": "follows the curve in magnitude; sign reported inverted "
                            "-- a DATA-FORMAT finding, not a conformance one",
        "genuinely_adverse": "substantial response in the wrong direction "
                             "-- adverse under EITHER orientation, safe to report",
        "adverse_but_inactive": "barely responding; direction uninformative "
                                "-- belongs with the inactive population",
    }
    out = (
        classified.groupby(["adverse_class", "cohort"], as_index=False)
        .agg(n_sites=("site_alias", "count"),
             median_ratio=("ratio_to_required", "median"),
             median_s_99=("s_99", "median"))
    )
    out["median_ratio"] = out.median_ratio.round(3)
    out["interpretation"] = out.adverse_class.map(meaning)
    return out.sort_values(["adverse_class", "cohort"])


def adverse_conformance_impact(
    con: duckdb.DuckDBPyConnection, classified: pd.DataFrame, config=None, params=None
) -> pd.DataFrame:
    from oem_analysis.lib import se_conformance as cf

    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()

    con.register("_adverse_class", classified[["site_alias", "adverse_class"]])
    try:
        return con.execute(
            f"""
            WITH scored AS ({cf.voltvar_interval_sql(config, params)})
            SELECT coalesce(a.adverse_class, 'unclassified')      AS adverse_class,
                   count(*) FILTER (WHERE scored.capability_assessable = 1)
                                                                  AS assessable_intervals,
                   count(*) FILTER (WHERE scored.Q_adverse > 0)   AS adverse_intervals,
                   count(DISTINCT scored.site_alias)              AS n_sites
            FROM scored
            LEFT JOIN _adverse_class a USING (site_alias)
            GROUP BY 1 ORDER BY adverse_intervals DESC
            """
        ).df()
    finally:
        con.unregister("_adverse_class")
