"""
AS/NZS 4777.2 Volt-VAr / Volt-Watt conformance, as DuckDB SQL -- Phase 7
(`bms_sa_review/ami_analysis`), notebooks 05/06.

Ported, not reimplemented, from this project's own already-validated
pipeline:

  bms_sa_review/shared/ciccada_config.py                    (AS4777 set-points)
  bms_sa_review/shared/as4777_curves.py                     (curve math, SQL-string form)
  bms_sa_review/data_calc_write/stage2_conformance/
      build_conformance_voltvar.py   (`_insert_sql`'s required_q -> tol_band
                                       -> clamped -> q_impact -> classified
                                       CTE chain, INCLUDING the Figure 2.1
                                       minimum-capability (QCAP) clamp)
      build_conformance_voltwatt.py  (`_insert_sql_basic`'s max-P / nonconformance)

One difference from those originals, a deliberate simplification, not an
oversight: interval-level output, not site-day-aggregated -- `ami_analysis`
notebooks already work at DuckDB scale directly, so there is no need to
pre-aggregate into an Athena table first the way the original pipeline does.

The Figure 2.1 minimum-capability (QCAP) clamp IS included (as of this
version), using `rating_column` as the `S_rated` stand-in -- the same proxy
caveat notebook 02's capacity comparison raised still applies (`ac_capacity_kw`/
`s_99` are documented proxies, not a validated manufacturer rating), but the
clamp itself is no longer skipped. Volt-Watt has no analogous capability
concept in the standard or in `build_conformance_voltwatt.py` -- the clamp is
Volt-VAr only.

Volt-VAr also gates on voltage exposure: `voltvar_exposed` is only TRUE in
the two bands the standard actually requires a response in (<=VVAR.V2
supplying, >=VVAR.V3 absorbing), FALSE in the 220-240V deadband. This is a
correction relative to `build_conformance_voltvar.py`'s own `total_count`
denominator, which never applied a voltage restriction at all (only the QCAP
`capability_assessable` gate) -- `exposed_count` existed there as a separate,
unused-for-conformance diagnostic column. Historical Athena results pulled
in notebook 05 Section 8 were computed the old way and are therefore not on
identical footing with this module's site-level 10% rule; see Section 8's
own markdown for that caveat.

Every function returns a SQL string built from `source_relation` (a table or
view name, or a parenthesised subquery) and assumes that relation carries at
least `V`, `P_kw`, `Q_kvar` (Volt-VAr only) and the chosen `rating_column`.
"""

from __future__ import annotations

# Import side effect only: `ami_config` calls `bootstrap_sys_path()` at
# module load, which puts `bms_sa_review` itself on `sys.path` -- required
# for the `shared.*`-style imports below to resolve, exactly as
# `as4777_curves.py`'s own `from shared.ciccada_config import AS4777` needs.
from bms_sa_review.synthetic_ami_creation.config import ami_config as _Config  # noqa: F401

from shared.as4777_curves import (  # noqa: E402
    q_cap_absorbing_sql,
    q_conformance_floor_absorbing_sql,
    q_impact_nearest_edge_sql,
    tol_kw_sql,
    vvar_required_q_sql,
    vw_max_p_sql,
)
from shared.ciccada_config import AS4777  # noqa: E402

__all__ = ["CAPABILITY_PROFILES", "build_voltvar_sql", "build_voltwatt_sql"]

_QIMP = AS4777["QIMP"]
_VW = AS4777["VW"]
_VVAR = AS4777["VVAR"]
_QCAP_P_MIN = AS4777["QCAP"]["P_MIN"]

#: The two Figure 2.1 minimum-capability interpretations this project has
#: implemented (same names, same meaning, as
#: `build_conformance_voltvar.py`'s own `capability_profile` argument):
#:
#: `review_corrected` -- reactive-power priority above 80% S_rated: the
#: minimum absorbing capability stays fixed at 0.60*S even near full active
#: -power output (a compliant inverter is expected to curtail P if it must,
#: to keep delivering Q). Intervals below 20% S_rated are excluded from
#: assessment entirely (Figure 2.1 specifies no quantified minimum there).
#: This is the project's own default (`AnalysisConfig.capability_profile`).
#:
#: `hossein_m3` -- the literal Figure 2.1 curve with no such priority: the
#: required capability tapers toward 0 as P approaches S_rated (the apparent
#: -power circle leaves less headroom for Q the closer P gets to S). Every
#: interval stays assessable, including below 20% S_rated.
CAPABILITY_PROFILES = ("review_corrected", "hossein_m3")


def build_voltvar_sql(
    source_relation: str, *,
    rating_column: str = "s_99",
    voltage_column: str = "V",
    reactive_column: str = "Q_kvar",
    power_column: str = "P_kw",
    capability_profile: str = "review_corrected",
) -> str:
    """
    Full Volt-VAr Q_impact classification, one row per row of
    `source_relation`, INCLUDING the Figure 2.1 minimum-capability (QCAP)
    clamp (see `CAPABILITY_PROFILES` for the two profiles) AND the standard's
    own voltage-exposure gate. Adds `Q_voltvar` (required Q), `Q_cap_absorbing`
    (the minimum absorbing capability at the measured `power_column`),
    `Q_min_final`/`Q_max_final` (the +/-4%-of-`rating_column` tolerance band,
    relaxed outward where the capability floor is narrower than the raw
    requirement), `capability_assessable`, `voltvar_exposed` (see below),
    `Q_impact` (normalised distance from the nearest permitted edge -- see
    `q_impact_nearest_edge`'s own docstring for the sign convention), and
    `voltvar_status`, one of: `Q_conformant`, `Q_adverse`, `Q_inactive`,
    `Q_significant_shortfall`, `Q_near_conformant`, `Q_major_surplus` (same
    five non-conformant buckets and thresholds as
    `build_conformance_voltvar.py`, from `AS4777["QIMP"]`), `Q_not_assessable`
    (only possible under `review_corrected`, below 20% S_rated), or
    `Q_not_exposed` (kept as explicit labels, not NULL, so both survive
    `value_counts`/`groupby` without silently vanishing).

    `voltvar_exposed` is `TRUE` when `voltage_column` sits in one of the two
    bands the standard actually requires a response in -- at/below `VVAR.V2`
    (the 207-220V supplying ramp, saturating below V1) or at/above `VVAR.V3`
    (the 240-258V absorbing ramp, saturating above V4) -- and `FALSE` in the
    220-240V deadband, where the curve already requires exactly Q=0 and
    scoring an interval there would be scoring against a requirement the
    standard never imposed. An interval only counts toward the site-level 10%
    rule when it is BOTH `capability_assessable` AND `voltvar_exposed`; the
    caller (notebook 05, Sections 6/7) is responsible for combining the two
    when it builds its own denominator, exactly as it already does for
    `capability_assessable` alone.
    """
    if capability_profile not in CAPABILITY_PROFILES:
        raise ValueError(f"capability_profile must be one of {CAPABILITY_PROFILES}")

    q_required = vvar_required_q_sql(voltage_column, rating_column)
    tol = tol_kw_sql(rating_column)
    exposed_sql = (
        f"CASE WHEN {voltage_column} <= {_VVAR['V2']} "
        f"OR {voltage_column} >= {_VVAR['V3']} THEN TRUE ELSE FALSE END"
    )

    if capability_profile == "review_corrected":
        q_cap = q_conformance_floor_absorbing_sql(power_column, rating_column)
        assessable_sql = (
            f"CASE WHEN abs({power_column}) >= {_QCAP_P_MIN} * {rating_column} "
            "THEN TRUE ELSE FALSE END"
        )
    else:  # hossein_m3
        q_cap = q_cap_absorbing_sql(power_column, rating_column)
        assessable_sql = "TRUE"

    q_impact_expr = q_impact_nearest_edge_sql(
        reactive_column, "Q_min_final", "Q_max_final", "capability_assessable",
    )

    return f"""
        WITH required_q AS (
            SELECT *,
                   ({q_required}) AS Q_voltvar,
                   ({q_cap}) AS Q_cap_absorbing,
                   ({assessable_sql}) AS capability_assessable,
                   ({exposed_sql}) AS voltvar_exposed
            FROM {source_relation}
        ),
        tol_band AS (
            SELECT *,
                   -Q_cap_absorbing AS Q_cap_supplying,
                   Q_voltvar + ({tol}) AS Q_voltvar_max,
                   Q_voltvar - ({tol}) AS Q_voltvar_min
            FROM required_q
        ),
        clamped AS (
            -- Relax the tolerance band outward to the minimum capability
            -- Figure 2.1 actually requires at this P -- never make the band
            -- stricter than the raw +/-4% requirement, only ever looser.
            SELECT *,
                   CASE WHEN Q_voltvar_max < 0
                        THEN greatest(Q_voltvar_max, Q_cap_absorbing + ({tol}))
                        ELSE Q_voltvar_max END AS Q_max_final,
                   CASE WHEN Q_voltvar_min > 0
                        THEN least(Q_voltvar_min, Q_cap_supplying - ({tol}))
                        ELSE Q_voltvar_min END AS Q_min_final
            FROM tol_band
        ),
        q_impact AS (
            SELECT *, ({q_impact_expr}) AS Q_impact
            FROM clamped
        )
        SELECT *,
            CASE
                WHEN NOT capability_assessable THEN 'Q_not_assessable'
                WHEN NOT voltvar_exposed THEN 'Q_not_exposed'
                WHEN {reactive_column} < Q_min_final OR {reactive_column} > Q_max_final THEN
                    CASE
                        WHEN Q_impact < {_QIMP['thr1']} THEN 'Q_adverse'
                        WHEN Q_impact <= {_QIMP['thr2']} THEN 'Q_inactive'
                        WHEN Q_impact < {_QIMP['thr3']} THEN 'Q_significant_shortfall'
                        WHEN Q_impact <= {_QIMP['thr4']} THEN 'Q_near_conformant'
                        ELSE 'Q_major_surplus'
                    END
                ELSE 'Q_conformant'
            END AS voltvar_status
        FROM q_impact
    """.strip()


def build_voltwatt_sql(
    source_relation: str, *,
    rating_column: str = "s_99",
    voltage_column: str = "V",
    power_column: str = "P_kw",
) -> str:
    """
    Volt-Watt max-allowed-P and nonconformance, one row per row of
    `source_relation`. Adds `max_P_volt_watt` (the curve limit plus the
    +/-4% tolerance) and `nonconformance_voltwatt` (`P_kw` above that limit,
    clipped at 0; NULL when `V` is at/below the exposure threshold `VW.V1`,
    matching `build_conformance_voltwatt.py`'s own "keep every interval, but
    only score above V1" convention). No capability clamp here -- Figure
    2.1's minimum-capability concept is specific to Volt-VAr reactive-power
    response; `build_conformance_voltwatt.py` has no equivalent for Volt-Watt.
    """
    max_p = vw_max_p_sql(voltage_column, rating_column)
    tol = tol_kw_sql(rating_column)
    return f"""
        WITH limits AS (
            SELECT *, ({max_p}) + ({tol}) AS max_P_volt_watt
            FROM {source_relation}
        )
        SELECT *,
            CASE WHEN {voltage_column} > {_VW['V1']}
                 THEN greatest(0, {power_column} - max_P_volt_watt)
                 ELSE NULL END AS nonconformance_voltwatt
        FROM limits
    """.strip()
