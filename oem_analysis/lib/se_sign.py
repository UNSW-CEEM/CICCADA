"""
Reactive-power sign convention: evidence and consequences.
==========================================================

Deliverable D8.

SUPERSEDED 13 Aug 2026 -- read this first.

The original conclusion here (a global flip to -1, from a 53-site sample) was
WRONG. The fleet-wide test in ``fleet_orientation_fit`` shows the reported sign is
inconsistent ACROSS the fleet and does not split by phase count: 213 sites fit the
value as delivered, 106 fit it flipped, 1,271 fit neither.

``se_config.REACTIVE_POWER_SIGN`` is therefore now +1 (as delivered), and the
residual 106 misoriented sites are triaged by ``se_adverse`` rather than pretended
away. The functions below remain useful as evidence; the framing in terms of a
single-phase / three-phase split does not.

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

from oem_analysis.config import se_config as C
from oem_analysis.lib import se_contract as contract
from oem_analysis.lib import se_params

C.bootstrap_sys_path()
from bms_sa_review.shared.as4777_curves import vvar_required_q_sql  # noqa: E402

__all__ = [
    "site_response_classification",
    "deadband_shape",
    "sign_flip_sensitivity",
    "fleet_orientation_fit",
    "fleet_sign_diagnosis",
    "sign_examples",
    "sign_example_errors",
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
    from oem_analysis.lib import se_conformance as cf

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


def fleet_orientation_fit(
    con: duckdb.DuckDBPyConnection,
    config=None,
    v_lo: float = 241.0,
    v_hi: float = 253.0,
    min_p_kw: float = 0.5,
    min_intervals: int = 200,
) -> pd.DataFrame:
    """
    For EVERY site in the fleet: does raw or stored Q fit the required curve?

    Runs over the whole fleet, not a sample. For each site, across the Volt-VAr
    ramp only (241-253 V, where the requirement is non-zero and Volt-Watt has not
    yet engaged), compute the median absolute deviation of measured Q from the
    required curve in BOTH orientations::

        Q_stored = -raw / 1000     (what the D2 flip produced)
        Q_raw    =  raw / 1000     (as OEM delivered it)

    ``verdict`` is the interesting column:

    ``raw fits (within tol)``
        The as-delivered orientation sits inside the AS/NZS 4777.2 +/-4% band
        while the flipped one does not. The site implements the curve, and the
        flip broke it.
    ``stored fits (within tol)``
        The reverse -- the flip was correct for this site.
    ``neither fits``
        The site is not following the curve in either orientation, so it carries
        no information about the convention. Expect most of the fleet here: D6
        found median power factor of 0.995-0.997.

    Sites that fit in ONE orientation and not the other are the evidence. Sites
    that fit in neither should be excluded from the argument entirely -- including
    them would dilute a clear signal with noise.
    """
    config = (config or se_params.CONFIG).validate()
    # Route through config.voltage_aggregation (default "mean") like the other
    # functions in this file (site_response_classification, deadband_shape,
    # sign_examples) instead of hardcoding max-of-phases -- this determination
    # feeds directly into the fleet's REACTIVE_POWER_SIGN convention, so a
    # max-vs-mean bias here could itself skew the orientation-fit verdict.
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    tol = _A["TOL_FRAC"]
    required = f"-{_A['VVAR']['Q4']} * c.s_99 * ({v} - {_A['VVAR']['V3']}) / " \
               f"({_A['VVAR']['V4']} - {_A['VVAR']['V3']})"

    return con.execute(
        f"""
        WITH per_interval AS (
            SELECT i.site_alias, s.is_three_phase, c.s_99,
                   i.Q_kvar                        AS q_stored,
                   -i.Q_kvar                       AS q_raw,
                   ({required})                    AS q_required,
                   {tol} * c.s_99                  AS tol_kvar
            FROM se_interval i
            JOIN se_site s USING (site_alias)
            JOIN se_site_capacity c USING (site_alias)
            WHERE i.P_kW > {min_p_kw}
              AND {v} BETWEEN {v_lo} AND {v_hi}
              AND c.s_99 > 0 AND i.Q_kvar IS NOT NULL
        ),
        per_site AS (
            SELECT site_alias,
                   any_value(is_three_phase)                       AS is_three_phase,
                   any_value(s_99)                                 AS s_99,
                   count(*)                                        AS n_intervals,
                   median(abs(q_stored - q_required))              AS mad_stored,
                   median(abs(q_raw    - q_required))              AS mad_raw,
                   any_value(tol_kvar)                             AS tol_kvar
            FROM per_interval
            GROUP BY site_alias HAVING count(*) >= {min_intervals}
        )
        SELECT site_alias,
               CASE WHEN is_three_phase THEN 'three-phase' ELSE 'single-phase' END AS cohort,
               n_intervals,
               round(s_99, 2)        AS s_99,
               round(mad_raw, 4)     AS mad_raw_kvar,
               round(mad_stored, 4)  AS mad_stored_kvar,
               round(tol_kvar, 4)    AS tol_kvar,
               CASE
                   WHEN mad_raw    <= tol_kvar AND mad_stored >  tol_kvar
                        THEN 'raw fits (within tol)'
                   WHEN mad_stored <= tol_kvar AND mad_raw    >  tol_kvar
                        THEN 'stored fits (within tol)'
                   WHEN mad_raw <= tol_kvar AND mad_stored <= tol_kvar
                        THEN 'both fit (uninformative)'
                   ELSE 'neither fits'
               END AS verdict
        FROM per_site
        ORDER BY cohort, mad_raw
        """
    ).df()


def fleet_sign_diagnosis(
    con: duckdb.DuckDBPyConnection,
    config=None,
    v_lo: float = 248.0,
    v_hi: float = 256.0,
    min_intervals: int = 500,
    strong_kvar: float = 1.0,
) -> pd.DataFrame:
    """
    THE decisive test, and the one that overturned the D2 assumption.

    For every site with enough observations in the upper Volt-VAr ramp, compare
    the MAGNITUDE of measured reactive power against the magnitude the AS/NZS
    4777.2 curve requires at the same voltage, scaled by the site's own ``s_99``.

    ``ratio_to_required`` near 1.0 means the site is delivering the required
    quantity of reactive power -- it is implementing the curve. That question is
    entirely independent of which SIGN the quantity was reported with, which is
    what makes it usable while the sign is in dispute.

    What it shows on this fleet
    ---------------------------
    Sites that demonstrably implement the curve appear with BOTH stored signs::

        cohort        sign group                n_sites   ratio (p25-p75)
        single-phase  stored NEGATIVE, strong        23   1.101 (0.87-1.41)
        single-phase  stored POSITIVE, strong        77   1.028 (1.017-1.035)
        three-phase   stored POSITIVE, strong        47   0.767 (0.64-0.87)
        (either)      weak, |Q| < 1 kvar            817   ~0.12

    Seventy-seven single-phase sites deliver 102.8% of the required magnitude with
    an interquartile range of 0.02. That tightness is a firmware signature, not a
    coincidence and not a misconfiguration -- those inverters are following the
    curve. Twenty-three more do the same with the opposite stored sign.

    The conclusion is therefore NOT the phase split assumed at D8. **The reported
    reactive sign is inconsistent across the fleet**, and no single global
    constant -- flip or no flip -- is correct for all sites.

    The circularity that blocks a code-only fix
    -------------------------------------------
    Per-site sign could be inferred by assuming each inverter absorbs above 240 V
    and orienting Q accordingly. That cannot be done in a CONFORMANCE study: it
    assumes the very behaviour under assessment, and would manufacture a 100%
    direction-conformance result by construction.

    So while the sign is unresolved:

      * **magnitude conformance is assessable** -- ``ratio_to_required`` is sound,
        and cleanly separates the ~150 sites implementing the curve from the ~800
        delivering about 12% of it;
      * **direction conformance is NOT assessable** -- ``Q_adverse`` versus
        conforming cannot be distinguished from a reporting-polarity difference.

    Resolving it needs OEM documentation or one site with known ground truth.
    """
    config = (config or se_params.CONFIG).validate()
    # Same fix as fleet_orientation_fit above: honour config.voltage_aggregation
    # instead of hardcoding max-of-phases.
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    required = f"-0.60 * c.s_99 * ({v} - 240.0) / 18.0"
    return con.execute(
        f"""
        WITH per_interval AS (
            SELECT i.site_alias, s.is_three_phase, i.Q_kvar,
                   {required} AS required_q
            FROM se_interval i
            JOIN se_site s USING (site_alias)
            JOIN se_site_capacity c USING (site_alias)
            WHERE i.P_kW > 0.5
              AND {v} BETWEEN {v_lo} AND {v_hi}
              AND c.s_99 > 0
        ),
        per_site AS (
            SELECT site_alias, is_three_phase,
                   median(Q_kvar)                                        AS med_q,
                   median(abs(Q_kvar) / nullif(abs(required_q), 0))       AS ratio,
                   count(*)                                              AS n
            FROM per_interval
            GROUP BY 1, 2 HAVING count(*) > {min_intervals}
        )
        SELECT CASE WHEN is_three_phase THEN 'three-phase' ELSE 'single-phase' END AS cohort,
               CASE WHEN med_q < -{strong_kvar} THEN 'stored NEGATIVE, strong'
                    WHEN med_q >  {strong_kvar} THEN 'stored POSITIVE, strong'
                    ELSE 'weak (|Q| < 1 kvar)' END                AS sign_group,
               count(*)                                           AS n_sites,
               round(median(ratio), 3)                            AS ratio_to_required,
               round(quantile_cont(ratio, 0.25), 3)               AS p25,
               round(quantile_cont(ratio, 0.75), 3)               AS p75
        FROM per_site
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).df()


def sign_examples(
    con: duckdb.DuckDBPyConnection,
    config=None,
    n_sites: int = 4,
    min_p_kw: float = 0.5,
    bin_v: float = 2.5,
) -> pd.DataFrame:
    """
    Measured reactive power against the AS/NZS 4777.2 requirement, per site,
    scored both as stored and sign-flipped.

    This is the demonstration the aggregate tables cannot give you: for individual
    three-phase sites, does flipping the sign move measured Q *toward* the required
    Volt-VAr curve or away from it?

    ``required_Q_kvar`` is the curve from ``vvar_required_q_sql`` scaled by the
    site's own ``s_99``. In the generator convention it is zero through the
    220-240 V deadband and falls negative above 240 V. So above 240 V:

      * if ``Q_flipped`` tracks toward ``required_Q_kvar`` while ``Q_as_stored``
        moves the opposite way, the stored sign is inverted for this cohort;
      * if neither tracks it, the site simply is not doing Volt-VAr, and the sign
        question is moot for that site.

    Returns interval-level medians binned by voltage, for the highest-response
    three-phase sites plus single-phase references.
    """
    config = (config or se_params.CONFIG).validate()
    v = contract.voltage_sql(config.voltage_aggregation, "i")
    required = vvar_required_q_sql("v_bin", "s_99")

    return con.execute(
        f"""
        WITH ranked AS (
            SELECT i.site_alias, s.is_three_phase,
                   count(*) AS n,
                   abs(median(i.Q_kvar) FILTER (WHERE {v} > 250)
                     - median(i.Q_kvar) FILTER (WHERE {v} < 235)) AS swing
            FROM se_interval i
            {contract.cohort_join_sql('i')}
            WHERE {contract.cohort_where_sql(config)}
              AND i.P_kW > {min_p_kw} AND {v} BETWEEN 200 AND 270
            GROUP BY 1, 2
            HAVING count(*) FILTER (WHERE {v} > 250) > 500
               AND count(*) FILTER (WHERE {v} < 235) > 500
        ),
        picked AS (
            SELECT site_alias, is_three_phase FROM ranked
            QUALIFY row_number() OVER (PARTITION BY is_three_phase ORDER BY swing DESC)
                    <= {n_sites}
        ),
        binned AS (
            SELECT p.site_alias, p.is_three_phase,
                   floor({v} / {bin_v}) * {bin_v}   AS v_bin,
                   count(*)                          AS n_intervals,
                   median(i.Q_kvar)                  AS q_as_stored,
                   median(-i.Q_kvar)                 AS q_flipped,
                   median(i.P_kW)                    AS med_P_kW,
                   any_value(c.s_99)                 AS s_99
            FROM se_interval i
            JOIN picked p USING (site_alias)
            LEFT JOIN se_site_capacity c USING (site_alias)
            WHERE i.P_kW > {min_p_kw} AND {v} BETWEEN 200 AND 270
            GROUP BY 1, 2, 3
            HAVING count(*) >= 200
        )
        SELECT site_alias,
               CASE WHEN is_three_phase THEN 'three-phase' ELSE 'single-phase' END AS cohort,
               v_bin, n_intervals,
               round(q_as_stored, 4)          AS Q_as_stored,
               round(q_flipped, 4)            AS Q_flipped,
               round(({required}), 4)         AS required_Q_kvar,
               round(med_P_kW, 3)             AS med_P_kW,
               round(s_99, 2)                 AS s_99
        FROM binned
        ORDER BY cohort, site_alias, v_bin
        """
    ).df()


def sign_example_errors(examples: pd.DataFrame, v_min: float = 240.0) -> pd.DataFrame:
    """
    Per site, how far measured Q sits from the required curve, both ways.

    Restricted to voltages above ``v_min``, because below 240 V the requirement is
    zero and both signs are equidistant from it -- averaging that region in would
    dilute the very comparison being made.

    CAVEAT -- this metric is close to degenerate, and ``fleet_sign_diagnosis`` is
    the one to trust. Required Q is large and negative above 240 V while most sites
    carry a small |Q| of either sign, so the negative orientation is nearly always
    "closer" regardless of what the site is doing. It is retained only to show
    per-site curves alongside the requirement, not as a discriminator.
    """
    hot = examples[examples.v_bin >= v_min].copy()
    hot["err_as_stored"] = (hot.Q_as_stored - hot.required_Q_kvar).abs()
    hot["err_flipped"] = (hot.Q_flipped - hot.required_Q_kvar).abs()

    out = (
        hot.groupby(["cohort", "site_alias"], as_index=False)
        .agg(n_bins=("v_bin", "count"),
             s_99=("s_99", "first"),
             mean_err_as_stored=("err_as_stored", "mean"),
             mean_err_flipped=("err_flipped", "mean"))
    )
    out["better"] = out.apply(
        lambda r: "flipped" if r.mean_err_flipped < r.mean_err_as_stored else "as stored",
        axis=1,
    )
    out["improvement_kvar"] = (out.mean_err_as_stored - out.mean_err_flipped).round(4)
    for col in ("mean_err_as_stored", "mean_err_flipped"):
        out[col] = out[col].round(4)
    return out.sort_values(["cohort", "improvement_kvar"], ascending=[True, False])


def sign_evidence_summary(
    classification: pd.DataFrame, shape: pd.DataFrame, flip: pd.DataFrame
) -> pd.DataFrame:
    """
    Assemble the three strands into one table a reviewer can read in 30 seconds.

    Deliberately reports evidence rather than a verdict. The decision needs
    OEM documentation or a site with known ground truth; this narrows what
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
