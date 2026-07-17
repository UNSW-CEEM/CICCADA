"""
Data calc-write pipeline: Stage 2. Volt-Watt conformance
=======================================================

Produces `conformance_voltvar_v2`:
- one row per (site_id, AEST date, day/night) with summed non-conformance magnitude, 
- the five Q_impact category buckets, and 
- the Volt-VAr-attributed active-power curtailment.

CAPACITY CONVENTION
------------------------------------------------
The standard refers to rated apparent power. 
This dataset has no verified manufacturer ``S_rated`` field, so each run must declare its proxy:

  * ``rating_basis`` scales required Q, the +/-4% band, the 20% assessability rule, and Figure 2.1. It defaults to provider ``ac_capacity_kw``.
  * ``empirical_limit_basis`` drives only the observed apparent-limit symptom. It defaults to empirical ``S_99``.

An all-``S_99`` run is a labelled sensitivity analysis, not evidence that ``S_99`` is manufacturer nameplate.

NAMING THRESHOLDS
-----------------------------------------------------

  Q_impact range | Original (Mileston3) | This module             | Meaning
  ---------------|----------------------|-------------------------|--------------------
  < -0.1         | Q_adverse            | Q_adverse               | wrong direction
  -0.1 .. 0.1    | Q_inactive           | Q_inactive              | no response
  0.1 .. 0.9     | Q_minor_deviation    | Q_significant_shortfall | responded, but far short
  0.9 .. 1.1     | Q_major_deficit      | Q_near_conformant       | essentially conformant
  > 1.1          | Q_major_surplus      | Q_major_surplus         | over-response

REDUCED NON-CONFORMANCE
--------------------------------------
    reduced = Q_adverse + Q_inactive + Q_significant_shortfall

Q_near_conformant (0.9-1.1) is EXCLUDED: those inverters deliver 90-110% of the
required reactive power. Milestone 3 included that band and excluded the
shortfall band -- an artefact of the swapped names (R4). This module does not
reproduce that. Figures will not match Milestone 3's Volt-VAr reduced
non-conformance rate, and should not.

SAFE-BY-DEFAULT
---------------
Writes to `conformance_voltvar_v2`. Original `conformance_voltvar` untouched.
"""

from as4777_curves import (
    vvar_required_q_sql,
    q_cap_absorbing_sql,
    q_conformance_floor_absorbing_sql,
    tol_kw_sql,
)

from stage2_common import (
    TABLE_SUFFIX, 
    WAREHOUSE, 
    UNCURTAILED,
    aest_month_window, 
    partition_predicate, 
    uncurtailed_partition_predicate,
    site_agg_cte, 
    temporal_cols, 
    run_months,
)
from ciccada_config import AS4777
from pipeline_options import capacity_column

TARGET = f"conformance_voltvar{TABLE_SUFFIX}"

# Q_impact classification thresholds
THR1, THR2, THR3, THR4 = -0.1, 0.1, 0.9, 1.1

# Volt-VAr absorbing zone (AS/NZS 4777.2 Australia A)

# V3 = 240.0      # absorption begins
# VW_V1 = 253.0   # Volt-Watt begins -- above this, P reduction is confounded

VVAR_V3 = AS4777["VVAR"]["V3"]
print(f"Using VVAR_V3 = {VVAR_V3} V (AS4777.2:2020 Australia A)")
VW_V1 = AS4777["VW"]["V1"]
print(f"Using VW_V1 = {VW_V1} V (AS4777.2:2020 Australia A)")
QCAP_P_MIN = AS4777["QCAP"]["P_MIN"]
print(
    f"Using QCAP_P_MIN = {QCAP_P_MIN:.0%} of rated apparent power "
    "(capacity proxy selected and labelled at run time)"
)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
def create_table(aq, database, target=TARGET):
    """
    Drop & recreate the empty target table. 
    Run once before loading.

    Partition columns (year, month) sit LAST in the column list and are
    identity-partitione
    The INSERT below is positional, so its final SELECT must list columns in exactly this order.
    """
    aq(f"DROP TABLE IF EXISTS {target}", database=database)
    aq(f"""
        CREATE TABLE {target} (
            site_id                            BIGINT,
            day                                INT,
            day_night                          STRING,
            P_kW_sum                           DOUBLE,
            nonconformance_voltvar_sum         DOUBLE,
            Q_adverse_sum                      DOUBLE,
            Q_inactive_sum                     DOUBLE,
            Q_significant_shortfall_sum        DOUBLE,
            Q_near_conformant_sum              DOUBLE,
            Q_major_surplus_sum                DOUBLE,
            curtailment_voltvar_sum            DOUBLE,
            nonconformance_voltvar_red_sum     DOUBLE,
            nonconformance_voltvar_count       BIGINT,
            Q_adverse_count                    BIGINT,
            Q_inactive_count                   BIGINT,
            Q_significant_shortfall_count      BIGINT,
            Q_near_conformant_count            BIGINT,
            Q_major_surplus_count              BIGINT,
            curtailment_voltvar_count          BIGINT,
            nonconformance_voltvar_red_count   BIGINT,
            curtailment_eligible_count         BIGINT,
            null_uncurtailed_P_count           BIGINT,
            exposed_count                      BIGINT,
            low_power_count                    BIGINT,
            low_power_exposed_count            BIGINT,
            all_intervals_count                BIGINT,
            total_count                        BIGINT,
            rating_basis                       STRING,
            empirical_limit_basis              STRING,
            capability_profile                 STRING,
            voltage_aggregation                STRING,
            flex_selection                     STRING,
            year                               INT,
            month                              INT
        )
        PARTITIONED BY (year, month)
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{target}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """, database=database)
    return f"Created empty {target}"

def _insert_sql(
    partitions, 
    utc_start, 
    utc_end, 
    part_filter, 
    exclude_flex=None, *,
    target=TARGET, 
    uncurtailed=UNCURTAILED,
    rating_basis="ac_capacity_kw", 
    empirical_limit_basis="s_99",
    capability_profile="review_corrected",
    voltage_aggregation="avg", 
    flex_selection="exclude",
):
    data = site_agg_cte(
        partitions,
        utc_start,
        utc_end,
        part_filter,
        exclude_flex=exclude_flex,
        with_reactive=True,
        flex_selection=flex_selection,
        voltage_aggregation=voltage_aggregation,
    )

    # Both the required-Q curve and Figure 2.1 minimum capability are normative
    # quantities based on S_rated. The selected basis is recorded in output.
    rating_col = capacity_column(rating_basis)
    empirical_col = capacity_column(empirical_limit_basis)
    q_required = vvar_required_q_sql("V", rating_col)

    ### Q-CAP ###
    qcap_p_min = AS4777["QCAP"]["P_MIN"]

    if capability_profile == "review_corrected":
        # Reactive-power priority above 80%; below 20% is reported separately.
        q_cap = q_conformance_floor_absorbing_sql("P_kW", rating_col)
        assessable_sql = f"""
            CASE
                WHEN abs(P_kW) >= {qcap_p_min} * {rating_col}
                THEN 1
                ELSE 0
            END
        """.strip()

    elif capability_profile == "hossein_m3":
        # Hossein's Figure 2.1 implementation:
        # zero below 20%, flat/0.8-PF branches, then the shrinking fixed-P circle.
        # He did not remove low-power intervals from conformance assessment.
        q_cap = q_cap_absorbing_sql("P_kW", rating_col)
        assessable_sql = "1"

    else:
        raise ValueError(
            "capability_profile must be "
            "'review_corrected' or 'hossein_m3'"
        )

    tol = tol_kw_sql(rating_col)

    unc_pred = uncurtailed_partition_predicate(partitions, alias="a")

    return f"""
    INSERT INTO {target}
    WITH
    {data},

    -- Required Q from Clause 3.3 and the minimum capability permitted by
    -- Clause 2.6 / Figure 2.1.
    required_q AS (
        SELECT
            site_id,
            t_stamp,
            P_kW,
            Q_kvar,
            V,
            ac_capacity_kw,
            S_99,
            {q_required} AS Q_voltvar,
            {q_cap} AS Q_cap_absorbing,
            {assessable_sql} AS capability_assessable
        FROM data
    ),

    -- +/-4% of the selected and output-labelled rating basis.
    tol_band AS (
        SELECT
            *,
            -Q_cap_absorbing AS Q_cap_supplying,
            Q_voltvar + {tol} AS Q_voltvar_max,
            Q_voltvar - {tol} AS Q_voltvar_min
        FROM required_q
    ),

    -- Clamp the response band to the minimum reactive-power capability that
    -- Figure 2.1 requires at the measured active-power level.
    clamped AS (
        SELECT
            *,
            CASE
                WHEN Q_voltvar_max < 0
                THEN greatest(
                    Q_voltvar_max,
                    Q_cap_absorbing + {tol}
                )
                ELSE Q_voltvar_max
            END AS Q_max_final,

            CASE
                WHEN Q_voltvar_min > 0
                THEN least(
                    Q_voltvar_min,
                    Q_cap_supplying - {tol}
                )
                ELSE Q_voltvar_min
            END AS Q_min_final
        FROM tol_band
    ),

    -- Normalised response score:
    --   1.0 = response at the nearest permitted band edge
    --   0.0 = no response
    --   < 0 = response in the wrong direction
    --
    -- Below 20% of the rated-power proxy, Q_impact is NULL because Figure 2.1
    -- provides no quantified minimum capability against which to assess it.
    q_impact AS (
        SELECT
            site_id,
            t_stamp,
            P_kW,
            Q_kvar,
            V,
            ac_capacity_kw,
            S_99,
            capability_assessable,
            Q_max_final,
            Q_min_final,

            CASE
                WHEN capability_assessable = 0 THEN NULL
                ELSE
                    CASE
                        WHEN abs(Q_kvar) / (abs(Q_max_final) + 1e-9)
                          <= abs(Q_kvar) / (abs(Q_min_final) + 1e-9)
                        THEN
                            (
                                CASE
                                    WHEN Q_max_final + Q_min_final = 0 THEN 1
                                    ELSE sign(Q_max_final) * sign(Q_kvar)
                                END
                            )
                            * (
                                abs(Q_kvar)
                                / (abs(Q_max_final) + 1e-9)
                            )
                        ELSE
                            (
                                CASE
                                    WHEN Q_max_final + Q_min_final = 0 THEN 1
                                    ELSE sign(Q_min_final) * sign(Q_kvar)
                                END
                            )
                            * (
                                abs(Q_kvar)
                                / (abs(Q_min_final) + 1e-9)
                            )
                    END
            END AS Q_impact
        FROM clamped
    ),

    -- Five mutually exclusive Q_impact buckets.
    --
    -- The value stored in each bucket is the kvar distance to the nearest edge
    -- of the permitted tolerance/capability band. Intervals below 20% of the
    -- rated-power proxy are retained but cannot enter a conformance bucket.
    classified AS (
        SELECT
            q.site_id,
            q.t_stamp,
            q.P_kW,
            q.Q_kvar,
            q.V,
            q.S_99,
            q.capability_assessable,
            q.Q_impact,

            CASE
                WHEN q.capability_assessable = 1
                  AND (
                      q.Q_kvar < q.Q_min_final
                      OR q.Q_kvar > q.Q_max_final
                  )
                  AND q.Q_impact < {THR1}
                THEN least(
                    abs(q.Q_kvar - q.Q_min_final),
                    abs(q.Q_kvar - q.Q_max_final)
                )
                ELSE 0
            END AS Q_adverse,

            CASE
                WHEN q.capability_assessable = 1
                  AND (
                      q.Q_kvar < q.Q_min_final
                      OR q.Q_kvar > q.Q_max_final
                  )
                  AND q.Q_impact >= {THR1}
                  AND q.Q_impact <= {THR2}
                THEN least(
                    abs(q.Q_kvar - q.Q_min_final),
                    abs(q.Q_kvar - q.Q_max_final)
                )
                ELSE 0
            END AS Q_inactive,

            CASE
                WHEN q.capability_assessable = 1
                  AND (
                      q.Q_kvar < q.Q_min_final
                      OR q.Q_kvar > q.Q_max_final
                  )
                  AND q.Q_impact > {THR2}
                  AND q.Q_impact < {THR3}
                THEN least(
                    abs(q.Q_kvar - q.Q_min_final),
                    abs(q.Q_kvar - q.Q_max_final)
                )
                ELSE 0
            END AS Q_significant_shortfall,

            CASE
                WHEN q.capability_assessable = 1
                  AND (
                      q.Q_kvar < q.Q_min_final
                      OR q.Q_kvar > q.Q_max_final
                  )
                  AND q.Q_impact >= {THR3}
                  AND q.Q_impact <= {THR4}
                THEN least(
                    abs(q.Q_kvar - q.Q_min_final),
                    abs(q.Q_kvar - q.Q_max_final)
                )
                ELSE 0
            END AS Q_near_conformant,

            CASE
                WHEN q.capability_assessable = 1
                  AND (
                      q.Q_kvar < q.Q_min_final
                      OR q.Q_kvar > q.Q_max_final
                  )
                  AND q.Q_impact > {THR4}
                THEN least(
                    abs(q.Q_kvar - q.Q_min_final),
                    abs(q.Q_kvar - q.Q_max_final)
                )
                ELSE 0
            END AS Q_major_surplus,

            -- Empirical apparent-power-limit symptom. This is deliberately
            -- separate from rating_basis and is labelled in the output.
            CASE
                WHEN q.V > {VVAR_V3}
                  AND q.V <= {VW_V1}
                  AND q.Q_kvar < 0
                  AND sqrt(
                      power(q.Q_kvar, 2) + power(q.P_kW, 2)
                  ) >= q.{empirical_col}
                  AND a.uncurtailed_P > q.P_kW
                THEN a.uncurtailed_P - q.P_kW
                ELSE NULL
            END AS curtailment_voltvar,

            CASE
                WHEN q.V > {VVAR_V3}
                  AND q.V <= {VW_V1}
                  AND q.Q_kvar < 0
                  AND sqrt(
                      power(q.Q_kvar, 2) + power(q.P_kW, 2)
                  ) >= q.{empirical_col}
                THEN 1
                ELSE 0
            END AS curtailment_eligible,

            CASE
                WHEN q.V > {VVAR_V3}
                  AND q.V <= {VW_V1}
                  AND q.Q_kvar < 0
                  AND sqrt(
                      power(q.Q_kvar, 2) + power(q.P_kW, 2)
                  ) >= q.{empirical_col}
                  AND a.uncurtailed_P IS NULL
                THEN 1
                ELSE 0
            END AS null_uncurtailed_P

        FROM q_impact q

        LEFT JOIN (
            SELECT
                a.site_id,
                a.t_stamp,
                a.uncurtailed_P
            FROM {uncurtailed} a
            WHERE {unc_pred}
        ) a
          ON q.site_id = a.site_id
         AND q.t_stamp = a.t_stamp
    ),

    -- All calendar fields are derived from fixed AEST.
    temporal AS (
        SELECT
            site_id,
            P_kW,
            V,
            capability_assessable,
            {temporal_cols('t_stamp')},

            Q_adverse,
            Q_inactive,
            Q_significant_shortfall,
            Q_near_conformant,
            Q_major_surplus,

            CASE
                WHEN capability_assessable = 1
                THEN
                    Q_adverse
                    + Q_inactive
                    + Q_significant_shortfall
                    + Q_near_conformant
                    + Q_major_surplus
                ELSE NULL
            END AS nonconformance_voltvar,

            curtailment_voltvar,
            curtailment_eligible,
            null_uncurtailed_P
        FROM classified
    )

    -- Column order must match create_table() exactly because the INSERT is
    -- positional.
    SELECT
        site_id,
        day_aest AS day,
        day_night,

        sum(P_kW) AS P_kW_sum,
        sum(nonconformance_voltvar)
            AS nonconformance_voltvar_sum,

        sum(Q_adverse)
            AS Q_adverse_sum,
        sum(Q_inactive)
            AS Q_inactive_sum,
        sum(Q_significant_shortfall)
            AS Q_significant_shortfall_sum,
        sum(Q_near_conformant)
            AS Q_near_conformant_sum,
        sum(Q_major_surplus)
            AS Q_major_surplus_sum,

        sum(curtailment_voltvar)
            AS curtailment_voltvar_sum,

        sum(Q_adverse)
            + sum(Q_inactive)
            + sum(Q_significant_shortfall)
            AS nonconformance_voltvar_red_sum,

        sum(
            CASE
                WHEN nonconformance_voltvar > 0 THEN 1
                ELSE 0
            END
        ) AS nonconformance_voltvar_count,

        sum(
            CASE
                WHEN Q_adverse > 0 THEN 1
                ELSE 0
            END
        ) AS Q_adverse_count,

        sum(
            CASE
                WHEN Q_inactive > 0 THEN 1
                ELSE 0
            END
        ) AS Q_inactive_count,

        sum(
            CASE
                WHEN Q_significant_shortfall > 0 THEN 1
                ELSE 0
            END
        ) AS Q_significant_shortfall_count,

        sum(
            CASE
                WHEN Q_near_conformant > 0 THEN 1
                ELSE 0
            END
        ) AS Q_near_conformant_count,

        sum(
            CASE
                WHEN Q_major_surplus > 0 THEN 1
                ELSE 0
            END
        ) AS Q_major_surplus_count,

        sum(
            CASE
                WHEN curtailment_voltvar > 0 THEN 1
                ELSE 0
            END
        ) AS curtailment_voltvar_count,

        sum(
            CASE
                WHEN Q_adverse > 0 THEN 1
                ELSE 0
            END
        )
        + sum(
            CASE
                WHEN Q_inactive > 0 THEN 1
                ELSE 0
            END
        )
        + sum(
            CASE
                WHEN Q_significant_shortfall > 0 THEN 1
                ELSE 0
            END
        ) AS nonconformance_voltvar_red_count,

        sum(curtailment_eligible)
            AS curtailment_eligible_count,

        sum(null_uncurtailed_P)
            AS null_uncurtailed_P_count,

        sum(
            CASE
                WHEN V > {VVAR_V3} THEN 1
                ELSE 0
            END
        ) AS exposed_count,

        sum(
            CASE
                WHEN capability_assessable = 0 THEN 1
                ELSE 0
            END
        ) AS low_power_count,

        sum(
            CASE
                WHEN capability_assessable = 0
                 AND V > {VVAR_V3}
                THEN 1
                ELSE 0
            END
        ) AS low_power_exposed_count,

        count(*) AS all_intervals_count,

        -- Only intervals at or above 20% of the rated-power proxy are included
        -- in the standards-based conformance denominator.
        count(nonconformance_voltvar) AS total_count,

        '{rating_basis}' AS rating_basis,
        '{empirical_limit_basis}' AS empirical_limit_basis,
        '{capability_profile}' AS capability_profile,
        '{voltage_aggregation}' AS voltage_aggregation,
        '{flex_selection}' AS flex_selection,

        year_aest AS year,
        month_aest AS month

    FROM temporal
    GROUP BY
        year_aest,
        month_aest,
        day_aest,
        day_night,
        site_id
    """

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_months_voltvar(aq, database, year, months, n_parts=8, parts=None,
                       exclude_flex=None, **run_options):
    """Load AEST months for one year. See stage2_common.run_months."""
    return run_months(aq, database, _insert_sql, year, months,
                      n_parts=n_parts, parts=parts, exclude_flex=exclude_flex,
                      **run_options)


def preview_sql(year=2024, month=1, n_parts=8, part=0,
                exclude_flex=None, **run_options):
    """Return the INSERT SQL for one slice WITHOUT running it. Read it first."""
    utc_start, utc_end, partitions = aest_month_window(year, month)
    return _insert_sql(
        partitions, utc_start, utc_end,
        f"site_id % {n_parts} = {part}",
        exclude_flex=exclude_flex,
        **run_options,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(aq, database, target=TARGET):
    """Sanity checks after loading."""
    shape = aq(f"""
        SELECT year, month,
               count(*)                  AS n_rows,
               count(DISTINCT site_id)   AS n_sites,
               count(DISTINCT day)       AS n_days
        FROM {target}
        GROUP BY year, month ORDER BY year, month
    """, database=database)

    # One row per (site, day, day_night). Anything > 1 means the AEST-month window logic leaked
    dupes = aq(f"""
        SELECT count(*) AS n_dupe_keys FROM (
            SELECT year, month, day, day_night, site_id
            FROM {target}
            GROUP BY year, month, day, day_night, site_id
            HAVING count(*) > 1
        )
    """, database=database)

    # Curtailment can never be negative, and Q_impact buckets must sum to nonconformance.
    coherence = aq(f"""
        SELECT
            sum(CASE WHEN curtailment_voltvar_sum < 0 THEN 1 ELSE 0 END) AS neg_curtailment_rows,
            sum(CASE WHEN abs(nonconformance_voltvar_sum
                              - (Q_adverse_sum + Q_inactive_sum + Q_significant_shortfall_sum
                                 + Q_near_conformant_sum + Q_major_surplus_sum)) > 1e-6
                     THEN 1 ELSE 0 END) AS bucket_mismatch_rows,
            sum(CASE WHEN total_count > all_intervals_count 
                   THEN 1 ELSE 0 END) AS count_inversion_rows,
            sum(CASE WHEN total_count + low_power_count <> all_intervals_count
                    THEN 1 ELSE 0 END) AS assessment_partition_mismatch_rows,
            sum(CASE WHEN low_power_exposed_count > exposed_count
                    THEN 1 ELSE 0 END) AS low_power_exposure_inversion_rows
        FROM {target}
    """, database=database)

    # How much of the curtailment zone has no counterfactual?
    cover = aq(f"""
        SELECT sum(curtailment_eligible_count) AS eligible,
               sum(null_uncurtailed_P_count)   AS no_counterfactual,
               round(100.0 * sum(null_uncurtailed_P_count)
                     / nullif(sum(curtailment_eligible_count), 0), 2) AS pct_missing
        FROM {target}
    """, database=database)

    print("Rows / sites / days per AEST month:")
    print(shape.to_string(index=False))
    print(f"\nDuplicate (year,month,day,day_night,site_id) keys (MUST be 0): "
          f"{int(dupes['n_dupe_keys'].iloc[0])}")
    print("\nCoherence (all MUST be 0):")
    print(coherence.to_string(index=False))
    print("\nCounterfactual coverage in the V-VAr curtailment zone:")
    print(cover.to_string(index=False))
    return shape, dupes, coherence, cover


def compare_to_original(aq, database, original="conformance_voltvar", target=TARGET):
    """
    Reconcile against original results. The original table is not touched.
    """
    new = aq(f"SELECT count(DISTINCT site_id) AS n FROM {target}", database=database)
    old = aq(f"SELECT count(DISTINCT site_id) AS n FROM {original}", database=database)
    only_old = aq(f"""
        SELECT count(*) AS n FROM (
            SELECT DISTINCT site_id FROM {original}
            EXCEPT
            SELECT DISTINCT site_id FROM {target}
        )
    """, database=database)
    flex = aq("""
        SELECT count(DISTINCT site_id) AS n FROM meta_up23c
        WHERE is_pv = True AND flex_export_detected = True
    """, database=database)
    print(f"sites in {target}:  {int(new['n'].iloc[0]):,}")
    print(f"sites in {original}: {int(old['n'].iloc[0]):,}")
    print(f"sites dropped:       {int(only_old['n'].iloc[0]):,}")
    print(f"flex-export sites (expected explanation for the drop): "
          f"{int(flex['n'].iloc[0]):,}")
    return new, old, only_old, flex
