"""
Data calc-write pipeline: Stage 2. Volt-Watt conformance
=======================================================

Produces `conformance_voltvar_v2`:
- one row per (site_id, AEST date, day/night) with summed non-conformance magnitude, 
- the five Q_impact category buckets, and 
- the Volt-VAr-attributed active-power curtailment.

RATED-VALUE CONVENTION
------------------------------------------------
  * The STANDARD's required-Q curve is defined against rated apparent power
    -> `ac_capacity_kw` (nameplate).
  * The inverter's CAPABILITY curve is a physical limit
    -> `s_99` (empirical 99th-percentile apparent power).
  * The +/-4% tolerance band is a fraction of NAMEPLATE.

NAMING THRESHOLDS
-----------------------------------------------------

  Q_impact range | Hossein's name       | This module            | Meaning
  ---------------|----------------------|-----------------------|--------------------
  < -0.1         | Q_adverse            | Q_adverse             | wrong direction
  -0.1 .. 0.1    | Q_inactive           | Q_inactive            | no response
  0.1 .. 0.9     | Q_minor_deviation    | Q_significant_shortfall | responded, but far short
  0.9 .. 1.1     | Q_major_deficit      | Q_near_conformant     | essentially conformant
  > 1.1          | Q_major_surplus      | Q_major_surplus       | over-response

SAFE-BY-DEFAULT
---------------
Writes to `conformance_voltvar_v2`. Original `conformance_voltvar` untouched.
"""

from shared.as4777_curves import (
    vvar_required_q_sql,
    q_cap_absorbing_sql,
    tol_kw_sql,
)
from stage2_common import (
    TABLE_SUFFIX, WAREHOUSE, UNCURTAILED,
    aest_month_window, partition_predicate, uncurtailed_partition_predicate,
    site_agg_cte, temporal_cols, run_months,
)

TARGET = f"conformance_voltvar{TABLE_SUFFIX}"

# Q_impact classification thresholds (Hossein's, unchanged -- only names changed)
THR1, THR2, THR3, THR4 = -0.1, 0.1, 0.9, 1.1

# Volt-VAr absorbing zone (AS/NZS 4777.2 Australia A)
V3 = 240.0      # absorption begins
VW_V1 = 253.0   # Volt-Watt begins -- above this, P reduction is confounded


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
def create_table(aq, database):
    """Drop & recreate the empty target table. Run once before loading.

    Partition columns (year, month) sit LAST in the column list and are
    identity-partitioned, mirroring build_structured_data.py. The INSERT below
    is positional, so its final SELECT must list columns in exactly this order.
    """
    aq(f"DROP TABLE IF EXISTS {TARGET}", database=database)
    aq(f"""
        CREATE TABLE {TARGET} (
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
            nonconformance_voltvar_red_alt_sum DOUBLE,
            nonconformance_voltvar_count       BIGINT,
            Q_adverse_count                    BIGINT,
            Q_inactive_count                   BIGINT,
            Q_significant_shortfall_count      BIGINT,
            Q_near_conformant_count            BIGINT,
            Q_major_surplus_count              BIGINT,
            curtailment_voltvar_count          BIGINT,
            nonconformance_voltvar_red_count   BIGINT,
            nonconformance_voltvar_red_alt_count BIGINT,
            curtailment_eligible_count         BIGINT,
            null_uncurtailed_P_count           BIGINT,
            exposed_count                      BIGINT,
            all_intervals_count                BIGINT,
            total_count                        BIGINT,
            year                               INT,
            month                              INT
        )
        PARTITIONED BY (year, month)
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{TARGET}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """, database=database)
    return f"Created empty {TARGET}"


# ---------------------------------------------------------------------------
# INSERT for one (AEST month x site-slice)
# ---------------------------------------------------------------------------
def _insert_sql(partitions, utc_start, utc_end, part_filter, exclude_flex=True):
    data = site_agg_cte(partitions, utc_start, utc_end, part_filter,
                        exclude_flex=exclude_flex, with_reactive=True)

    # R11: every curve comes from the keystone. R10: capability on S_99.
    q_required = vvar_required_q_sql("V", "ac_capacity_kw")
    q_cap      = q_cap_absorbing_sql("P_kW", "S_99")
    tol        = tol_kw_sql("ac_capacity_kw")

    unc_pred = uncurtailed_partition_predicate(partitions, alias="a")

    return f"""
    INSERT INTO {TARGET}
    WITH
    {data},

    -- required Q from the standard's curve + what the inverter can physically do
    required_q AS (
        SELECT site_id, t_stamp, P_kW, Q_kvar, V, ac_capacity_kw, S_99,
               {q_required} AS Q_voltvar,
               {q_cap}      AS Q_cap_absorbing
        FROM data
    ),

    -- +/-4% of nameplate around the required value
    tol_band AS (
        SELECT *,
               -Q_cap_absorbing        AS Q_cap_supplying,
               Q_voltvar + {tol}       AS Q_voltvar_max,
               Q_voltvar - {tol}       AS Q_voltvar_min
        FROM required_q
    ),

    -- do not penalise an inverter for failing to exceed its own S-circle
    clamped AS (
        SELECT *,
               CASE WHEN Q_voltvar_max < 0
                    THEN greatest(Q_voltvar_max, Q_cap_absorbing + {tol})
                    ELSE Q_voltvar_max END AS Q_max_final,
               CASE WHEN Q_voltvar_min > 0
                    THEN least(Q_voltvar_min, Q_cap_supplying - {tol})
                    ELSE Q_voltvar_min END AS Q_min_final
        FROM tol_band
    ),

    -- normalised response score: |Q_actual| / |Q_required|, signed by direction.
    -- 1.0 = exactly on the curve, 0 = no response, <0 = responding the wrong way.
    q_impact AS (
        SELECT site_id, t_stamp, P_kW, Q_kvar, V, ac_capacity_kw, S_99,
               Q_max_final, Q_min_final,
               CASE
                   WHEN abs(Q_kvar) / (abs(Q_max_final) + 1e-9)
                     <= abs(Q_kvar) / (abs(Q_min_final) + 1e-9)
                   THEN (CASE WHEN Q_max_final + Q_min_final = 0 THEN 1
                              ELSE sign(Q_max_final) * sign(Q_kvar) END)
                        * (abs(Q_kvar) / (abs(Q_max_final) + 1e-9))
                   ELSE (CASE WHEN Q_max_final + Q_min_final = 0 THEN 1
                              ELSE sign(Q_min_final) * sign(Q_kvar) END)
                        * (abs(Q_kvar) / (abs(Q_min_final) + 1e-9))
               END AS Q_impact
        FROM clamped
    ),

    -- five mutually exclusive buckets. The VALUE in each is the kvar shortfall
    -- (distance to the nearest edge of the tolerance band); 0 when inside band.
    classified AS (
        SELECT
            q.site_id, q.t_stamp, q.P_kW, q.Q_kvar, q.V, q.S_99, q.Q_impact,

            CASE WHEN (q.Q_kvar < q.Q_min_final OR q.Q_kvar > q.Q_max_final)
                  AND q.Q_impact < {THR1}
                 THEN least(abs(q.Q_kvar - q.Q_min_final), abs(q.Q_kvar - q.Q_max_final))
                 ELSE 0 END AS Q_adverse,

            CASE WHEN (q.Q_kvar < q.Q_min_final OR q.Q_kvar > q.Q_max_final)
                  AND q.Q_impact >= {THR1} AND q.Q_impact <= {THR2}
                 THEN least(abs(q.Q_kvar - q.Q_min_final), abs(q.Q_kvar - q.Q_max_final))
                 ELSE 0 END AS Q_inactive,

            -- was Q_minor_deviation (R4): 0.1-0.9 is a SIGNIFICANT shortfall
            CASE WHEN (q.Q_kvar < q.Q_min_final OR q.Q_kvar > q.Q_max_final)
                  AND q.Q_impact > {THR2} AND q.Q_impact < {THR3}
                 THEN least(abs(q.Q_kvar - q.Q_min_final), abs(q.Q_kvar - q.Q_max_final))
                 ELSE 0 END AS Q_significant_shortfall,

            -- was Q_major_deficit (R4): 0.9-1.1 is essentially CONFORMANT
            CASE WHEN (q.Q_kvar < q.Q_min_final OR q.Q_kvar > q.Q_max_final)
                  AND q.Q_impact >= {THR3} AND q.Q_impact <= {THR4}
                 THEN least(abs(q.Q_kvar - q.Q_min_final), abs(q.Q_kvar - q.Q_max_final))
                 ELSE 0 END AS Q_near_conformant,

            CASE WHEN (q.Q_kvar < q.Q_min_final OR q.Q_kvar > q.Q_max_final)
                  AND q.Q_impact > {THR4}
                 THEN least(abs(q.Q_kvar - q.Q_min_final), abs(q.Q_kvar - q.Q_max_final))
                 ELSE 0 END AS Q_major_surplus,

            -- R7 CURTAILMENT. Hossein flagged curtailment whenever V <= 253 and
            -- the site sat on its S-circle -- which fires across the whole
            -- deadband and the supplying zone, where Volt-VAr cannot be the
            -- cause. Restricted here to the absorbing zone, below the Volt-Watt
            -- knee, with Q actually being absorbed:
            --     240 < V <= 253      absorbing zone, no Volt-Watt confound
            --     Q_kvar < 0          inverter really is absorbing
            --     sqrt(P^2+Q^2) >= S_99   sitting on the apparent-power limit
            --     uncurtailed_P > P_kW    counterfactual says it lost energy
            -- Hard S-circle test, no tolerance band: matches Hossein and
            -- Method A. Conservative; noted as such.
            CASE WHEN q.V > {V3} AND q.V <= {VW_V1}
                  AND q.Q_kvar < 0
                  AND sqrt(power(q.Q_kvar, 2) + power(q.P_kW, 2)) >= q.S_99
                  AND a.uncurtailed_P > q.P_kW
                 THEN a.uncurtailed_P - q.P_kW
                 ELSE NULL END AS curtailment_voltvar,   -- R15: NULL, not 0

            -- interval sat in the curtailment-detection zone (denominator)
            CASE WHEN q.V > {V3} AND q.V <= {VW_V1}
                  AND q.Q_kvar < 0
                  AND sqrt(power(q.Q_kvar, 2) + power(q.P_kW, 2)) >= q.S_99
                 THEN 1 ELSE 0 END AS curtailment_eligible,

            -- R14: eligible interval with no GHI counterfactual to compare to
            CASE WHEN q.V > {V3} AND q.V <= {VW_V1}
                  AND q.Q_kvar < 0
                  AND sqrt(power(q.Q_kvar, 2) + power(q.P_kW, 2)) >= q.S_99
                  AND a.uncurtailed_P IS NULL
                 THEN 1 ELSE 0 END AS null_uncurtailed_P

        FROM q_impact q
        LEFT JOIN (
            SELECT a.site_id, a.t_stamp, a.uncurtailed_P
            FROM {UNCURTAILED} a
            WHERE {unc_pred}
        ) a ON q.site_id = a.site_id AND q.t_stamp = a.t_stamp
    ),

    -- R3 / R9 / R12: everything temporal comes off t_stamp + 10h
    temporal AS (
        SELECT
            site_id, P_kW, V,
            {temporal_cols('t_stamp')},
            Q_adverse, Q_inactive, Q_significant_shortfall,
            Q_near_conformant, Q_major_surplus,
            Q_adverse + Q_inactive + Q_significant_shortfall
                      + Q_near_conformant + Q_major_surplus AS nonconformance_voltvar,
            curtailment_voltvar, curtailment_eligible, null_uncurtailed_P
        FROM classified
    )

    -- Column order below MUST match create_table() exactly (positional INSERT).
    SELECT
        site_id,
        day_aest                                        AS day,
        day_night,
        sum(P_kW)                                       AS P_kW_sum,
        sum(nonconformance_voltvar)                     AS nonconformance_voltvar_sum,
        sum(Q_adverse)                                  AS Q_adverse_sum,
        sum(Q_inactive)                                 AS Q_inactive_sum,
        sum(Q_significant_shortfall)                    AS Q_significant_shortfall_sum,
        sum(Q_near_conformant)                          AS Q_near_conformant_sum,
        sum(Q_major_surplus)                            AS Q_major_surplus_sum,
        sum(curtailment_voltvar)                        AS curtailment_voltvar_sum,

        -- R5, PENDING BARAN ---------------------------------------------------
        -- `_red` reproduces Hossein's definition literally:
        --     adverse + inactive + Q_major_deficit
        -- which, after the R4 renaming, is adverse + inactive + NEAR-CONFORMANT.
        -- i.e. "reduced non-conformance" currently counts the sites that are
        -- essentially complying and ignores the ones falling well short.
        -- `_red_alt` is what the definition almost certainly SHOULD be:
        --     adverse + inactive + SIGNIFICANT SHORTFALL
        -- Both are emitted so the table does not have to be rebuilt after the
        -- conversation with Baran. DO NOT publish either number until he has
        -- confirmed which range CANVAS intended.
        -- ---------------------------------------------------------------------
        sum(Q_adverse) + sum(Q_inactive) + sum(Q_near_conformant)
                                                        AS nonconformance_voltvar_red_sum,
        sum(Q_adverse) + sum(Q_inactive) + sum(Q_significant_shortfall)
                                                        AS nonconformance_voltvar_red_alt_sum,

        sum(CASE WHEN nonconformance_voltvar  > 0 THEN 1 ELSE 0 END) AS nonconformance_voltvar_count,
        sum(CASE WHEN Q_adverse               > 0 THEN 1 ELSE 0 END) AS Q_adverse_count,
        sum(CASE WHEN Q_inactive              > 0 THEN 1 ELSE 0 END) AS Q_inactive_count,
        sum(CASE WHEN Q_significant_shortfall > 0 THEN 1 ELSE 0 END) AS Q_significant_shortfall_count,
        sum(CASE WHEN Q_near_conformant       > 0 THEN 1 ELSE 0 END) AS Q_near_conformant_count,
        sum(CASE WHEN Q_major_surplus         > 0 THEN 1 ELSE 0 END) AS Q_major_surplus_count,
        sum(CASE WHEN curtailment_voltvar     > 0 THEN 1 ELSE 0 END) AS curtailment_voltvar_count,

        sum(CASE WHEN Q_adverse > 0 THEN 1 ELSE 0 END)
          + sum(CASE WHEN Q_inactive > 0 THEN 1 ELSE 0 END)
          + sum(CASE WHEN Q_near_conformant > 0 THEN 1 ELSE 0 END)
                                                        AS nonconformance_voltvar_red_count,
        sum(CASE WHEN Q_adverse > 0 THEN 1 ELSE 0 END)
          + sum(CASE WHEN Q_inactive > 0 THEN 1 ELSE 0 END)
          + sum(CASE WHEN Q_significant_shortfall > 0 THEN 1 ELSE 0 END)
                                                        AS nonconformance_voltvar_red_alt_count,

        sum(curtailment_eligible)                       AS curtailment_eligible_count,
        sum(null_uncurtailed_P)                         AS null_uncurtailed_P_count,   -- R14
        sum(CASE WHEN V > {V3} THEN 1 ELSE 0 END)       AS exposed_count,              -- V-VAr can act
        count(*)                                        AS all_intervals_count,        -- D4
        count(nonconformance_voltvar)                   AS total_count,                -- == Hossein's
        year_aest                                       AS year,
        month_aest                                      AS month
    FROM temporal
    GROUP BY year_aest, month_aest, day_aest, day_night, site_id
    """


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_months_voltvar(aq, database, year, months, n_parts=8, parts=None,
                       exclude_flex=True):
    """Load AEST months for one year. See stage2_common.run_months."""
    return run_months(aq, database, _insert_sql, year, months,
                      n_parts=n_parts, parts=parts, exclude_flex=exclude_flex)


def preview_sql(year=2024, month=1, n_parts=8, part=0, exclude_flex=True):
    """Return the INSERT SQL for one slice WITHOUT running it. Read it first."""
    utc_start, utc_end, partitions = aest_month_window(year, month)
    return _insert_sql(partitions, utc_start, utc_end,
                       f"site_id % {n_parts} = {part}", exclude_flex)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(aq, database):
    """Sanity checks after loading."""
    shape = aq(f"""
        SELECT year, month,
               count(*)                  AS n_rows,
               count(DISTINCT site_id)   AS n_sites,
               count(DISTINCT day)       AS n_days
        FROM {TARGET}
        GROUP BY year, month ORDER BY year, month
    """, database=database)

    # One row per (site, day, day_night). Anything > 1 means the AEST-month
    # window logic leaked -- this is the D2 regression test.
    dupes = aq(f"""
        SELECT count(*) AS n_dupe_keys FROM (
            SELECT year, month, day, day_night, site_id
            FROM {TARGET}
            GROUP BY year, month, day, day_night, site_id
            HAVING count(*) > 1
        )
    """, database=database)

    # Curtailment can never be negative, and Q_impact buckets must sum to
    # nonconformance.
    coherence = aq(f"""
        SELECT
            sum(CASE WHEN curtailment_voltvar_sum < 0 THEN 1 ELSE 0 END) AS neg_curtailment_rows,
            sum(CASE WHEN abs(nonconformance_voltvar_sum
                              - (Q_adverse_sum + Q_inactive_sum + Q_significant_shortfall_sum
                                 + Q_near_conformant_sum + Q_major_surplus_sum)) > 1e-6
                     THEN 1 ELSE 0 END) AS bucket_mismatch_rows,
            sum(CASE WHEN total_count > all_intervals_count THEN 1 ELSE 0 END) AS count_inversion_rows
        FROM {TARGET}
    """, database=database)

    # How much of the curtailment zone has no counterfactual? (R14)
    cover = aq(f"""
        SELECT sum(curtailment_eligible_count) AS eligible,
               sum(null_uncurtailed_P_count)   AS no_counterfactual,
               round(100.0 * sum(null_uncurtailed_P_count)
                     / nullif(sum(curtailment_eligible_count), 0), 2) AS pct_missing
        FROM {TARGET}
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


def compare_to_original(aq, database, original="conformance_voltvar"):
    """
    Reconcile against Hossein's table. Differences are EXPECTED and should be
    explainable by: R2 (flex-export sites dropped), R1 (max vs avg voltage),
    R3 (AEST vs UTC day boundaries), R10 (S_99 capability).
    """
    new = aq(f"SELECT count(DISTINCT site_id) AS n FROM {TARGET}", database=database)
    old = aq(f"SELECT count(DISTINCT site_id) AS n FROM {original}", database=database)
    only_old = aq(f"""
        SELECT count(*) AS n FROM (
            SELECT DISTINCT site_id FROM {original}
            EXCEPT
            SELECT DISTINCT site_id FROM {TARGET}
        )
    """, database=database)
    flex = aq("""
        SELECT count(DISTINCT site_id) AS n FROM meta_up23c
        WHERE is_pv = True AND flex_export_detected = True
    """, database=database)
    print(f"sites in {TARGET}:  {int(new['n'].iloc[0]):,}")
    print(f"sites in {original}: {int(old['n'].iloc[0]):,}")
    print(f"sites dropped:       {int(only_old['n'].iloc[0]):,}")
    print(f"flex-export sites (expected explanation for the drop): "
          f"{int(flex['n'].iloc[0]):,}")
    return new, old, only_old, flex
