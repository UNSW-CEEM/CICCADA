"""
Data calc-write pipeline: Stage 2. Volt-VAr zone counts (three metrics)
=======================================================================

Produces `conformance_voltvar_zones_v2`: one row per
(site_id, AEST date, day/night, tol) holding interval COUNTS per zone x category,
so that every Volt-VAr metric is a sum of columns:

    nonconformance_frequency  (with voltage exposure)
    remote_conformance_rate   (default)
    legacy_conformance_rate   (the original Stage 2 headline, for back-comparison)

The scoring is the shared SQL in `shared/vvar_sql.py` (the same fragments the
standard_plots and OEM pipelines use), so the rules cannot drift between
pipelines. Metrics and site tags are computed downstream by
`shared/vvar_metrics.conformance_metrics` (see data_query/lib).

WHY A SEPARATE TABLE
--------------------
`conformance_voltvar_v2*` is left untouched (its columns, curtailment and
Q_impact magnitudes are still used). This table adds what the three metrics
need and the original cannot give:
  * all timestamps, day and night, at every power level (zones A-D), not only
    P >= 20% of rating (the original `total_count`);
  * voltage deadband intervals scored (|Q| within tol);
  * missing reactive power dropped and counted, not scored as a pass;
  * both tolerances (0% and 4% of the rating) from ONE scan of `ts`:
    the site-level data is cross-joined with the tolerance list, so the extra
    tolerance costs compute, not bytes scanned.
No join to the uncurtailed (Stage 1) table is needed.

VALID INTERVALS
---------------
P, Q and the rating present, rating > 0, 0 < V < 300 V. The circuit-level
voltage gate in `stage2_common.site_agg_cte` (shared with the other Stage 2
builders) already removes out-of-range voltage before the site aggregate, so
this table cannot count missing / out-of-range voltage (n_missing_v and
n_v_out_of_range are not stored); `n_rows` counts site intervals that passed
that gate. Coverage downstream = n_valid / (calendar days x 288).

ZONE A DAYS
-----------
n_days_A is not stored (a day is split into a day and a night row); the query
layer counts distinct dates with n_A > 0.

SAFE-BY-DEFAULT
---------------
Writes only to the target named here (default `conformance_voltvar_zones_v2`).
No existing table is read for writing, altered or dropped.
"""

from ciccada_config import CONFORMANCE
from pipeline_options import capacity_column
from stage2_common import (
    TABLE_SUFFIX,
    WAREHOUSE,
    aest_month_window,
    run_months,
    site_agg_cte,
    temporal_cols,
)

from shared.vvar_sql import vvar_sql_categories, vvar_sql_columns, zone_count_sql

TARGET = f"conformance_voltvar_zones{TABLE_SUFFIX}"
TOLS = tuple(CONFORMANCE["TOL_OPTIONS"])  # (0.0, 0.04) of the rating basis
V_LO, V_HI = CONFORMANCE["VALID_V_OPEN"]

# Keys first, partition columns last (positional INSERT).
KEY_COLUMNS = [("site_id", "bigint"), ("day", "int"), ("day_night", "string")]
TOL_COLUMNS = [("tol", "double"), ("p_tol_vvar", "double")]
LABEL_COLUMNS = [
    ("rating_basis", "string"),
    ("voltage_aggregation", "string"),
    ("flex_selection", "string"),
]
PARTITION_COLUMNS = [("year", "int"), ("month", "int")]


def _count_columns():
    """[(sql_expression, name)] for the count columns, from the shared SQL."""
    out = []
    for part in zone_count_sql("valid", "d", with_days=False).split(",\n"):
        expr, name = part.strip().rsplit(" AS ", 1)
        out.append((expr, name))
    out += [
        ("count(*) FILTER (WHERE P IS NULL)", "n_missing_p"),
        ("count(*) FILTER (WHERE Q IS NULL)", "n_missing_q"),
        ("count(*) FILTER (WHERE S IS NULL OR S <= 0)", "n_missing_rating"),
    ]
    return out


def column_names():
    """Every output column, in table order."""
    return (
        [c for c, _ in KEY_COLUMNS]
        + [c for c, _ in TOL_COLUMNS]
        + [n for _, n in _count_columns()]
        + [c for c, _ in LABEL_COLUMNS]
        + [c for c, _ in PARTITION_COLUMNS]
    )


def create_table_sql(target=TARGET):
    cols = (
        KEY_COLUMNS
        + TOL_COLUMNS
        + [(n, "bigint") for _, n in _count_columns()]
        + LABEL_COLUMNS
        + PARTITION_COLUMNS
    )
    body = ",\n            ".join(f"{c} {t}" for c, t in cols)
    return f"""
        CREATE TABLE {target} (
            {body}
        )
        PARTITIONED BY (year, month)
        LOCATION 's3://project-ciccada/{WAREHOUSE}/{target}/'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """


def create_table(aq, database, target=TARGET):
    """Create the EMPTY target (drops it first if it exists). Only this target."""
    if not target.startswith("conformance_voltvar_zones"):
        raise ValueError(f"refusing to (re)create {target!r} from the zones builder")
    aq(f"DROP TABLE IF EXISTS {target}", database=database)
    aq(create_table_sql(target), database=database)
    return f"created {target}"


def _select_sql(
    partitions,
    utc_start,
    utc_end,
    part_filter,
    exclude_flex=None,
    rating_basis="ac_capacity_kw",
    voltage_aggregation="avg",
    flex_selection="exclude",
    tols=TOLS,
    p_tol_vvar=0.0,
):
    """The SELECT (no INSERT): one row per site x AEST day x day/night x tol."""
    if p_tol_vvar != 0.0:
        raise NotImplementedError(
            "p_tol_vvar != 0 is a phase 7 sensitivity; the zone SQL takes it, "
            "but the table records a single value per run."
        )
    for t in tols:
        if not 0.0 <= float(t) < 1.0:
            raise ValueError(f"tolerance must be a fraction of the rating, got {t!r}")
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
    rating_col = capacity_column(rating_basis)
    tol_values = ", ".join(f"(CAST({float(t)!r} AS double))" for t in tols)
    counts = ",\n        ".join(f"{e} AS {n}" for e, n in _count_columns())

    return f"""
    WITH
    {data},

    -- Site-level interval in the shared column names (generator convention).
    site AS (
        SELECT
            site_id,
            P_kW AS P,
            Q_kvar AS Q,
            V,
            {rating_col} AS S,
            {temporal_cols('t_stamp')}
        FROM data
    ),

    -- Both tolerances from one scan.
    tols AS (SELECT tol FROM (VALUES {tol_values}) AS t (tol)),

    flagged AS (
        SELECT
            s.*,
            tols.tol,
            (s.P IS NOT NULL AND s.Q IS NOT NULL AND s.S > 0
             AND s.V > {V_LO} AND s.V < {V_HI}) AS valid,
            day_aest AS d
        FROM site s CROSS JOIN tols
    ),
    a AS (SELECT *, {vvar_sql_columns('V', 'P', 'Q', 'S', 'tol', p_tol_vvar)} FROM flagged),
    b AS (SELECT *, {vvar_sql_categories('Q', 'S', 'tol')} FROM a)

    -- Column order must match create_table_sql() (positional INSERT).
    SELECT
        site_id,
        day_aest AS day,
        day_night,
        tol,
        CAST({float(p_tol_vvar)!r} AS double) AS p_tol_vvar,
        {counts},
        '{rating_basis}' AS rating_basis,
        '{voltage_aggregation}' AS voltage_aggregation,
        '{flex_selection}' AS flex_selection,
        year_aest AS year,
        month_aest AS month
    FROM b
    GROUP BY year_aest, month_aest, day_aest, day_night, site_id, tol
    """


def _insert_sql(partitions, utc_start, utc_end, part_filter, exclude_flex=None,
                target=TARGET, **options):
    return f"INSERT INTO {target}\n" + _select_sql(
        partitions, utc_start, utc_end, part_filter, exclude_flex=exclude_flex, **options
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_months_voltvar_zones(
    aq, database, year, months, n_parts=8, parts=None, exclude_flex=None, **run_options
):
    """Load AEST months for one year. See stage2_common.run_months. COSTS ATHENA."""
    return run_months(
        aq,
        database,
        _insert_sql,
        year,
        months,
        n_parts=n_parts,
        parts=parts,
        exclude_flex=exclude_flex,
        **run_options,
    )


def preview_sql(year=2025, month=1, n_parts=8, part=0, exclude_flex=None, insert=True,
                **run_options):
    """Return the SQL for one slice WITHOUT running it. Read it first."""
    utc_start, utc_end, partitions = aest_month_window(year, month)
    build = _insert_sql if insert else _select_sql
    return build(
        partitions,
        utc_start,
        utc_end,
        f"site_id % {n_parts} = {part}",
        exclude_flex=exclude_flex,
        **run_options,
    )


# ---------------------------------------------------------------------------
# Validation (after loading; each query scans only this small table)
# ---------------------------------------------------------------------------
def invariant_sql(target=TARGET):
    """Rows that break an invariant, per year/month/tol. Every count MUST be 0."""
    from shared.as4777_curves import VVAR_CATEGORIES, ZONES

    zone_sum = " + ".join(f"n_{z}" for z in ZONES)
    cat_checks = " OR ".join(
        f"n_{z} <> " + " + ".join(f"n_{z}_{c}" for c in VVAR_CATEGORIES) for z in ZONES
    )
    return f"""
        SELECT year, month, tol,
               count(*)                                   AS n_rows_table,
               count(DISTINCT site_id)                    AS n_sites,
               count_if(n_valid <> {zone_sum})            AS zones_not_summing_to_valid,
               count_if({cat_checks})                     AS categories_not_summing_to_zone,
               count_if(n_valid > n_rows)                 AS valid_above_rows,
               count_if(n_legacy_fail > n_legacy_total)   AS legacy_fail_above_total,
               count_if(n_legacy_total > n_A + n_B)       AS legacy_total_above_zone_AB,
               count_if(n_rows > 144)                     AS rows_above_144_per_half_day
        FROM {target}
        GROUP BY year, month, tol
        ORDER BY year, month, tol
    """


def duplicate_key_sql(target=TARGET):
    return f"""
        SELECT count(*) AS n_dupe_keys FROM (
            SELECT year, month, day, day_night, site_id, tol
            FROM {target}
            GROUP BY year, month, day, day_night, site_id, tol
            HAVING count(*) > 1
        )
    """


def validate(aq, database, target=TARGET):
    inv = aq(invariant_sql(target), database=database)
    dup = aq(duplicate_key_sql(target), database=database)
    print("Invariants per AEST month and tol (all counts except n_rows/n_sites MUST be 0):")
    print(inv.to_string(index=False))
    print(f"\nDuplicate keys (MUST be 0): {int(dup['n_dupe_keys'].iloc[0])}")
    return inv, dup


def legacy_parity_sql(year, month, target=TARGET,
                      original="conformance_voltvar_v2_flex_included"):
    """
    Checkpoint 5c: legacy buckets (tol 4%) vs the original Stage 2 table, one
    month, per site. Failures must match exactly. Denominators differ only by
    intervals with missing Q, which the original scores as a pass (checkpoint 5d).
    """
    return f"""
        WITH z AS (
            SELECT site_id,
                   sum(legacy_adverse) AS adverse,
                   sum(legacy_inactive) AS inactive,
                   sum(legacy_significant_shortfall) AS shortfall,
                   sum(legacy_near_conformant + legacy_inside_band) AS near_or_inside,
                   sum(legacy_major_surplus) AS surplus,
                   sum(n_legacy_total) AS legacy_total
            FROM {target}
            WHERE year = {year} AND month = {month} AND tol = 0.04
            GROUP BY site_id
        ),
        o AS (
            SELECT site_id,
                   sum(Q_adverse_count) AS adverse,
                   sum(Q_inactive_count) AS inactive,
                   sum(Q_significant_shortfall_count) AS shortfall,
                   sum(Q_major_surplus_count) AS surplus,
                   sum(total_count) AS total
            FROM {original}
            WHERE year = {year} AND month = {month}
            GROUP BY site_id
        )
        SELECT count(*) AS n_sites_both,
               count_if(z.adverse <> o.adverse) AS adverse_differs,
               count_if(z.inactive <> o.inactive) AS inactive_differs,
               count_if(z.shortfall <> o.shortfall) AS shortfall_differs,
               count_if(z.surplus <> o.surplus) AS surplus_differs,
               sum(o.total - z.legacy_total) AS original_minus_new_denominator
        FROM z JOIN o USING (site_id)
    """


def missing_q_sql(year, month, target=TARGET):
    """Checkpoint 5d: intervals with missing Q (the original counts them as passes)."""
    return f"""
        SELECT sum(n_rows) AS n_rows, sum(n_missing_q) AS n_missing_q,
               sum(n_missing_p) AS n_missing_p,
               count(DISTINCT CASE WHEN n_missing_q > 0 THEN site_id END) AS sites_with_missing_q
        FROM {target}
        WHERE year = {year} AND month = {month} AND tol = 0.04
    """
