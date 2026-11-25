"""
Volt-VAr interval scoring as SQL SELECT-list fragments (Athena/Trino/DuckDB).
=============================================================================

One place for the SQL every pipeline uses 
(standard_plots local scans, the OEM pipeline, and Stage 2 later), 
built from the SQL twins in as4777_curves so the rules cannot drift from the Python versions in vvar_metrics.

Use in two steps, on a relation with site-total voltage, active power, 
reactive power and rating columns (kW / kvar, generator convention):

    a AS (SELECT *, {vvar_sql_columns(v, p, q, s, tol)} FROM data),
    b AS (SELECT *, {vvar_sql_categories(q, s, tol)} FROM a)

Columns added by vvar_sql_columns:
    q_req, q_lo, q_hi      required Q and the accepted band
    zone                   'A' / 'B' / 'C' / 'D' (as4777_curves.ZONE_LABELS)
    q_cap_legacy           Figure 2.1 floor used by the legacy band
    legacy_assessable      |P| >= 20% of rating (legacy denominator)
Columns added by vvar_sql_categories:
    category               as4777_curves.vvar_category
    legacy_category        pre-rework bucket (nearest-edge Q_impact), or
                           'inside_band' / 'not_assessable'
"""

from shared.as4777_curves import (
    q_conformance_floor_absorbing_sql,
    vvar_band_sql,
    vvar_category_sql,
    vvar_required_q_sql,
    vvar_zone_sql,
)
from shared.ciccada_config import AS4777

#: Legacy categories (outside the band, P >= 20%) counted as failures.
LEGACY_FAILURES = ("adverse", "inactive", "significant_shortfall")

#: Every value legacy_category can take, in plotting order.
LEGACY_CATEGORIES = (
    "adverse",
    "inactive",
    "significant_shortfall",
    "near_conformant",
    "inside_band",
    "major_surplus",
    "not_assessable",
)


def vvar_sql_columns(v="V", p="P", q="Q", s="S", tol=0.04, p_tol_vvar=0.0) -> str:
    """SELECT-list fragment: q_req, q_lo, q_hi, zone, q_cap_legacy, legacy_assessable."""
    q_min, q_max = vvar_band_sql(v, p, s, tol=tol, p_tol=p_tol_vvar)
    pmin = AS4777["QCAP"]["P_MIN"]
    return f"""
            {vvar_required_q_sql(v, s)} AS q_req,
            {q_min} AS q_lo,
            {q_max} AS q_hi,
            {vvar_zone_sql(v, p, s, p_tol_vvar)} AS zone,
            {q_conformance_floor_absorbing_sql(p, s)} AS q_cap_legacy,
            abs({p}) >= {pmin} * {s} AS legacy_assessable"""


def vvar_sql_categories(q="Q", s="S", tol=0.04) -> str:
    """SELECT-list fragment on top of vvar_sql_columns: category, legacy_category."""
    qi = AS4777["QIMP"]
    # Legacy band (Stage 2 / OEM 'review_corrected' clamp) and nearest-edge Q_impact.
    l_hi = f"""CASE WHEN q_req + {tol} * {s} < 0
                    THEN greatest(q_req + {tol} * {s}, q_cap_legacy + {tol} * {s})
                    ELSE q_req + {tol} * {s} END"""
    l_lo = f"""CASE WHEN q_req - {tol} * {s} > 0
                    THEN least(q_req - {tol} * {s}, -q_cap_legacy - {tol} * {s})
                    ELSE q_req - {tol} * {s} END"""
    edge = f"(CASE WHEN abs({q} - ({l_hi})) <= abs({q} - ({l_lo})) THEN ({l_hi}) ELSE ({l_lo}) END)"
    impact = f"""((CASE WHEN abs(({l_hi}) + ({l_lo})) <= 1e-9 THEN 1 ELSE sign({edge}) * sign({q}) END)
                  * abs({q}) / (abs({edge}) + 1e-9))"""
    legacy = f"""
            CASE WHEN NOT legacy_assessable THEN 'not_assessable'
                 WHEN {q} >= ({l_lo}) AND {q} <= ({l_hi}) THEN 'inside_band'
                 WHEN {impact} <  {qi['thr1']} THEN 'adverse'
                 WHEN {impact} <= {qi['thr2']} THEN 'inactive'
                 WHEN {impact} <  {qi['thr3']} THEN 'significant_shortfall'
                 WHEN {impact} <= {qi['thr4']} THEN 'near_conformant'
                 ELSE 'major_surplus' END"""
    return f"""
            {vvar_category_sql(q, 'q_req', 'q_lo', 'q_hi')} AS category,
            {legacy} AS legacy_category"""


def zone_count_sql(valid="valid", day="day", with_days=True) -> str:
    """
    Aggregate SELECT-list (per GROUP BY) giving the shared.vvar_metrics.zone_counts
    columns from rows carrying `valid`, `zone`, `category`, `legacy_category`,
    `legacy_assessable` and a local-date column `day`.

    with_days=False leaves out n_days_A, for tables stored at day grain or finer
    (Stage 2), where distinct days are counted when the rows are summed.
    """
    from shared.as4777_curves import VVAR_CATEGORIES, ZONES

    lf = ", ".join(f"'{c}'" for c in LEGACY_FAILURES)
    parts = [
        "count(*) AS n_rows",
        f"count(*) FILTER (WHERE {valid}) AS n_valid",
    ]
    parts += [f"count(*) FILTER (WHERE {valid} AND zone = '{z}') AS n_{z}" for z in ZONES]
    parts += [
        f"count(*) FILTER (WHERE {valid} AND zone = '{z}' AND category = '{c}') AS n_{z}_{c}"
        for z in ZONES
        for c in VVAR_CATEGORIES
    ]
    if with_days:
        parts.append(
            f"count(DISTINCT {day}) FILTER (WHERE {valid} AND zone = 'A') AS n_days_A"
        )
    parts += [
        f"count(*) FILTER (WHERE {valid} AND legacy_assessable) AS n_legacy_total",
        f"count(*) FILTER (WHERE {valid} AND legacy_category IN ({lf})) AS n_legacy_fail",
    ]
    parts += [
        f"count(*) FILTER (WHERE {valid} AND legacy_category = '{c}') AS legacy_{c}"
        for c in LEGACY_CATEGORIES
    ]
    return ",\n           ".join(parts)


def zone_count_names(with_days=True) -> list:
    """Column names produced by zone_count_sql, in order."""
    return [
        part.strip().rsplit(" AS ", 1)[1]
        for part in zone_count_sql(with_days=with_days).split(",\n")
    ]
