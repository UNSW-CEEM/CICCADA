"""
Analysis contract: the manifest and the SQL predicates it describes.
====================================================================

Deliverable D5. The SolarEdge counterpart of
``data_query/lib/analysis_contract.py``.

Two jobs, and they are the same job seen from two sides:

* ``manifest()`` renders every methodological choice as a table, so no number
  leaves a notebook without the settings that produced it.
* The ``*_sql()`` helpers turn those same choices into SQL predicates, so the
  manifest cannot drift away from what the queries actually did.

That coupling is the point. A manifest assembled by hand alongside independently
written SQL is a manifest that will eventually lie.
"""

from __future__ import annotations

import pandas as pd

from solar_edge.config import se_config as C
from solar_edge.lib import se_params

__all__ = [
    "manifest",
    "months_sql",
    "voltage_sql",
    "v_band_sql",
    "peak_hours_sql",
    "day_night_sql",
    "derating_sql",
    "cohort_join_sql",
    "cohort_where_sql",
    "capacity_column",
]


# ═══════════════════════════════════════════════════════════════════════════
# MANIFEST
# ═══════════════════════════════════════════════════════════════════════════

def manifest(config=None, params=None, include_conventions: bool = True) -> pd.DataFrame:
    """
    Every methodological choice behind a result, as one table.

    Grouped into sections so a reader can see at a glance which choices are
    dataset properties (conventions), which are cohort definitions (config), and
    which are detection parameters (params).

    Substitutions forced by the data are called out explicitly rather than left to
    be inferred from a default value.
    """
    config = (config or se_params.CONFIG).validate()
    params = (params or se_params.PARAMS).validate()

    rows: list[tuple[str, str, object]] = []

    if include_conventions:
        for row in C.describe_conventions().itertuples(index=False):
            rows.append(("convention", row.convention, row.value))

    rows.extend([
        ("cohort", "months", f"{config.months[0]} .. {config.months[-1]} "
                             f"({len(config.months)} months)"),
        ("cohort", "interval_h", config.interval_h),
        ("cohort", "phase_cohort", config.phase_cohort),
        ("cohort", "day_night", config.day_night),
        ("cohort", "derating_selection", config.derating_selection),
        ("cohort", "night_anomaly_selection", config.night_anomaly_selection),
        ("cohort", "min_days_observed", config.min_days_observed),
        ("cohort", "max_capacity_kva", config.max_capacity_kva),
        ("cohort", "min_site_intervals", config.min_site_intervals),

        ("basis", "rating_basis", config.rating_basis),
        ("basis", "empirical_limit_basis", config.empirical_limit_basis),
        ("basis", "tolerance_basis", config.tolerance_basis),
        ("basis", "tolerance_fraction", config.tolerance_fraction),
        ("basis", "voltage_aggregation", config.voltage_aggregation),
        ("basis", "site_nonconf_threshold", config.site_nonconf_threshold),

        ("detection", "voltage band", f"{params.v_low:.0f} - {params.v_high:.0f} V"),
        ("detection", "peak hours (AEST)",
         f"{params.peak_hour_start:02d}:00 - {params.peak_hour_end:02d}:00"),
        ("detection", "require_apparent_limit_symptom",
         params.require_apparent_limit_symptom),
        ("detection", "minimum_flagged_intervals", params.minimum_flagged_intervals),
        ("detection", "apply_ghi_filter", params.apply_ghi_filter),
        ("detection", "ghi_cs_ratio_min", params.ghi_cs_ratio_min),
    ])

    # Substitutions stated, not implied.
    rows.extend([
        ("substitution", "nameplate capacity",
         "NOT AVAILABLE in this delivery -- s_99 used as the sole capacity basis"),
        ("substitution", "AS/NZS 4777.2 tolerance anchor",
         f"{config.tolerance_fraction:.0%} of {config.tolerance_basis} "
         "(Solar Analytics anchors it to nameplate ac_capacity_kw)"),
        ("substitution", "flexible-export flag",
         "NOT AVAILABLE -- no plateau-heuristic equivalent built; "
         "derating_active is a different signal and is selected separately"),
        ("substitution", "irradiance / clear-sky gate",
         "NOT AVAILABLE until D12 (bom_nci.solar extract) -- Method B not yet runnable"),
    ])

    return pd.DataFrame(rows, columns=["section", "setting", "value"])


# ═══════════════════════════════════════════════════════════════════════════
# PREDICATES
# ═══════════════════════════════════════════════════════════════════════════

def capacity_column(basis: str, alias: str = "c") -> str:
    """Resolve a capacity basis to its column in ``se_site_capacity``."""
    if basis not in se_params.CAPACITY_BASES:
        raise ValueError(f"unknown capacity basis {basis!r}")
    return f"{alias}.{basis}"


def voltage_sql(aggregation: str, alias: str = "i") -> str:
    """Resolve the voltage-aggregation choice to its stored column."""
    if aggregation == "max":
        return f"{alias}.V_max"
    if aggregation == "mean":
        return f"{alias}.V_mean"
    raise ValueError(f"unknown voltage_aggregation {aggregation!r}")


def months_sql(months, column: str = "i.dt_month") -> str:
    values = ", ".join(f"'{m}'" for m in months)
    return f"{column} IN ({values})"


def v_band_sql(params, voltage_expr: str) -> str:
    """The Volt-VAr detection band. Open interval, matching the legacy queries."""
    return f"{voltage_expr} > {params.v_low} AND {voltage_expr} < {params.v_high}"


def peak_hours_sql(params, column: str = "i.ts_aest") -> str:
    """
    Peak-solar window in the AEST analysis frame.

    Half-open, ``[start, end)``. Note the legacy `fetch_method_a_site_year` uses
    ``BETWEEN start AND end``, which is INCLUSIVE of the end hour and so spans four
    hours when the parameters say three. The half-open form is used here; a
    like-for-like comparison against the Solar Analytics numbers should set
    ``peak_hour_end=15`` to reproduce the legacy window.
    """
    return (
        f"hour({column}) >= {params.peak_hour_start} "
        f"AND hour({column}) < {params.peak_hour_end}"
    )


def day_night_sql(selection: str, column: str = "i.P_kW") -> str:
    """
    Day/night split.

    Defined by observed generation rather than by clock hour or solar geometry:
    without irradiance or per-site coordinates, generation is the only signal that
    actually distinguishes the two. Revisit once D12 supplies solar position.
    """
    if selection == "all":
        return "1 = 1"
    if selection == "day":
        return f"{column} > 0.1"
    if selection == "night":
        return f"coalesce({column}, 0) <= 0.1"
    raise ValueError(f"unknown day_night {selection!r}")


def derating_sql(selection: str, column: str = "i.derating_active") -> str:
    if selection == "include":
        return "1 = 1"
    if selection == "exclude":
        return f"NOT {column}"
    if selection == "only":
        return f"{column}"
    raise ValueError(f"unknown derating_selection {selection!r}")


def phase_cohort_sql(cohort: str, column: str = "s.is_three_phase") -> str:
    if cohort == "all":
        return "1 = 1"
    if cohort == "three":
        return f"{column}"
    if cohort == "single":
        return f"NOT {column}"
    raise ValueError(f"unknown phase_cohort {cohort!r}")


def cohort_join_sql(interval_alias: str = "i") -> str:
    """The standard join from intervals to the site dimension and capacity table."""
    return (
        f"JOIN se_site s ON {interval_alias}.site_alias = s.site_alias\n"
        f"        LEFT JOIN se_site_capacity c ON {interval_alias}.site_alias = c.site_alias"
    )


def cohort_where_sql(config, extra: list[str] | None = None) -> str:
    """
    Every site- and interval-level cohort filter implied by an ``SEAnalysisConfig``.

    Returned as a single indented WHERE body so callers compose rather than
    re-derive. Keeping this in one place is what stops notebook A and notebook B
    quietly using different cohorts.
    """
    config = config.validate()
    limit_col = capacity_column(config.empirical_limit_basis)

    clauses = [
        months_sql(config.months),
        phase_cohort_sql(config.phase_cohort),
        derating_sql(config.derating_selection),
        day_night_sql(config.day_night),
        f"{limit_col} > 0",
        f"{limit_col} <= {config.max_capacity_kva}",
        f"s.n_days_observed >= {config.min_days_observed}",
    ]
    if config.night_anomaly_selection == "exclude":
        clauses.append("NOT s.has_night_generation_anomaly")
    if extra:
        clauses.extend(extra)

    return "\n          AND ".join(f"({c})" for c in clauses)
