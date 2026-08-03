"""Read-only, reconciling views over the three Delivery 4 mechanism-result tables.

This module never rebuilds a result table (no ``build_*`` call anywhere
here) and never accepts free-form SQL or column names from a caller --
every dimension a view can group by is drawn from an explicit allow-list.
Every percentage is computed from a named denominator with explicit
null-on-zero handling: an unassessable group is never silently reported as
a zero failure/conformance rate.

Volt-VAr and Volt-Watt proxy results are namespaced by
``MechanismAnalysisConfig.phase_scope_basis`` (``der_inferred`` vs
``all_phases``; see ``mechanism_config.py`` and ``mechanism_paths.py``).
Every view below therefore accepts an optional ``mechanism`` argument that
selects which track's files to read -- ``None`` resolves to the original,
unnamespaced ``der_inferred`` paths, exactly like every other
mechanism-aware path helper in this project. This is a deliberate extension
beyond docs/DELIVERY_5_ACCEPTANCE_SPEC.md's original recommended interface
(written before the dual-track feature existed): the spec explicitly allows
renaming/extending the interface "only with a documented reason", and this
comment is that reason.

``response_observability.parquet`` is the one exception: it is always a
single shared, unnamespaced file regardless of ``phase_scope_basis`` (see
``build_response_observability``'s docstring in ``mechanism_results.py``),
so observability views never pass a caller's ``mechanism`` into
``response_observability_path`` -- only ``voltvar_*`` and ``voltwatt_*``
views do that. The ``mechanism`` parameter on the two observability view
functions exists only to attach review-state context to the returned frame,
mirroring the recommended interface in the acceptance spec.

There is deliberately no curtailment view in this module. Gate 7 is unmet;
see ``CURTAILMENT_UNAVAILABLE_CONTEXT``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .config import FoundationConfig, SourceScope
from .db import connect
from .mechanism_config import MechanismAnalysisConfig
from .mechanism_paths import (
    response_observability_path,
    voltvar_results_path,
    voltwatt_results_path,
)
from .schemas import quote_identifier, sql_string

# ---------------------------------------------------------------------------
# Allow-listed dimensions.
#
# Curve tables (Volt-VAr/Volt-Watt) use ``phase_scope``, a *mapped* value
# (a DER-inferred phase letter, 'unmapped', or 'all_phases' under the
# all_phases basis). Response observability uses ``phase``, the *actual*
# telemetry phase letter regardless of DER mapping. These must never be
# confused -- they answer different questions -- so they get separate
# allow-lists and separate dimension names.
# ---------------------------------------------------------------------------

CURVE_ALLOWED_DIMENSIONS: tuple[str, ...] = (
    "serial",
    "year_utc",
    "month_utc",
    "analysis_cohort",
    "phase_scope",
    "voltage_bin_lower_v",
)

OBSERVABILITY_ALLOWED_DIMENSIONS: tuple[str, ...] = (
    "serial",
    "year_utc",
    "month_utc",
    "analysis_cohort",
    "phase",
)

VOLTVAR_DENOMINATOR_COLUMNS: tuple[str, ...] = (
    "n_ineligible_site",
    "n_missing_input",
    "n_not_activated",
    "n_sign_unverified",
    "n_capacity_unavailable",
    "n_below_minimum_active_power",
    "n_assessable",
)

VOLTWATT_DENOMINATOR_COLUMNS: tuple[str, ...] = (
    "n_ineligible_site",
    "n_missing_input",
    "n_not_activated",
    "n_sign_unverified",
    "n_not_exporting",
    "n_capacity_unavailable",
    "n_assessable",
)

VOLTVAR_STATUS_COLUMNS: tuple[str, ...] = (
    "n_conformant",
    "n_adverse",
    "n_inactive",
    "n_major_deficit",
    "n_minor_deviation",
    "n_major_surplus",
)

# Reviewed project conformance methodology (not a code-derived split -- a
# deliberate business rule): non-conformance groups 'wrong direction', 'no
# response' and 'not enough response'; conformance groups 'exactly on
# curve', 'close enough' and 'more than required'. Both are always reported
# alongside the six raw buckets, never in place of them.
VOLTVAR_NON_CONFORMANCE_COLUMNS: tuple[str, ...] = (
    "n_adverse",
    "n_inactive",
    "n_major_deficit",
)
VOLTVAR_CONFORMANCE_COLUMNS: tuple[str, ...] = (
    "n_conformant",
    "n_minor_deviation",
    "n_major_surplus",
)

VOLTWATT_STATUS_COLUMNS: tuple[str, ...] = (
    "n_proxy_exceeds_curve_ceiling",
    "n_proxy_does_not_exceed_curve_ceiling",
)

_OBSERVABILITY_SHARED_STATUSES: tuple[str, ...] = (
    "ineligible_site",
    "not_inferred_der_phase",
    "sign_unverified",
    "insufficient_excitation",
)
VOLTVAR_OBSERVABILITY_STATUSES: tuple[str, ...] = _OBSERVABILITY_SHARED_STATUSES + (
    "expected_direction_observed",
    "opposite_or_flat_direction_observed",
)
VOLTWATT_OBSERVABILITY_STATUSES: tuple[str, ...] = _OBSERVABILITY_SHARED_STATUSES + (
    "drop_direction_observed",
    "opposite_or_flat_direction_observed",
)

_CONTEXT_COLUMNS_SHARED: tuple[str, ...] = (
    "measurement_basis",
    "voltage_measurement_location",
    "active_sign_review_state",
    "formal_inverter_conformance_assessable",
)
# Column availability is not uniform across the three result tables:
#   - reactive_sign_review_state: voltvar and response_observability only
#     (Volt-Watt is a pure active-power mechanism and never stamps a
#     reactive-sign column -- see mechanism_results.py's _voltwatt_sql).
#   - voltage_basis / capacity_basis: voltvar and voltwatt only (response
#     observability does not use a capacity-normalised magnitude curve).
_CONTEXT_COLUMNS_CURVE_ONLY: tuple[str, ...] = (
    "reactive_sign_review_state",
    "voltage_basis",
    "capacity_basis",
)
_CONTEXT_COLUMNS_VOLTWATT_ONLY: tuple[str, ...] = ("voltage_basis", "capacity_basis")
_CONTEXT_COLUMNS_RESPONSE_ONLY: tuple[str, ...] = ("reactive_sign_review_state",)

CURTAILMENT_UNAVAILABLE_CONTEXT: dict[str, object] = {
    "status": "unavailable",
    "reason": "methodology gate 7 unmet",
    "detail": (
        "Counterfactual-supported curtailment requires a validated "
        "load-PV decomposition and an uncertainty-aware counterfactual "
        "(docs/METHODOLOGY_GATES.md, gate 7). Neither is built. No "
        "curtailment estimate, rate, energy figure or blended score "
        "exists anywhere in this module."
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_dimensions(
    dimensions: Sequence[str],
    allowed: Iterable[str],
    label: str,
) -> tuple[str, ...]:
    dims = tuple(dimensions)
    allowed_set = set(allowed)
    unknown = [d for d in dims if d not in allowed_set]
    if unknown:
        raise ValueError(
            f"{label} dimensions must be drawn from {sorted(allowed_set)}; "
            f"received unknown dimension(s): {unknown}"
        )
    if len(set(dims)) != len(dims):
        raise ValueError(f"{label} dimensions must not repeat: {dims}")
    return dims


def _select_prefix(dimensions: tuple[str, ...]) -> str:
    if not dimensions:
        return ""
    return ", ".join(quote_identifier(d) for d in dimensions) + ",\n            "


def _group_by_clause(dimensions: tuple[str, ...]) -> str:
    if not dimensions:
        return ""
    return "GROUP BY " + ", ".join(quote_identifier(d) for d in dimensions)


def _finite(expr: str) -> str:
    """Guard a DOUBLE SQL expression against literal ``NaN``.

    ``regr_slope``/``corr`` can return ``NaN`` (not SQL ``NULL``) from a
    zero-variance group -- unlike ``NULL``, a single ``NaN`` poisons any
    ``SUM``/weighted-mean it participates in. This rewrites ``NaN`` to
    ``NULL`` so aggregate functions correctly skip it instead of silently
    turning an otherwise-real fleet-level result into ``NaN``.
    """

    return f"(CASE WHEN isnan({expr}) THEN NULL ELSE {expr} END)"


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found at {path}. Build it with notebook 04 first; "
            "this module never builds results, only reads them."
        )
    return path


def _run_query(config: FoundationConfig, query: str) -> pd.DataFrame:
    connection = connect(config)
    try:
        return connection.execute(query).fetchdf()
    finally:
        connection.close()


def _raw_column_sums(
    config: FoundationConfig,
    path: Path,
    columns: tuple[str, ...],
) -> dict[str, int]:
    select_cols = ", ".join(f"sum({c}) AS {c}" for c in columns)
    connection = connect(config)
    try:
        row = connection.execute(
            f"SELECT {select_cols} FROM read_parquet({sql_string(str(path))})"
        ).fetchone()
    finally:
        connection.close()
    return {col: (0 if value is None else int(value)) for col, value in zip(columns, row)}


def _add_fractions(
    frame: pd.DataFrame,
    denominator_col: str,
    numerator_cols: tuple[str, ...],
    suffix: str,
) -> pd.DataFrame:
    """Add ``<label>_fraction_of_<suffix>`` columns; null (not zero) when the
    denominator is zero, per the acceptance spec ("If n_assessable == 0,
    classification fractions are null, not zero.").
    """

    denominator = frame[denominator_col].astype("Float64")
    safe_denominator = denominator.mask(denominator == 0)
    for col in numerator_cols:
        label = col[2:] if col.startswith("n_") else col
        frame[f"{label}_fraction_of_{suffix}"] = (
            frame[col].astype("Float64") / safe_denominator
        )
    return frame


# ---------------------------------------------------------------------------
# Inventory and context
# ---------------------------------------------------------------------------


def result_inventory(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    mechanism: MechanismAnalysisConfig | None = None,
) -> pd.DataFrame:
    """One row per Delivery 4 result file: existence, size, rows, methodology.

    ``mechanism`` selects the voltvar/voltwatt phase_scope_basis track
    (default ``der_inferred``). ``response_observability`` is always the
    single shared file and ignores ``mechanism`` for path resolution.
    """

    entries = (
        ("voltvar_proxy_results", voltvar_results_path(config, scope, mechanism)),
        ("voltwatt_proxy_results", voltwatt_results_path(config, scope, mechanism)),
        ("response_observability", response_observability_path(config, scope)),
    )
    rows: list[dict[str, Any]] = []
    connection = connect(config)
    try:
        for table, path in entries:
            exists = path.is_file()
            row: dict[str, Any] = {
                "table": table,
                "path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
                "modified_utc": (
                    datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat()
                    if exists
                    else None
                ),
                "row_count": None,
                "distinct_methodology_ids": None,
                "methodology_id": None,
            }
            if exists:
                count, distinct_ids, one_id = connection.execute(
                    f"""
                    SELECT count(*), count(DISTINCT methodology_id),
                           any_value(methodology_id)
                    FROM read_parquet({sql_string(str(path))})
                    """
                ).fetchone()
                row["row_count"] = int(count)
                row["distinct_methodology_ids"] = int(distinct_ids)
                row["methodology_id"] = one_id
            rows.append(row)
    finally:
        connection.close()
    return pd.DataFrame(rows)


def result_context(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    mechanism: MechanismAnalysisConfig | None = None,
) -> dict[str, object]:
    """Methodology/provenance context, reconciled across every result table
    that currently exists for this ``mechanism`` track.

    Raises ``ValueError`` if a provenance field disagrees within a table or
    across tables -- the acceptance spec requires ``result_context`` to
    "reject mixed methodology/provenance values unless the caller explicitly
    splits them" (i.e. calls this once per mechanism/scope rather than
    expecting one call to blend two different builds).

    ``methodology_id`` is handled separately from the rest: Volt-VAr and
    Volt-Watt are both namespaced by ``phase_scope_basis`` and built together
    under the same ``mechanism``, so their ``methodology_id`` values must
    agree. ``response_observability`` is deliberately the one shared,
    unnamespaced table (see ``build_response_observability``'s docstring) --
    it always carries whichever ``mechanism`` built it (historically
    ``der_inferred``), so comparing its ``methodology_id`` to an all_phases
    curve-table run would raise on a difference that is expected, not an
    error. Its id is reported separately as
    ``response_observability_methodology_id``, with an explicit
    ``response_observability_methodology_matches_curve_tables`` flag instead
    of a hard failure.
    """

    context: dict[str, object] = {}
    sources: dict[str, str] = {}
    sources_tables = (
        (
            "voltvar_proxy_results",
            voltvar_results_path(config, scope, mechanism),
            _CONTEXT_COLUMNS_SHARED + _CONTEXT_COLUMNS_CURVE_ONLY,
        ),
        (
            "voltwatt_proxy_results",
            voltwatt_results_path(config, scope, mechanism),
            _CONTEXT_COLUMNS_SHARED + _CONTEXT_COLUMNS_VOLTWATT_ONLY,
        ),
        (
            "response_observability",
            response_observability_path(config, scope),
            _CONTEXT_COLUMNS_SHARED + _CONTEXT_COLUMNS_RESPONSE_ONLY,
        ),
    )
    curve_methodology_ids: dict[str, str] = {}
    response_methodology_id: str | None = None
    for table, path, columns in sources_tables:
        if not path.is_file():
            continue
        all_columns = ("methodology_id", *columns)
        value_select = ", ".join(f"any_value({c}) AS {c}" for c in all_columns)
        distinct_select = ", ".join(
            f"count(DISTINCT {c}) AS distinct_{c}" for c in all_columns
        )
        connection = connect(config)
        try:
            row = connection.execute(
                f"""
                SELECT {value_select}, {distinct_select}
                FROM read_parquet({sql_string(str(path))})
                """
            ).fetchone()
        finally:
            connection.close()
        values = dict(zip(all_columns, row[: len(all_columns)]))
        distinct_counts = dict(zip(all_columns, row[len(all_columns):]))
        for name in all_columns:
            if distinct_counts[name] > 1:
                raise ValueError(
                    f"{table} contains {distinct_counts[name]} distinct values for "
                    f"{name} -- split by an explicit mechanism/scope before calling "
                    "result_context."
                )
        if table == "response_observability":
            response_methodology_id = values["methodology_id"]
        else:
            curve_methodology_ids[table] = values["methodology_id"]
        for name in columns:
            if name in context and context[name] != values[name]:
                raise ValueError(
                    f"{name} disagrees between {sources[name]} ({context[name]!r}) "
                    f"and {table} ({values[name]!r}) -- call result_context on one "
                    "table/mechanism at a time rather than mixing."
                )
            context[name] = values[name]
            sources[name] = table

    distinct_curve_ids = set(curve_methodology_ids.values())
    if len(distinct_curve_ids) > 1:
        raise ValueError(
            f"methodology_id disagrees between voltvar and voltwatt: "
            f"{curve_methodology_ids!r} -- these are always built together "
            "under one mechanism, so this indicates a stale or mismatched file."
        )
    curve_methodology_id = next(iter(distinct_curve_ids), None)
    context["methodology_id"] = curve_methodology_id
    context["response_observability_methodology_id"] = response_methodology_id
    context["response_observability_methodology_matches_curve_tables"] = (
        response_methodology_id == curve_methodology_id
        if curve_methodology_id is not None and response_methodology_id is not None
        else None
    )
    context["phase_scope_basis"] = (mechanism or MechanismAnalysisConfig()).phase_scope_basis
    context["scope"] = scope.label
    return context


# ---------------------------------------------------------------------------
# Volt-VAr views
# ---------------------------------------------------------------------------


def voltvar_denominator_view(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    mechanism: MechanismAnalysisConfig | None = None,
    dimensions: Sequence[str] = (),
) -> pd.DataFrame:
    dims = _validate_dimensions(dimensions, CURVE_ALLOWED_DIMENSIONS, "voltvar denominator")
    path = _require_file(
        voltvar_results_path(config, scope, mechanism), "Volt-VAr proxy results"
    )
    sum_cols = ",\n            ".join(
        f"sum({c}) AS {c}" for c in ("n_source_intervals", *VOLTVAR_DENOMINATOR_COLUMNS)
    )
    query = f"""
        SELECT
            {_select_prefix(dims)}{sum_cols}
        FROM read_parquet({sql_string(str(path))})
        {_group_by_clause(dims)}
        ORDER BY {", ".join(quote_identifier(d) for d in dims) or "1"}
    """
    frame = _run_query(config, query)
    return _add_fractions(frame, "n_source_intervals", VOLTVAR_DENOMINATOR_COLUMNS, "source")


def voltvar_status_view(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    mechanism: MechanismAnalysisConfig | None = None,
    dimensions: Sequence[str] = (),
    minimum_denominator: int = 0,
) -> pd.DataFrame:
    dims = _validate_dimensions(dimensions, CURVE_ALLOWED_DIMENSIONS, "voltvar status")
    path = _require_file(
        voltvar_results_path(config, scope, mechanism), "Volt-VAr proxy results"
    )
    sum_cols = ",\n            ".join(
        f"sum({c}) AS {c}" for c in ("n_assessable", *VOLTVAR_STATUS_COLUMNS)
    )
    query = f"""
        SELECT
            {_select_prefix(dims)}{sum_cols},
            sum(mean_q_impact * n_assessable)
                / nullif(sum(n_assessable), 0) AS mean_q_impact_weighted
        FROM read_parquet({sql_string(str(path))})
        {_group_by_clause(dims)}
        ORDER BY {", ".join(quote_identifier(d) for d in dims) or "1"}
    """
    frame = _run_query(config, query)
    frame = _add_fractions(frame, "n_assessable", VOLTVAR_STATUS_COLUMNS, "assessable")
    frame["n_conformance"] = frame[list(VOLTVAR_CONFORMANCE_COLUMNS)].sum(axis=1)
    frame["n_non_conformance"] = frame[list(VOLTVAR_NON_CONFORMANCE_COLUMNS)].sum(axis=1)
    frame = _add_fractions(
        frame, "n_assessable", ("n_conformance", "n_non_conformance"), "assessable"
    )
    frame["low_denominator_warning"] = frame["n_assessable"] < minimum_denominator
    return frame


def voltvar_site_conformance_view(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    mechanism: MechanismAnalysisConfig | None = None,
    conformance_threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-site conformance rollup: is *this site's own* majority of
    assessable Volt-VAr intervals conformant or not, using the same
    conformance/non-conformance grouping as ``voltvar_status_view``
    (conformance = conformant + minor_deviation + major_surplus;
    non-conformance = adverse + inactive + major_deficit).

    A site with zero assessable intervals (e.g. every row is
    ``capacity_unavailable`` under this track) is reported as
    ``site_status='not_assessable'`` -- it is never silently folded into
    "non-conformant". ``conformance_threshold`` (default 0.5, strict
    majority of that site's own assessable intervals) is a deliberate,
    explicit judgment call, not a derived constant -- change it per caller
    if a different rule is wanted.
    """

    path = _require_file(
        voltvar_results_path(config, scope, mechanism), "Volt-VAr proxy results"
    )
    sum_cols = ",\n            ".join(
        f"sum({c}) AS {c}" for c in ("n_assessable", *VOLTVAR_STATUS_COLUMNS)
    )
    query = f"""
        SELECT serial, {sum_cols}
        FROM read_parquet({sql_string(str(path))})
        GROUP BY serial
        ORDER BY serial
    """
    frame = _run_query(config, query)
    frame["n_conformance"] = frame[list(VOLTVAR_CONFORMANCE_COLUMNS)].sum(axis=1)
    frame["n_non_conformance"] = frame[list(VOLTVAR_NON_CONFORMANCE_COLUMNS)].sum(axis=1)

    n_assessable = frame["n_assessable"].astype("Float64")
    safe_denominator = n_assessable.mask(n_assessable == 0)
    frame["conformance_fraction"] = (
        frame["n_conformance"].astype("Float64") / safe_denominator
    )

    assessable_mask = frame["n_assessable"] > 0
    site_status = pd.Series("not_assessable", index=frame.index, dtype="object")
    site_status.loc[assessable_mask] = frame.loc[assessable_mask, "conformance_fraction"].apply(
        lambda fraction: "conformant" if fraction >= conformance_threshold else "non_conformant"
    )
    frame["site_status"] = site_status
    frame["conformance_threshold"] = conformance_threshold
    return frame


# ---------------------------------------------------------------------------
# Volt-Watt views
# ---------------------------------------------------------------------------


def voltwatt_denominator_view(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    mechanism: MechanismAnalysisConfig | None = None,
    dimensions: Sequence[str] = (),
) -> pd.DataFrame:
    dims = _validate_dimensions(dimensions, CURVE_ALLOWED_DIMENSIONS, "voltwatt denominator")
    path = _require_file(
        voltwatt_results_path(config, scope, mechanism), "Volt-Watt proxy results"
    )
    sum_cols = ",\n            ".join(
        f"sum({c}) AS {c}" for c in ("n_source_intervals", *VOLTWATT_DENOMINATOR_COLUMNS)
    )
    query = f"""
        SELECT
            {_select_prefix(dims)}{sum_cols}
        FROM read_parquet({sql_string(str(path))})
        {_group_by_clause(dims)}
        ORDER BY {", ".join(quote_identifier(d) for d in dims) or "1"}
    """
    frame = _run_query(config, query)
    return _add_fractions(frame, "n_source_intervals", VOLTWATT_DENOMINATOR_COLUMNS, "source")


def voltwatt_status_view(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    mechanism: MechanismAnalysisConfig | None = None,
    dimensions: Sequence[str] = (),
    minimum_denominator: int = 0,
) -> pd.DataFrame:
    """Never relabel ``proxy_does_not_exceed_curve_ceiling`` as conformance:
    household load can suppress net export below the ceiling regardless of
    inverter behaviour, so this column name is preserved verbatim.
    """

    dims = _validate_dimensions(dimensions, CURVE_ALLOWED_DIMENSIONS, "voltwatt status")
    path = _require_file(
        voltwatt_results_path(config, scope, mechanism), "Volt-Watt proxy results"
    )
    sum_cols = ",\n            ".join(
        f"sum({c}) AS {c}" for c in ("n_assessable", *VOLTWATT_STATUS_COLUMNS)
    )
    query = f"""
        SELECT
            {_select_prefix(dims)}{sum_cols}
        FROM read_parquet({sql_string(str(path))})
        {_group_by_clause(dims)}
        ORDER BY {", ".join(quote_identifier(d) for d in dims) or "1"}
    """
    frame = _run_query(config, query)
    frame = _add_fractions(frame, "n_assessable", VOLTWATT_STATUS_COLUMNS, "assessable")
    frame["low_denominator_warning"] = frame["n_assessable"] < minimum_denominator
    return frame


def fleet_summary_view(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    mechanism_name: str,
    mechanism: MechanismAnalysisConfig | None = None,
) -> pd.Series:
    """One-row fleet-level KPI summary for a Volt-VAr or Volt-Watt track.

    Reports: sites analysed (eligible: solar_only, no battery, no controlled
    load, high-confidence phase mapping, full power coverage -- see
    ``core_site_gate_sql``), total timestamps analysed, sites/timestamps
    that crossed the mechanism's activation threshold ("required a
    response"), and -- Volt-VAr only -- timestamps where a response was
    actually observed (``proxy_curve_status != 'inactive'`` among assessable
    rows, i.e. ``n_responded`` on the Volt-VAr table).

    Volt-Watt has no analogous "had a response" figure and this function
    deliberately does not fabricate one: being below the Volt-Watt ceiling
    is not evidence of curtailment (there is no counterfactual P to compare
    against -- see the table's own ``interpretation_guardrail`` column), so
    ``n_timestamps_with_response``/``pct_timestamps_with_response`` are
    returned as ``None`` for Volt-Watt.
    """

    is_voltwatt = (
        mechanism_name.lower().startswith("volt-watt")
        or mechanism_name.lower().startswith("voltwatt")
    )
    path_fn = voltwatt_results_path if is_voltwatt else voltvar_results_path
    path = _require_file(path_fn(config, scope, mechanism), f"{mechanism_name} proxy results")
    query = f"""
        WITH per_site AS (
            SELECT serial,
                sum(n_source_intervals) AS n_source_intervals,
                sum(n_ineligible_site) AS n_ineligible_site,
                sum(n_source_intervals - n_ineligible_site - n_missing_input - n_not_activated)
                    AS n_requiring_response
            FROM read_parquet({sql_string(str(path))})
            GROUP BY serial
        )
        SELECT
            coalesce(count(DISTINCT serial) FILTER (WHERE n_ineligible_site = 0), 0)
                AS n_sites_analyzed,
            coalesce(sum(n_source_intervals) FILTER (WHERE n_ineligible_site = 0), 0)
                AS n_timestamps_analyzed,
            coalesce(count(DISTINCT serial) FILTER (WHERE n_requiring_response > 0), 0)
                AS n_sites_requiring_response,
            coalesce(sum(n_requiring_response), 0) AS n_timestamps_requiring_response
        FROM per_site
    """
    summary = _run_query(config, query).iloc[0]
    n_requiring = int(summary["n_timestamps_requiring_response"])

    n_with_response: int | None = None
    if not is_voltwatt:
        responded = _run_query(
            config,
            f"SELECT coalesce(sum(n_responded), 0) AS n "
            f"FROM read_parquet({sql_string(str(path))})",
        ).iloc[0]
        n_with_response = int(responded["n"])

    return pd.Series(
        {
            "mechanism": mechanism_name,
            "n_sites_analyzed": int(summary["n_sites_analyzed"]),
            "n_timestamps_analyzed": int(summary["n_timestamps_analyzed"]),
            "n_sites_requiring_response": int(summary["n_sites_requiring_response"]),
            "n_timestamps_requiring_response": n_requiring,
            "n_timestamps_with_response": n_with_response,
            "pct_timestamps_with_response": (
                round(100.0 * n_with_response / n_requiring, 2)
                if n_with_response is not None and n_requiring
                else None
            ),
        }
    )


# ---------------------------------------------------------------------------
# Response-observability views
# ---------------------------------------------------------------------------


def observability_status_view(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    mechanism: MechanismAnalysisConfig,
    dimensions: Sequence[str] = (),
) -> pd.DataFrame:
    """Volt-VAr and Volt-Watt observability statuses, kept in separate
    ``n_voltvar_status_*`` / ``n_voltwatt_status_*`` columns so the two
    mechanisms are never merged into one response label.
    """

    dims = _validate_dimensions(
        dimensions, OBSERVABILITY_ALLOWED_DIMENSIONS, "observability status"
    )
    path = _require_file(response_observability_path(config, scope), "response observability")
    vv_cols = ",\n            ".join(
        f"count_if(voltvar_observability_status = {sql_string(status)}) "
        f"AS n_voltvar_status_{status}"
        for status in VOLTVAR_OBSERVABILITY_STATUSES
    )
    vw_cols = ",\n            ".join(
        f"count_if(voltwatt_observability_status = {sql_string(status)}) "
        f"AS n_voltwatt_status_{status}"
        for status in VOLTWATT_OBSERVABILITY_STATUSES
    )
    query = f"""
        SELECT
            {_select_prefix(dims)}count(*) AS n_site_phase_months,
            {vv_cols},
            {vw_cols}
        FROM read_parquet({sql_string(str(path))})
        {_group_by_clause(dims)}
        ORDER BY {", ".join(quote_identifier(d) for d in dims) or "1"}
    """
    frame = _run_query(config, query)
    frame = _add_fractions(
        frame,
        "n_site_phase_months",
        tuple(f"n_voltvar_status_{s}" for s in VOLTVAR_OBSERVABILITY_STATUSES),
        "site_phase_months",
    )
    frame = _add_fractions(
        frame,
        "n_site_phase_months",
        tuple(f"n_voltwatt_status_{s}" for s in VOLTWATT_OBSERVABILITY_STATUSES),
        "site_phase_months",
    )
    frame["active_sign_review_state"] = mechanism.active_sign_review_state
    frame["reactive_sign_review_state"] = mechanism.reactive_sign_review_state
    return frame


def observability_metric_view(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    mechanism: MechanismAnalysisConfig,
    dimensions: Sequence[str] = (),
    minimum_excited_intervals: int = 0,
) -> pd.DataFrame:
    """Excitation-weighted slope/correlation summaries, kept separate for
    Volt-VAr excited intervals and Volt-Watt excited-*export* intervals --
    these are different filters (voltage threshold alone vs voltage
    threshold AND exporting), never a shared denominator name.
    """

    dims = _validate_dimensions(
        dimensions, OBSERVABILITY_ALLOWED_DIMENSIONS, "observability metric"
    )
    path = _require_file(response_observability_path(config, scope), "response observability")
    voltvar_slope = _finite("q_generator_slope_var_per_v")
    voltvar_corr = _finite("q_generator_voltage_correlation")
    voltwatt_slope = _finite("p_export_slope_w_per_v")
    voltwatt_corr = _finite("p_export_voltage_correlation")
    query = f"""
        SELECT
            {_select_prefix(dims)}
            sum(n_voltvar_excited_intervals) AS n_voltvar_excited_intervals,
            sum(n_voltvar_excited_intervals * {voltvar_slope})
                / nullif(sum(n_voltvar_excited_intervals)
                    FILTER (WHERE {voltvar_slope} IS NOT NULL), 0)
                AS voltvar_slope_var_per_v_weighted_mean,
            avg({voltvar_corr})
                FILTER (WHERE n_voltvar_excited_intervals > 0)
                AS voltvar_voltage_correlation_mean,
            avg(voltvar_voltage_span_v)
                FILTER (WHERE n_voltvar_excited_intervals > 0)
                AS voltvar_voltage_span_v_mean,
            min(voltvar_minimum_voltage_v) AS voltvar_minimum_voltage_v,
            max(voltvar_maximum_voltage_v) AS voltvar_maximum_voltage_v,
            sum(n_voltwatt_excited_export_intervals)
                AS n_voltwatt_excited_export_intervals,
            sum(n_voltwatt_excited_export_intervals * {voltwatt_slope})
                / nullif(sum(n_voltwatt_excited_export_intervals)
                    FILTER (WHERE {voltwatt_slope} IS NOT NULL), 0)
                AS voltwatt_slope_w_per_v_weighted_mean,
            avg({voltwatt_corr})
                FILTER (WHERE n_voltwatt_excited_export_intervals > 0)
                AS voltwatt_voltage_correlation_mean,
            avg(voltwatt_voltage_span_v)
                FILTER (WHERE n_voltwatt_excited_export_intervals > 0)
                AS voltwatt_voltage_span_v_mean,
            min(voltwatt_minimum_voltage_v) AS voltwatt_minimum_voltage_v,
            max(voltwatt_maximum_voltage_v) AS voltwatt_maximum_voltage_v
        FROM read_parquet({sql_string(str(path))})
        {_group_by_clause(dims)}
        ORDER BY {", ".join(quote_identifier(d) for d in dims) or "1"}
    """
    frame = _run_query(config, query)
    frame["voltvar_low_denominator_warning"] = (
        frame["n_voltvar_excited_intervals"] < minimum_excited_intervals
    )
    frame["voltwatt_low_denominator_warning"] = (
        frame["n_voltwatt_excited_export_intervals"] < minimum_excited_intervals
    )
    frame["active_sign_review_state"] = mechanism.active_sign_review_state
    frame["reactive_sign_review_state"] = mechanism.reactive_sign_review_state
    return frame


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def validate_result_views(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    mechanism: MechanismAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Reconcile every fleet-level view back to its own source file.

    This does not replace ``mechanism_validation.validate_mechanism_results``
    -- that validates the Delivery 4 *build*. This validates that this
    module's own SQL (grouping, filtering, fraction math) has not silently
    dropped or double-counted rows relative to the parquet file it reads.
    """

    failures: list[str] = []
    checks: dict[str, Any] = {}
    inventory = result_inventory(config, scope, mechanism=mechanism)
    checks["inventory"] = inventory.to_dict(orient="records")

    voltvar_path = voltvar_results_path(config, scope, mechanism)
    if voltvar_path.is_file():
        raw = _raw_column_sums(
            config, voltvar_path, ("n_source_intervals", *VOLTVAR_DENOMINATOR_COLUMNS)
        )
        fleet = voltvar_denominator_view(config, scope, mechanism=mechanism)
        for col, expected in raw.items():
            actual = int(fleet[col].iloc[0])
            if actual != expected:
                failures.append(
                    f"voltvar_denominator_view fleet {col}={actual} != source sum {expected}"
                )
        status = voltvar_status_view(config, scope, mechanism=mechanism)
        classified = sum(int(status[c].iloc[0]) for c in VOLTVAR_STATUS_COLUMNS)
        assessable = int(status["n_assessable"].iloc[0])
        if classified != assessable:
            failures.append(
                f"voltvar_status_view classification sum {classified} != "
                f"n_assessable {assessable}"
            )
        checks["voltvar_n_source_intervals"] = raw["n_source_intervals"]
        checks["voltvar_n_assessable"] = assessable

    voltwatt_path = voltwatt_results_path(config, scope, mechanism)
    if voltwatt_path.is_file():
        raw = _raw_column_sums(
            config, voltwatt_path, ("n_source_intervals", *VOLTWATT_DENOMINATOR_COLUMNS)
        )
        fleet = voltwatt_denominator_view(config, scope, mechanism=mechanism)
        for col, expected in raw.items():
            actual = int(fleet[col].iloc[0])
            if actual != expected:
                failures.append(
                    f"voltwatt_denominator_view fleet {col}={actual} != source sum {expected}"
                )
        status = voltwatt_status_view(config, scope, mechanism=mechanism)
        classified = sum(int(status[c].iloc[0]) for c in VOLTWATT_STATUS_COLUMNS)
        assessable = int(status["n_assessable"].iloc[0])
        if classified != assessable:
            failures.append(
                f"voltwatt_status_view classification sum {classified} != "
                f"n_assessable {assessable}"
            )
        checks["voltwatt_n_source_intervals"] = raw["n_source_intervals"]
        checks["voltwatt_n_assessable"] = assessable

    response_path = response_observability_path(config, scope)
    if response_path.is_file():
        connection = connect(config)
        try:
            (raw_count,) = connection.execute(
                f"SELECT count(*) FROM read_parquet({sql_string(str(response_path))})"
            ).fetchone()
        finally:
            connection.close()
        observability_mechanism = mechanism or MechanismAnalysisConfig()
        status_view = observability_status_view(config, scope, mechanism=observability_mechanism)
        view_count = int(status_view["n_site_phase_months"].iloc[0])
        if view_count != int(raw_count):
            failures.append(
                f"observability_status_view n_site_phase_months={view_count} != "
                f"source row count {int(raw_count)}"
            )
        checks["response_observability_rows"] = int(raw_count)

    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": checks,
        "scope": scope.label,
        "phase_scope_basis": (mechanism or MechanismAnalysisConfig()).phase_scope_basis,
    }
