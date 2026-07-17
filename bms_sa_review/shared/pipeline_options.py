"""
Validated run options shared by the Stage 1 and Stage 2 builders.
The project has two different capacity concepts:

rating_basis:
    The capacity used to scale AS/NZS 4777.2 curves, tolerance bands and assessability thresholds. 
    ``ac_capacity_kw`` is the provider field.
    ``s_99`` is an empirical sensitivity case, not a verified S_rated value.

empirical_limit_basis:
    The observed/assumed apparent-power boundary used only for curtailment diagnostics. 
    This defaults to ``s_99``.

The allow-lists below also prevent user-supplied strings being interpolated directly into Athena/Trino SQL.
"""

CAPACITY_COLUMNS = {
    "ac_capacity_kw": "ac_capacity_kw",
    "s_99": "S_99",
}

VOLTAGE_AGGREGATIONS = {
    "avg": "avg",
    "max": "max",
}

FLEX_SELECTIONS = {"exclude", "include", "only"}


def capacity_column(basis):
    """Return the SQL column for a validated capacity-basis label."""
    try:
        return CAPACITY_COLUMNS[basis]
    except KeyError as exc:
        raise ValueError(
            f"capacity basis must be one of {sorted(CAPACITY_COLUMNS)}, got {basis!r}"
        ) from exc


def voltage_aggregate_sql(method, column="voltage"):
    """Return ``avg(column)`` or ``max(column)`` after validating ``method``."""
    try:
        function = VOLTAGE_AGGREGATIONS[method]
    except KeyError as exc:
        raise ValueError(
            f"voltage aggregation must be one of {sorted(VOLTAGE_AGGREGATIONS)}, "
            f"got {method!r}"
        ) from exc
    return f"{function}({column})"


def flex_predicate(selection, column="flex_export_detected"):
    """Return the metadata predicate for the requested flex-export cohort.

    ``exclude`` removes only explicitly true flags. False and NULL are retained.
    ``only`` keeps only explicitly true flags.
    ``include`` applies no flag predicate.
    """
    if selection not in FLEX_SELECTIONS:
        raise ValueError(
            f"flex selection must be one of {sorted(FLEX_SELECTIONS)}, "
            f"got {selection!r}"
        )

    if selection == "exclude":
        return f"coalesce({column}, False) = False"

    if selection == "only":
        return f"{column} = True"

    return "1 = 1"


def labelled_table(base, run_label):
    """Return a safe non-overwriting table name such as ``base_v3_s99``."""
    if not run_label or not run_label.replace("_", "").isalnum():
        raise ValueError("run_label must contain only letters, numbers and underscores")
    return f"{base}_{run_label.lower()}"
