"""
Phase 3a: what are the circuits, and which of them are already aggregates?
============================================================================

Two separate questions, kept separate on purpose:

1. `summarise_circuit_types` / `flag_suspected_aggregates` / `cohort_completeness`
   describe the fleet -- how many circuits of each type, at how many sites,
   and which names LOOK like they might be whole-site aggregates.

2. `check_aggregation` PROVES or disproves that hypothesis, for one candidate
   at one site: does its own reading equal the sum of the components' readings,
   interval by interval? A name like `ac_load_net` is a hypothesis, not
   evidence -- summing a circuit that already IS the sum of its siblings
   double-counts every one of them, silently, and the result still looks like
   a plausible load curve. That is the single highest-risk defect in this
   build (see `ami_config.py`'s CIRCUIT -> SIGNAL MAPPING section), which is
   why this module never lets a name alone decide `AGGREGATE_CIRCUIT_TYPES`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "summarise_circuit_types", "flag_suspected_aggregates", "cohort_completeness",
    "signal_coverage_summary",
    "AggregationCheck", "check_aggregation", "pick_aggregation_test_site",
    "find_duplicate_circuits", "find_inactive_circuits", "sites_missing_day_data",
    "grouping_keys_agree", "circuits_grouped_by_device",
]


def summarise_circuit_types(
    raw_counts: pd.DataFrame, *,
    type_column: str = "circuit_type",
    is_pv_column: str = "is_pv",
    circuit_column: str = "n_circuits",
) -> pd.DataFrame:
    """
    Tidy + share-annotate a circuit_type x is_pv census. Pure.

    Expects one row per (circuit_type, is_pv) with counts already grouped by
    the caller's query -- `meta_up23c` is small (423,990 rows), so the GROUP
    BY belongs in SQL, not pandas. Adds `share_within_is_pv` so a rare
    circuit_type doesn't hide inside a fleet-wide total.
    """
    if raw_counts is None or not len(raw_counts):
        return pd.DataFrame()
    frame = raw_counts.copy()
    totals = frame.groupby(is_pv_column)[circuit_column].transform("sum")
    frame["share_within_is_pv"] = frame[circuit_column] / totals
    return frame.sort_values(
        [is_pv_column, circuit_column], ascending=[True, False]
    ).reset_index(drop=True)


#: Substrings that suggest a circuit_type might already be a whole-site (or
#: whole-is_pv-side) aggregate rather than one sub-load or sub-generator.
#: A HYPOTHESIS GENERATOR ONLY. `check_aggregation` is what actually proves or
#: disproves it -- nothing in this module should let a name on this list
#: become an entry in `ami_config.AGGREGATE_CIRCUIT_TYPES` on its own.
_SUSPECT_AGGREGATE_PATTERNS = ("net", "site", "total", "combined", "whole", "main")


def flag_suspected_aggregates(
    census: pd.DataFrame, *, type_column: str = "circuit_type"
) -> pd.DataFrame:
    """
    Name-pattern hypothesis for which circuit_types might be aggregates. Pure.

    Returns `census` with a `suspected_aggregate` column added. Nothing is
    excluded or decided here -- this exists to give `check_aggregation` a
    prioritised list of candidates to actually test, not to replace the test.
    """
    if census is None or not len(census):
        return census
    pattern = re.compile("|".join(_SUSPECT_AGGREGATE_PATTERNS), re.IGNORECASE)
    out = census.copy()
    out["suspected_aggregate"] = out[type_column].astype(str).str.contains(pattern)
    return out


def cohort_completeness(
    meta: pd.DataFrame, *,
    site_column: str = "site_id",
    type_column: str = "circuit_type",
    circuit_column: str = "circuit_id",
) -> pd.DataFrame:
    """
    Per-site circuit_type inventory: one row per site, one column per
    circuit_type, values = how many circuits of that type the site has. Pure.

    This is what turns "is `ac_load_net` an aggregate?" into an answerable
    question: a site needs the suspected-aggregate type AND at least one
    sibling circuit present to test at all (`pick_aggregation_test_site` reads
    this frame for exactly that). It is also the raw material for a later
    build-readiness cut: a site needs at least one PV-side and one load-side
    circuit_type present to be usable at all (`ami_config.CORE_SIGNALS`).
    """
    if meta is None or not len(meta):
        return pd.DataFrame()
    pivot = meta.pivot_table(
        index=site_column, columns=type_column, values=circuit_column,
        aggfunc="count", fill_value=0,
    )
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot.reset_index()


def signal_coverage_summary(cohort: pd.DataFrame, *, pv_types, load_types) -> dict:
    """
    What share of sites have at least one PV-side AND one load-side circuit
    present. Pure. Reads `cohort_completeness`'s per-site pivot.

    This is the fleet-wide version of the question `pick_aggregation_test_site`
    asks for one site -- not "can we test the aggregation hypothesis here" but
    "how much of the fleet is even usable ground truth at all". A site missing
    one whole side of `ami_config.CORE_SIGNALS` cannot serve as ground truth
    for disaggregation no matter how clean its circuits are, so this number
    belongs in the Phase 3 record even though nothing downstream consumes it
    directly yet -- Phase 4/5 scoping needs it.
    """
    pv_types = [t for t in pv_types if t in cohort.columns]
    load_types = [t for t in load_types if t in cohort.columns]
    if cohort is None or not len(cohort) or not pv_types or not load_types:
        return {
            "n_sites": 0 if cohort is None else len(cohort),
            "n_with_both": None, "share_with_both": None,
            "reason": "cohort is empty, or none of pv_types/load_types are present as columns",
        }
    has_pv = cohort[pv_types].sum(axis=1) > 0
    has_load = cohort[load_types].sum(axis=1) > 0
    n_sites = len(cohort)
    n_with_both = int((has_pv & has_load).sum())
    return {
        "n_sites": n_sites,
        "n_with_both": n_with_both,
        "share_with_both": n_with_both / n_sites,
        "n_pv_only": int((has_pv & ~has_load).sum()),
        "n_load_only": int((~has_pv & has_load).sum()),
        "n_neither": int((~has_pv & ~has_load).sum()),
    }


@dataclass(frozen=True)
class AggregationCheck:
    """
    The result of testing one candidate aggregate against its siblings, at one
    site, over one window. Evidence in, verdict out -- mirrors
    `ami_sources.SourceCandidate`: every field traces to a specific comparison,
    not a bare assertion.
    """
    site_id: object
    candidate_type: str
    component_types: tuple
    n_intervals_compared: int
    mean_abs_diff: float
    max_abs_diff: float
    reference_scale: float          # typical component magnitude, for judging "small"
    is_aggregate: bool | None       # None = inconclusive -- see `reason`
    reason: str


def check_aggregation(
    long: pd.DataFrame, *,
    site_id,
    candidate_circuit_ids,
    component_circuit_ids,
    candidate_type: str,
    component_types,
    time_column: str = "t_stamp",
    circuit_column: str = "circuit_id",
    power_column: str = "power_signed",
    mean_relative_tolerance: float = 0.02,
    max_relative_tolerance: float = 0.10,
    min_intervals: int = 20,
) -> "AggregationCheck":
    """
    Does the candidate's reading equal the sum of the components', interval
    by interval? Pure -- `long` is real `ts` rows already pulled and already
    polarity-corrected (`power_signed = power * circuit_polarity`); this
    function runs no queries.

    This is the check the brief requires INSTEAD OF trusting the name "net": a
    genuine duplicate aggregate matches its components almost exactly, EVERY
    interval, not just on average. Matching only on average is what two
    independent circuits that happen to be similarly sized would also do --
    which is why both `mean_abs_diff` and `max_abs_diff` must be small
    relative to the components' own typical magnitude for `is_aggregate` to
    come back True. Either alone is not proof.
    """
    component_types = tuple(component_types)
    if time_column not in long.columns or circuit_column not in long.columns:
        return AggregationCheck(
            site_id, candidate_type, component_types, 0,
            float("nan"), float("nan"), float("nan"), None,
            "input frame is missing required columns",
        )

    candidate = (
        long[long[circuit_column].isin(candidate_circuit_ids)]
        .groupby(time_column)[power_column].sum()
    )
    components = (
        long[long[circuit_column].isin(component_circuit_ids)]
        .groupby(time_column)[power_column].sum()
    )
    aligned = pd.concat({"candidate": candidate, "components": components}, axis=1).dropna()
    n = len(aligned)

    if n < min_intervals:
        return AggregationCheck(
            site_id, candidate_type, component_types, n,
            float("nan"), float("nan"), float("nan"), None,
            f"only {n} overlapping interval(s) -- need at least {min_intervals} to conclude anything",
        )

    diff = (aligned["candidate"] - aligned["components"]).abs()
    reference_scale = float(aligned["components"].abs().mean())
    mean_abs_diff = float(diff.mean())
    max_abs_diff = float(diff.max())

    if reference_scale == 0:
        is_aggregate = max_abs_diff < 1e-9
        reason = (
            "components are all ~zero and candidate matches (both idle)"
            if is_aggregate else
            "components are all ~zero but candidate is not -- not an aggregate of these"
        )
    else:
        mean_ok = mean_abs_diff <= mean_relative_tolerance * reference_scale
        max_ok = max_abs_diff <= max_relative_tolerance * reference_scale
        is_aggregate = bool(mean_ok and max_ok)
        reason = (
            f"mean diff {mean_abs_diff:.3g} ({mean_abs_diff / reference_scale:.1%} of typical "
            f"component magnitude), max diff {max_abs_diff:.3g} "
            f"({max_abs_diff / reference_scale:.1%}) over {n} interval(s) -- "
            + ("within tolerance: candidate == sum(components)" if is_aggregate else
               "NOT within tolerance: candidate is an independent circuit, not a duplicate")
        )

    return AggregationCheck(
        site_id, candidate_type, component_types, n,
        mean_abs_diff, max_abs_diff, reference_scale, is_aggregate, reason,
    )


def find_duplicate_circuits(
    long: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    time_column: str = "t_stamp",
    power_column: str = "power",
    correlation_threshold: float = 0.99,
    min_overlap: int = 20,
) -> pd.DataFrame:
    """
    Which circuit pairs, AT THE SAME SITE, read essentially the same physical
    signal (up to a sign flip)? Pure -- no query, no plotting.

    Deliberately correlates the RAW `power_column` (default `"power"`), not
    `power_signed`: `circuit_polarity` is itself a per-circuit correction
    guessed from the circuit's own name/type, so comparing already-corrected
    values could mask a duplicate whose corrections happen to differ, or
    manufacture a false one whose corrections happen to agree. Correlating
    the raw meter reading is blind to that choice.

    This is a different failure mode from `check_aggregation`: that asks
    "is the candidate the SUM of several named siblings" (a whole-site
    aggregate). This asks "are these two circuit_ids reading the SAME
    physical circuit" (a data/tagging duplicate) -- found in real data as a
    `load`-tagged circuit whose raw reading was identical to a `pv`-tagged
    circuit's, just carrying an opposite `circuit_polarity`. Summing that
    "load" into `gross_load` would double-count the PV signal with a sign
    error, not add a genuinely separate appliance.

    A correlation this close to +-1 across a real day's noisy readings is not
    what two independent circuits do by chance, so a hit here is worth
    inspecting even before deciding what it means (two CTs on one physical
    circuit, a metadata mix-up, or a genuine near-perfect antiphase
    relationship) -- this function only flags it, it does not explain it.
    """
    if long is None or not len(long):
        return pd.DataFrame(
            columns=["site_id", "circuit_id_a", "circuit_id_b", "correlation",
                     "sign", "n_overlap"]
        )

    results = []
    for site_id, site_frame in long.groupby(site_column):
        wide = site_frame.pivot_table(
            index=time_column, columns=circuit_column, values=power_column
        )
        ids = wide.columns.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                paired = wide[[ids[i], ids[j]]].dropna()
                if len(paired) < min_overlap:
                    continue
                a, b = paired.iloc[:, 0], paired.iloc[:, 1]
                if a.std() == 0 or b.std() == 0:
                    continue
                corr = float(a.corr(b))
                if abs(corr) >= correlation_threshold:
                    results.append({
                        "site_id": site_id,
                        "circuit_id_a": ids[i], "circuit_id_b": ids[j],
                        "correlation": corr,
                        "sign": "same" if corr > 0 else "opposite",
                        "n_overlap": len(paired),
                    })
    return pd.DataFrame(results)


def find_inactive_circuits(
    long: pd.DataFrame, *,
    circuit_column: str = "circuit_id",
    power_column: str = "power",
    inactive_threshold_w: float = 5.0,
) -> pd.DataFrame:
    """
    Which circuit_ids show essentially NO activity at all in this sample --
    a likely-retired/placeholder circuit_id, not a live meter. Pure -- no
    query, no plotting.

    The companion check to `find_duplicate_circuits` for "why does this site
    have 3 circuits of one type": a duplicate is a circuit_id that IS live
    but mirrors another; this catches the other real pattern seen in the
    fleet -- extra circuit_ids left in `meta_up23c` after a meter or logger
    was re-registered, which are simply dead (near-enough-constant zero, no
    real reading ever recorded), not a genuine second measurement point to
    reconcile.

    Deliberately checks RAW `power` (not `power_signed`): whether a circuit
    ever moves at all does not depend on its `circuit_polarity` sign
    correction -- a dead circuit reads ~0 regardless of which sign convention
    would apply to it.

    `inactive_threshold_w` should be well below any circuit's real load --
    the default (5W) is chosen to clear ordinary standby/CT noise, which
    `find_duplicate_circuits`'s and the PV night-diagnostic's evidence both
    put in the tens of Watts, while still catching a circuit whose reading
    never leaves a few Watts of zero.
    """
    if long is None or not len(long):
        return pd.DataFrame(columns=[circuit_column, "max_abs_power", "inactive"])
    grouped = long.groupby(circuit_column)[power_column].apply(lambda s: float(s.abs().max()))
    out = grouped.reset_index()
    out.columns = [circuit_column, "max_abs_power"]
    out["inactive"] = out["max_abs_power"] < inactive_threshold_w
    return out.sort_values(circuit_column).reset_index(drop=True)


def sites_missing_day_data(
    test_site_ids, meta: pd.DataFrame, reporting_circuit_ids, *,
    candidate_type: str, component_types,
    site_column: str = "site_id",
    type_column: str = "circuit_type",
    circuit_column: str = "circuit_id",
) -> dict:
    """
    Which of `test_site_ids` have NO chance of a conclusive aggregation check
    on the target day, because neither the candidate nor any component
    circuit_id actually reported a single row -- {site_id: reason}. Pure.

    This is a DIFFERENT failure mode from `check_aggregation`'s own
    inconclusive reason ("N overlapping intervals, need at least M"): that
    one means data existed for both sides but didn't line up (a partial-day
    gap); this one means the day was hopeless before any per-site query ran
    -- `meta_up23c` lists the circuit, but it never reported on this day at
    all (real fleet churn: sites join/leave, meters get swapped). A full
    real run found 83 of 200 tested sites came back inconclusive, indistin-
    guishable in the results table from a genuine partial-overlap case --
    this function exists to tell the two apart, and let the caller skip the
    (wasted) per-site query for sites already known to be hopeless.

    `reporting_circuit_ids` is the set of circuit_ids confirmed (by one
    cheap presence query, run once for the whole batch of `test_site_ids`
    rather than once per site) to have at least one row on the target day.
    """
    reporting = set(reporting_circuit_ids)
    component_types = list(component_types)
    result = {}
    for site_id in test_site_ids:
        site_meta = meta[meta[site_column] == site_id]
        candidate_ids = site_meta[site_meta[type_column] == candidate_type][circuit_column].tolist()
        component_ids = site_meta[site_meta[type_column].isin(component_types)][circuit_column].tolist()
        candidate_has_data = any(c in reporting for c in candidate_ids)
        component_has_data = any(c in reporting for c in component_ids)
        if not candidate_has_data or not component_has_data:
            missing = []
            if not candidate_has_data:
                missing.append("candidate")
            if not component_has_data:
                missing.append("component(s)")
            result[site_id] = (
                f"no {' and '.join(missing)} data reported this day "
                "(listed in meta_up23c, but zero `ts` rows this day)"
            )
    return result


def pick_aggregation_test_site(
    cohort: pd.DataFrame, *,
    candidate_type: str,
    component_types,
    site_column: str = "site_id",
    min_components: int = 1,
) -> list:
    """
    Which sites actually have both the suspected aggregate AND at least
    `min_components` of its proposed siblings present. Pure.

    Reads `cohort_completeness`'s output. Returns site_ids ordered by total
    sibling-circuit count descending (more siblings present = a stronger test,
    since a match across more components is harder to get by coincidence),
    so the notebook can just take the first one rather than eyeballing a table.
    """
    component_types = list(component_types)
    if cohort is None or not len(cohort):
        return []
    if candidate_type not in cohort.columns:
        return []
    present_components = [c for c in component_types if c in cohort.columns]
    if not present_components:
        return []

    has_candidate = cohort[candidate_type] > 0
    n_components = cohort[present_components].sum(axis=1)
    qualifying = cohort[has_candidate & (n_components >= min_components)].copy()
    qualifying["_n_components"] = n_components[has_candidate & (n_components >= min_components)]
    qualifying = qualifying.sort_values("_n_components", ascending=False)
    return qualifying[site_column].tolist()


def grouping_keys_agree(meta: pd.DataFrame, *, key_a: str, key_b: str) -> dict:
    """
    Do `key_a` and `key_b` partition the rows of `meta` the SAME way? Pure.

    Exists to answer "what actually IS this column, and does it agree with
    a column we already understand" without guessing -- e.g. `meta_up23c`'s
    `m_id` next to the already-used `device_id`. If every distinct `key_a`
    value maps to exactly one `key_b` value and vice versa, the two columns
    are the same grouping under different names. If they disagree, the
    counts here show how much, which is a real clue to what `key_a` means
    (a coarser or finer grouping than `key_b`) rather than a guess.
    """
    if meta is None or not len(meta) or key_a not in meta.columns or key_b not in meta.columns:
        return {"agree": None, "reason": f"missing {key_a!r} or {key_b!r} column, or empty meta"}
    valid = meta[[key_a, key_b]].dropna()
    if not len(valid):
        return {"agree": None, "reason": f"no rows with both {key_a!r} and {key_b!r} present"}

    a_to_b = valid.groupby(key_a)[key_b].nunique()
    b_to_a = valid.groupby(key_b)[key_a].nunique()
    n_a_with_multiple_b = int((a_to_b > 1).sum())
    n_b_with_multiple_a = int((b_to_a > 1).sum())
    agree = (n_a_with_multiple_b == 0) and (n_b_with_multiple_a == 0)
    return {
        "agree": agree,
        "n_distinct_a": int(valid[key_a].nunique()),
        "n_distinct_b": int(valid[key_b].nunique()),
        "n_a_with_multiple_b": n_a_with_multiple_b,
        "n_b_with_multiple_a": n_b_with_multiple_a,
        "reason": (
            f"{key_a} <-> {key_b}: "
            + (f"every {key_a} maps to exactly one {key_b} and vice versa -- same grouping"
               if agree else
               f"{n_a_with_multiple_b} {key_a} value(s) span more than one {key_b}, "
               f"{n_b_with_multiple_a} {key_b} value(s) span more than one {key_a} -- "
               "these are different groupings, not the same key twice")
        ),
    }


def circuits_grouped_by_device(
    meta: pd.DataFrame, *,
    candidate_type: str,
    site_column: str = "site_id",
    type_column: str = "circuit_type",
    circuit_column: str = "circuit_id",
    device_column: str = "device_id",
) -> pd.DataFrame:
    """
    For every site with MORE THAN ONE `candidate_type` circuit_id, do those
    circuit_ids share one `device_id` (consistent with separately-metered
    phases/channels of one physical device) or span multiple `device_id`s
    (consistent with independent/duplicate registrations, or genuinely
    separate sub-panels)? Pure.

    This is the metadata-first companion to `find_duplicate_circuits` (which
    tests BEHAVIOUR -- do the readings correlate): `device_id` is what the
    installer/registration process actually recorded, so agreement between
    the two is strong corroborating evidence, and disagreement (e.g. a
    correlated pair under two different device_ids) is worth a closer look
    rather than trusting either signal alone.
    """
    empty = pd.DataFrame(columns=[site_column, "n_circuits", "n_distinct_devices", "single_device"])
    if meta is None or not len(meta) or device_column not in meta.columns:
        return empty
    candidates = meta[meta[type_column] == candidate_type]
    if not len(candidates):
        return empty
    counts = candidates.groupby(site_column)[circuit_column].nunique()
    multi_sites = counts[counts > 1].index
    if not len(multi_sites):
        return empty

    subset = candidates[candidates[site_column].isin(multi_sites)]
    grouped = subset.groupby(site_column).agg(
        n_circuits=(circuit_column, "nunique"),
        n_distinct_devices=(device_column, "nunique"),
    ).reset_index()
    grouped["single_device"] = grouped["n_distinct_devices"] == 1
    return grouped.sort_values(site_column).reset_index(drop=True)
