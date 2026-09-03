"""
Phase 4: per-site circuit resolution -- turning a messy meta_up23c/ts pull
into the two ground-truth tables (interval-level + site metadata sidecar).
============================================================================

Phase 3 proved WHAT needs resolving (device_id phase-grouping, duplicate/
inactive circuits, a per-circuit device/meter-model power-derivation
correction) and HOW OFTEN (roughly 1 in 5 `CLEAN_SITE_IDS` sites has a
duplicate `ac_load_net`/`pv_site_net` pair, 2% an inactive one, ~74% of
`CATCH Power` circuits need the `energy/implied_interval_minutes`
correction). This module turns those findings into a deterministic,
auditable per-site decision -- every circuit kept or dropped, and why --
rather than one-off notebook code.

Three decisions this module makes that Phase 3 did NOT establish by itself,
made explicit here because guessing wrong would be a silent correctness bug:

- A CROSS-type duplicate (one `ac_load_net` + one `pv_site_net` circuit
  reading the same physical signal) drops the LOAD-side member and keeps
  the PV-tagged one -- this mirrors the real `load_other`/`pv_site_net`
  example Phase 3 found (site 2141096187): the duplicate there was a
  load-side mistagging of a PV reading, not the other way around.
- A SAME-type duplicate group (e.g. three `pv_site_net` circuit_ids all
  pairwise-correlated) has no such asymmetry to lean on -- there is no
  metadata signal for which registration is "the real one". This module
  keeps the circuit with the most reporting coverage in the sample (ties
  broken by lowest `circuit_id`, for determinism) and marks the SITE
  `needs_manual_review=True` rather than silently trusting the tie-break.
- Reactive power has no raw instantaneous column at all in `ts` (it only
  ever logs `energy_reactive`, never a `power_reactive`), so it is derived
  the same way for every kept circuit, not just device/meter-model-flagged
  ones: `energy_reactive * 60 / interval_minutes`, using that circuit's own
  `implied_interval_minutes` where `ami_resample.energy_granularity_and_implied_interval`
  flags it, otherwise the nominal interval -- the same interval basis each
  circuit's corrected active power uses, since both are accumulated by the
  same physical meter over the same window.
"""

from __future__ import annotations

import pandas as pd

from . import ami_resample as Resample
from . import ami_taxonomy as Taxonomy

__all__ = [
    "classify_circuit_counts",
    "resolve_site_circuits",
    "exclude_flagged_sites",
    "build_interval_table",
    "build_site_metadata",
    "build_coverage_report",
]


def classify_circuit_counts(
    counts_by_site: pd.DataFrame, *,
    circuit_count_column: str = "n_circuits",
    site_column: str = "site_id",
    clean_counts=(1, 3),
) -> tuple[list, list]:
    """
    Split site_ids into CLEAN_SITE_IDS (an unambiguous 1 or 3 count of the
    load-side candidate type) and OTHER_COUNT_SITE_IDS (everything else).
    Pure -- lifts notebook 3 section 3d's inline logic into a reusable,
    tested function rather than copy-pasted notebook code.

    `counts_by_site` is one row per site with a count column already
    restricted to circuits > 0 (mirrors `pv_cohort`'s per-site count in
    notebook 3) -- this function only classifies, it does not query or
    filter the zero-count rows itself.
    """
    if counts_by_site is None or not len(counts_by_site):
        return [], []
    clean_counts = set(clean_counts)
    is_clean = counts_by_site[circuit_count_column].isin(clean_counts)
    clean_site_ids = counts_by_site.loc[is_clean, site_column].tolist()
    other_site_ids = counts_by_site.loc[~is_clean, site_column].tolist()
    return clean_site_ids, other_site_ids


def resolve_site_circuits(
    meta: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    type_column: str = "circuit_type",
    device_column: str = "device_id",
    is_pv_column: str = "is_pv",
    power_column: str = "power",
    time_column: str = "t_stamp",
    load_type: str = "ac_load_net",
    pv_type: str = "pv_site_net",
    correlation_threshold: float = 0.99,
    min_overlap: int = 20,
    inactive_threshold_w: float = 5.0,
    granularity_share_threshold: float = 0.9,
    granularity_interval_tolerance: float = 0.01,
    nominal_interval_minutes: float = 5.0,
) -> pd.DataFrame:
    """
    Per-circuit keep/drop resolution for every `load_type`/`pv_type` circuit
    present in `meta`, across as many sites as `sample` covers in one call
    (mirrors notebook 3 section 8b's batched-not-per-site-loop pattern --
    `find_duplicate_circuits`/`find_inactive_circuits` already group
    internally by site/circuit). Pure -- `meta` and `sample` are
    already-pulled frames, no query here.

    Returns one row per (site_id, circuit_id) with:
      - device_id, circuit_type
      - kept (bool)
      - drop_reason: None, "duplicate_cross_type", "duplicate_same_type", or
        "inactive"
      - needs_manual_review (bool) -- True only where a same-type duplicate
        group was involved (including the circuit that was KEPT from that
        group), since the tie-break is a coverage heuristic, not a rule
        Phase 3 actually established (see module docstring)
      - power_correction_applied (bool), implied_interval_minutes (float or
        NaN) -- from `ami_resample.energy_granularity_and_implied_interval`,
        computed only among surviving circuits. Feeds BOTH the active-power
        correction (only applied where flagged) and the reactive-power
        derivation (applied to every kept circuit; see `build_interval_table`)

    Order of operations mirrors the settled Phase 4 pipeline: duplicates are
    resolved before inactivity is checked (a circuit that is both a
    duplicate AND inactive is reported as a duplicate -- the more specific,
    more diagnostic reason), and the device/meter-model correction is
    computed only for circuits that survive both, never for one already
    dropped.
    """
    candidate_types = [load_type, pv_type]
    candidates = meta[meta[type_column].isin(candidate_types)]
    base_cols = [site_column, circuit_column, type_column, device_column, is_pv_column]
    base = candidates[base_cols].drop_duplicates(circuit_column).reset_index(drop=True)

    empty_columns = [
        site_column, circuit_column, type_column, device_column,
        "kept", "drop_reason", "needs_manual_review",
        "power_correction_applied", "implied_interval_minutes",
    ]
    if not len(base):
        return pd.DataFrame(columns=empty_columns)

    result = base.copy()
    result["kept"] = True
    result["drop_reason"] = None
    result["needs_manual_review"] = False

    sample_candidates = sample[sample[circuit_column].isin(base[circuit_column])]

    is_pv_lookup = base.set_index(circuit_column)[is_pv_column]
    coverage = sample_candidates.groupby(circuit_column)[time_column].nunique()

    dropped: dict = {}  # circuit_id -> (reason, needs_review)
    review_circuit_ids: set = set()

    dupes = Taxonomy.find_duplicate_circuits(
        sample_candidates, site_column=site_column, circuit_column=circuit_column,
        time_column=time_column, power_column=power_column,
        correlation_threshold=correlation_threshold, min_overlap=min_overlap,
    )
    if len(dupes):
        parent: dict = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        cross_type_pairs = []
        for _, row in dupes.iterrows():
            a, b = row.circuit_id_a, row.circuit_id_b
            if bool(is_pv_lookup.get(a)) != bool(is_pv_lookup.get(b)):
                cross_type_pairs.append((a, b))
            else:
                union(a, b)

        # Cross-type: deterministic rule, no ambiguity -- drop the load-side
        # member, keep the PV-tagged one (mirrors the real fleet example).
        for a, b in cross_type_pairs:
            load_id = a if not is_pv_lookup.get(a) else b
            dropped.setdefault(load_id, ("duplicate_cross_type", False))

        # Same-type: group by union-find root, keep the most-covered member.
        groups: dict = {}
        for cid in list(parent):
            groups.setdefault(find(cid), []).append(cid)
        for members in groups.values():
            if len(members) < 2:
                continue
            still_alive = [m for m in members if m not in dropped]
            if len(still_alive) < 2:
                continue
            ranked = sorted(still_alive, key=lambda c: (-int(coverage.get(c, 0)), c))
            keeper = ranked[0]
            for loser in ranked[1:]:
                dropped[loser] = ("duplicate_same_type", True)
            review_circuit_ids.add(keeper)

    survivors = sample_candidates[~sample_candidates[circuit_column].isin(dropped)]
    inactive = Taxonomy.find_inactive_circuits(
        survivors, circuit_column=circuit_column, power_column=power_column,
        inactive_threshold_w=inactive_threshold_w,
    )
    for cid in inactive.loc[inactive.inactive, circuit_column]:
        dropped.setdefault(cid, ("inactive", False))

    for cid, (reason, needs_review) in dropped.items():
        mask = result[circuit_column] == cid
        result.loc[mask, "kept"] = False
        result.loc[mask, "drop_reason"] = reason
        result.loc[mask, "needs_manual_review"] = needs_review
    for cid in review_circuit_ids:
        result.loc[result[circuit_column] == cid, "needs_manual_review"] = True

    kept_ids = result.loc[result.kept, circuit_column]
    granularity = Resample.energy_granularity_and_implied_interval(
        sample_candidates[sample_candidates[circuit_column].isin(kept_ids)],
        circuit_column=circuit_column, power_column=power_column,
    )
    flagged_ids: set = set()
    implied_lookup: dict = {}
    if len(granularity):
        implied_lookup = granularity.set_index(circuit_column)["implied_interval_minutes"].to_dict()
        flagged = granularity[
            (granularity.share_integer_energy > granularity_share_threshold)
            & (
                (granularity.implied_interval_minutes - nominal_interval_minutes).abs()
                / nominal_interval_minutes > granularity_interval_tolerance
            )
        ]
        flagged_ids = set(flagged[circuit_column])

    result["power_correction_applied"] = result[circuit_column].isin(flagged_ids)
    result["implied_interval_minutes"] = result[circuit_column].map(implied_lookup)

    return result.reset_index(drop=True)


def build_interval_table(
    sample: pd.DataFrame,
    resolution: pd.DataFrame,
    *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    type_column: str = "circuit_type",
    device_column: str = "device_id",
    time_column: str = "t_stamp",
    power_column: str = "power",
    energy_column: str = "energy",
    reactive_energy_column: str = "energy_reactive",
    voltage_column: str = "voltage",
    current_column: str = "current",
    power_factor_column: str = "power_factor",
    energy_import_column: str = "energy_import",
    energy_export_column: str = "energy_export",
    nominal_interval_minutes: float = 5.0,
    apply_power_correction: bool = True,
) -> pd.DataFrame:
    """
    Assemble the core interval-level ground-truth table: one row per
    (circuit_id, t_stamp), timestamps exactly as reported (no forced grid),
    restricted to circuits `resolve_site_circuits` kept. NOT pre-summed
    across phases -- `device_id` is carried as a tag so a consumer can group
    and sum on demand (see the settled Phase 4 decisions).

    Active power: when `apply_power_correction` is True (the default), uses
    raw `power` unless the circuit was flagged for the device/meter-model
    correction, in which case it is re-derived as
    `energy * 60 / implied_interval_minutes` (that circuit's OWN implied
    interval). This flag exists because Phase 3 found real evidence that
    `CATCH Power`-model circuits (identified by `power_correction_applied`,
    from a whole-Wh energy-register diagnostic -- see
    `energy_granularity_and_implied_interval`) report an unreliable raw
    `power` field; re-deriving it from `energy` is the more defensible
    per-circuit value, but because the source energy register only ticks in
    whole Wh, it also makes the reconstructed power visibly stair-stepped
    rather than smooth.

    Set `apply_power_correction=False` to instead use raw `power` for
    EVERY circuit unconditionally, ignoring `power_correction_applied`
    entirely -- this is the same treatment `structured_data`'s own build
    (`Write_structured_data.ipynb`) gives every circuit, so pass this when
    you specifically want AMI-dataset numbers that are comparable to
    `structured_data`'s, at the cost of reintroducing the known-unreliable
    raw reading for flagged circuits. This is a deliberate per-call choice,
    not a permanent decision -- both tables can be built side by side by
    calling `Build.run_build`/`Build.run_phase_split_build` twice, once per
    setting, into different `store_dir`s. See `sites_with_power_correction`
    below for identifying (and optionally excluding) the sites this affects
    -- e.g. to get an AMI dataset with the correction ON but built
    ONLY from sites where it never fires, filter `resolution` down to
    `sites_with_power_correction(resolution)`-False sites before calling
    this (or the `Build.run_*` orchestrators) at all.

    Reactive power has no raw instantaneous column in `ts` at all -- it is
    ALWAYS derived, `energy_reactive * 60 / interval_minutes`, using the
    same interval basis (implied where flagged, nominal otherwise) as the
    active-power correction, REGARDLESS of `apply_power_correction` (there
    is no raw reactive-power field to fall back to). `current`,
    `power_factor`, `energy_import`, and `energy_export` are all carried
    through as landed (raw, uncorrected) -- none of them have an alternate
    derivation the way power/reactive power do, so neither correction
    setting touches them.
    """
    out_columns = [site_column, device_column, circuit_column, type_column,
                   time_column, "power", "reactive_power", voltage_column,
                   current_column, power_factor_column,
                   energy_import_column, energy_export_column]
    kept = resolution[resolution.kept]
    if not len(kept):
        return pd.DataFrame(columns=out_columns)

    overlap = [c for c in (site_column, device_column, type_column) if c in sample.columns]
    frame = sample[sample[circuit_column].isin(kept[circuit_column])].drop(columns=overlap).copy()
    frame = frame.merge(
        kept[[circuit_column, site_column, device_column, type_column,
              "power_correction_applied", "implied_interval_minutes"]],
        on=circuit_column, how="left",
    )

    interval_minutes = frame["implied_interval_minutes"].where(
        frame["power_correction_applied"], nominal_interval_minutes
    )
    if apply_power_correction:
        frame["power"] = frame[power_column].where(
            ~frame["power_correction_applied"],
            frame[energy_column] * 60.0 / interval_minutes,
        )
    else:
        frame["power"] = frame[power_column]
    frame["reactive_power"] = frame[reactive_energy_column] * 60.0 / interval_minutes

    present_columns = [c for c in out_columns if c in frame.columns]
    return frame[present_columns].sort_values([circuit_column, time_column]).reset_index(drop=True)


def sites_with_power_correction(
    resolution: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
) -> pd.Series:
    """
    Per-site rollup of `power_correction_applied`, restricted to KEPT
    circuits: a boolean Series indexed by `site_id`, True if that site has
    AT LEAST ONE kept circuit whose raw `power` field was flagged
    unreliable (see `build_interval_table`'s `apply_power_correction`
    docstring). A site is rolled up with `.any()`, not checked per side,
    because a ground-truth row needs BOTH sides -- one untrustworthy
    circuit (load OR PV) taints the whole site's reconstruction, not just
    that circuit's own columns.

    Use this to build an exclusion list before calling
    `Build.run_build`/`Build.run_phase_split_build`, so the flagged sites
    never enter the AMI tables at all (rather than filtering them out of
    an already-built table after the fact):

        flagged = Resolution.sites_with_power_correction(final_resolution)
        clean_site_ids = flagged[~flagged].index
        clean_resolution = final_resolution[
            final_resolution.site_id.isin(clean_site_ids)
        ]
        # then pass clean_resolution as the `resolution` argument to
        # Build.run_build / Build.run_phase_split_build instead of
        # final_resolution.

    An empty/all-dropped `resolution` returns an empty Series (rather than
    raising), matching this module's other empty-input conventions.
    """
    kept = resolution[resolution.kept]
    if not len(kept):
        return pd.Series(dtype=bool)
    return kept.groupby(site_column)["power_correction_applied"].any()


def build_site_metadata(
    site_level_meta: pd.DataFrame,
    resolution: pd.DataFrame,
    *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    device_column: str = "device_id",
) -> pd.DataFrame:
    """
    One row per site: `site_level_meta`'s curated columns (whatever the
    caller already selected -- this function does not know or care which
    ones) plus a resolution audit trail built from `resolve_site_circuits`'s
    output -- `kept_circuit_ids`, `dropped_circuits` (a list of
    `{circuit_id, reason}`), `device_id_groups`, and `needs_manual_review`.

    Sites present in one input but not the other are dropped by the inner
    join below, not silently unioned -- this function only builds the audit
    trail for sites `site_level_meta` already lists.
    """
    audit_columns = [site_column, "kept_circuit_ids", "dropped_circuits",
                      "device_id_groups", "needs_manual_review"]
    if resolution is None or not len(resolution):
        audit = pd.DataFrame(columns=audit_columns)
    else:
        rows = []
        for site_id, group in resolution.groupby(site_column):
            kept_ids = sorted(int(c) for c in group.loc[group.kept, circuit_column])
            dropped = [
                {"circuit_id": int(row[circuit_column]), "reason": row["drop_reason"]}
                for _, row in group.loc[~group.kept].iterrows()
            ]
            device_groups = {
                int(device_id): sorted(int(c) for c in circuit_ids)
                for device_id, circuit_ids in
                group.loc[group.kept].groupby(device_column)[circuit_column]
            }
            rows.append({
                site_column: site_id,
                "kept_circuit_ids": kept_ids,
                "dropped_circuits": dropped,
                "device_id_groups": device_groups,
                "needs_manual_review": bool(group.needs_manual_review.any()),
            })
        audit = pd.DataFrame(rows)

    if site_level_meta is None or not len(site_level_meta):
        return audit
    return site_level_meta.merge(audit, on=site_column, how="inner")


def exclude_flagged_sites(
    resolution: pd.DataFrame, excluded_site_ids, *,
    site_column: str = "site_id",
    reason: str = "storage_or_sign_issue",
) -> pd.DataFrame:
    """
    Drop every SURVIVING circuit at each site in `excluded_site_ids` --
    the simplest, most conservative response to a site failing
    `ami_signal.evaluate_load_reconstruction`'s night-time check or being
    named by `ami_signal.sites_with_storage_circuits`. Pure.

    Whichever the actual cause (a per-circuit sign/polarity bug, or a real
    battery/EV effect netted into `ac_load_net` behind the same CT), the
    practical consequence is identical for a solar-disaggregation ground
    -truth build: `ac_load_net` is not trustworthy as pure house load at
    that site, so nothing kept there should end up in the final dataset.
    Distinguishing the two causes only matters for salvaging a site later
    (e.g. fixing one circuit's polarity) -- it is NOT needed to justify the
    drop, which is why this function takes one undifferentiated exclusion
    list rather than requiring a diagnosis per site.

    A circuit already dropped for another reason (duplicate, inactive)
    KEEPS that original `drop_reason` -- only circuits that were otherwise
    `kept=True` get overwritten, so the audit trail still shows everything
    that was found at an excluded site, not just the reason it got excluded
    last.
    """
    if resolution is None or not len(resolution):
        return resolution
    excluded_site_ids = set(excluded_site_ids)
    if not excluded_site_ids:
        return resolution.copy()

    out = resolution.copy()
    mask = out[site_column].isin(excluded_site_ids) & out["kept"]
    out.loc[mask, "kept"] = False
    out.loc[mask, "drop_reason"] = reason
    return out


def build_coverage_report(resolution: pd.DataFrame, *, site_column: str = "site_id") -> dict:
    """
    Fleet-wide tally of how each site was resolved: no intervention needed,
    auto-resolved via a cross-type duplicate drop, auto-resolved via an
    inactive-circuit drop, flagged for manual review (a same-type duplicate
    group was involved), or excluded entirely for a storage/sign issue
    (`exclude_flagged_sites`). `needs_manual_review` is reported separately,
    not as a mutually-exclusive bucket, since a site can both have an
    auto-resolved cross-type duplicate AND a flagged same-type one.
    """
    empty = {
        "n_sites": 0, "n_no_intervention": 0,
        "n_auto_resolved_duplicate_cross_type": 0,
        "n_auto_resolved_inactive": 0,
        "n_flagged_manual_review": 0,
        "n_excluded_storage_or_sign_issue": 0,
        "n_excluded_inactive_full_year": 0,
        "n_excluded_storage_or_sign_issue_full_year": 0,
        "n_power_correction_applied": 0,
    }
    if resolution is None or not len(resolution):
        return empty

    by_site = resolution.groupby(site_column)
    reasons_by_site = by_site["drop_reason"].apply(lambda s: set(s.dropna()))
    return {
        "n_sites": by_site.ngroups,
        "n_no_intervention": int((reasons_by_site.apply(len) == 0).sum()),
        "n_auto_resolved_duplicate_cross_type": int(
            reasons_by_site.apply(lambda s: "duplicate_cross_type" in s).sum()
        ),
        "n_auto_resolved_inactive": int(
            reasons_by_site.apply(lambda s: "inactive" in s).sum()
        ),
        "n_flagged_manual_review": int(by_site["needs_manual_review"].any().sum()),
        "n_excluded_storage_or_sign_issue": int(
            reasons_by_site.apply(lambda s: "storage_or_sign_issue" in s).sum()
        ),
        "n_excluded_inactive_full_year": int(
            reasons_by_site.apply(lambda s: "inactive_full_year" in s).sum()
        ),
        "n_excluded_storage_or_sign_issue_full_year": int(
            reasons_by_site.apply(lambda s: "storage_or_sign_issue_full_year" in s).sum()
        ),
        "n_power_correction_applied": int(
            resolution.loc[resolution.kept, "power_correction_applied"].sum()
        ),
    }
