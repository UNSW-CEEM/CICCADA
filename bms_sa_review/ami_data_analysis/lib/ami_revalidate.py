"""
Phase 5, step 3 -- re-check Phase 4's findings against the FULL landed
year, not just the single sampled day the fleet-wide resolution ran on.
=====================================================================

A single day proves the SHAPE of a problem (a duplicate pair, an inactive
circuit, a night-time reconstruction failure) but not its PERSISTENCE: a
circuit can go quiet in month 8, a device can be swapped mid-year, a
battery can start cycling in winter that wasn't installed when the sample
day was pulled. This module re-runs the two checks that can drift --
inactivity and the load-reconstruction/storage check -- across every
landed month, one month at a time, so nothing about the final dataset
depends on a single day being representative of the whole year.

Two different granularities, on purpose, matching Phase 4's own
`ami_resolution` distinction:
  - a circuit newly found inactive in some month is dropped on its own --
    a dead phase doesn't have to take an otherwise-good site down with it.
  - a site newly failing the reconstruction check is excluded ENTIRELY
    (via `ami_resolution.exclude_flagged_sites`), same rationale as
    Section 7b: `ac_load_net` at that site isn't trustworthy as house
    load, whichever month revealed it.

Every function here takes an already-iterated sequence of (year, month,
frame) tuples, never a store path directly for the two revalidation
checks -- `iter_month_partitions` is the one function that actually reads
Parquet, so the checking logic itself is fully unit-testable against
in-memory fixture frames, no real store needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import ami_plots as Plots
from . import ami_resolution as Resolution
from . import ami_signal as Signal
from . import ami_taxonomy as Taxonomy

__all__ = [
    "iter_month_partitions",
    "revalidate_inactive_circuits_over_history",
    "revalidate_reconstruction_over_history",
    "apply_full_year_findings",
]


def iter_month_partitions(
    store_dir, year_month_pairs, *,
    table_name: str = "ami_extract",
    partition_key: str = "dt_month",
    reader=pd.read_parquet,
):
    """
    Yield (year, month, frame) for each (year, month) in `year_month_pairs`,
    reading and concatenating every part file under that month's landed
    partition directory. A month with no landed files yields an EMPTY
    frame, not a skipped one -- a caller counting months checked must not
    silently undercount a month that failed to extract.
    """
    for year, month in year_month_pairs:
        partition_dir = Path(store_dir) / table_name / f"{partition_key}={year}-{month:02d}"
        files = sorted(partition_dir.glob("*.parquet")) if partition_dir.exists() else []
        frame = (
            pd.concat([reader(f) for f in files], ignore_index=True)
            if files else pd.DataFrame()
        )
        yield year, month, frame


def revalidate_inactive_circuits_over_history(
    month_frames, *,
    circuit_column: str = "circuit_id",
    power_column: str = "power",
    inactive_threshold_w: float = 5.0,
) -> pd.DataFrame:
    """
    Which circuit_ids are inactive (max |power| below threshold) in AT
    LEAST ONE landed month, even if they looked fine on Phase 4's single
    sampled day? `month_frames` is an iterable of (year, month, frame) --
    e.g. `iter_month_partitions` -- consumed one at a time, so the full
    year is never held in memory together.

    Returns one row per circuit_id inactive in at least one month:
    `circuit_id, n_months_checked, months_inactive` (sorted "YYYY-MM"
    list). A circuit inactive in zero months does not appear -- this is a
    list of NEW findings to act on, not a full per-circuit census.
    """
    inactive_months: dict[int, list[str]] = {}
    n_months_checked = 0
    for year, month, frame in month_frames:
        n_months_checked += 1
        if frame is None or not len(frame):
            continue
        result = Taxonomy.find_inactive_circuits(
            frame, circuit_column=circuit_column, power_column=power_column,
            inactive_threshold_w=inactive_threshold_w,
        )
        for cid in result.loc[result.inactive, circuit_column]:
            inactive_months.setdefault(int(cid), []).append(f"{year}-{month:02d}")

    columns = [circuit_column, "n_months_checked", "months_inactive"]
    if not inactive_months:
        return pd.DataFrame(columns=columns)
    rows = [
        {circuit_column: cid, "n_months_checked": n_months_checked,
         "months_inactive": sorted(months)}
        for cid, months in inactive_months.items()
    ]
    return pd.DataFrame(rows).sort_values(circuit_column).reset_index(drop=True)


def revalidate_reconstruction_over_history(
    month_frames, resolution: pd.DataFrame, circuit_polarity: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    type_column: str = "circuit_type",
    device_column: str = "device_id",
    time_column: str = "t_stamp",
    night_hour_start: int = 1,
    night_hour_end: int = 4,
    negative_threshold_w: float = -100.0,
) -> pd.DataFrame:
    """
    Which site_ids fail `ami_signal.evaluate_load_reconstruction`'s
    night-time check in AT LEAST ONE landed month? `month_frames` as
    above -- consumed one at a time.

    Each month's kept-circuit interval table is built via
    `ami_resolution.build_interval_table` from that month's raw frame plus
    the (already resolved, day-sample) `resolution`, then reconstructed
    and evaluated exactly as Section 7 does for one sampled day -- just
    looped across every landed month instead of one.

    Returns one row per flagged site_id: `site_id, n_months_checked,
    months_flagged` (sorted "YYYY-MM" list), `worst_min_reconstructed_load`
    (the most negative single-month minimum, for a quick severity skim). A
    site never flagged does not appear.
    """
    flagged_months: dict[int, list[str]] = {}
    worst_min: dict[int, float] = {}
    n_months_checked = 0
    for year, month, frame in month_frames:
        n_months_checked += 1
        if frame is None or not len(frame):
            continue
        interval_table = Resolution.build_interval_table(
            frame, resolution,
            site_column=site_column, circuit_column=circuit_column,
            type_column=type_column, device_column=device_column,
            time_column=time_column,
        )
        if not len(interval_table):
            continue
        reconstructed = Signal.reconstruct_gross_load(
            interval_table, circuit_polarity,
            site_column=site_column, circuit_column=circuit_column,
            type_column=type_column, time_column=time_column,
        )
        if not len(reconstructed):
            continue
        local = reconstructed.copy()
        local[time_column] = Plots.to_aest(local[time_column])
        evaluated = Signal.evaluate_load_reconstruction(
            local, site_column=site_column, time_column=time_column,
            night_hour_start=night_hour_start, night_hour_end=night_hour_end,
            negative_threshold_w=negative_threshold_w,
        )
        newly_flagged = evaluated.loc[evaluated.likely_storage_or_sign_issue]
        for _, row in newly_flagged.iterrows():
            sid = int(row[site_column])
            flagged_months.setdefault(sid, []).append(f"{year}-{month:02d}")
            candidate_min = float(row["min_reconstructed_load"])
            worst_min[sid] = min(worst_min.get(sid, candidate_min), candidate_min)

    columns = [site_column, "n_months_checked", "months_flagged", "worst_min_reconstructed_load"]
    if not flagged_months:
        return pd.DataFrame(columns=columns)
    rows = [
        {site_column: sid, "n_months_checked": n_months_checked,
         "months_flagged": sorted(months), "worst_min_reconstructed_load": worst_min[sid]}
        for sid, months in flagged_months.items()
    ]
    return pd.DataFrame(rows).sort_values(site_column).reset_index(drop=True)


def apply_full_year_findings(
    resolution: pd.DataFrame,
    inactive_over_history: pd.DataFrame,
    reconstruction_over_history: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    inactive_reason: str = "inactive_full_year",
    exclusion_reason: str = "storage_or_sign_issue_full_year",
) -> pd.DataFrame:
    """
    Fold the two full-year revalidation findings into `resolution`. Pure.

    Circuit-level: a circuit named in `inactive_over_history` gets dropped
    individually (`inactive_reason` -- distinct from Phase 4's own
    `"inactive"`, so the audit trail shows WHICH pass caught it: the
    single-day sample or the full-year recheck).

    Site-level: a site named in `reconstruction_over_history` gets
    excluded entirely via `ami_resolution.exclude_flagged_sites`
    (`exclusion_reason` -- distinct from Section 7b's
    `"storage_or_sign_issue"` for the same reason).

    Order matters: circuit-level drops are applied FIRST, so a site that
    only loses a since-gone-inactive circuit (and was NOT independently
    flagged by reconstruction) keeps its other, still-good circuits --
    only sites the reconstruction check itself flags lose everything.
    """
    if resolution is None or not len(resolution):
        return resolution

    out = resolution.copy()

    if inactive_over_history is not None and len(inactive_over_history):
        newly_inactive_ids = set(inactive_over_history[circuit_column].astype(int))
        mask = out[circuit_column].isin(newly_inactive_ids) & out["kept"]
        out.loc[mask, "kept"] = False
        out.loc[mask, "drop_reason"] = inactive_reason

    if reconstruction_over_history is not None and len(reconstruction_over_history):
        flagged_site_ids = set(reconstruction_over_history[site_column].astype(int))
        out = Resolution.exclude_flagged_sites(
            out, flagged_site_ids, site_column=site_column, reason=exclusion_reason,
        )

    return out
