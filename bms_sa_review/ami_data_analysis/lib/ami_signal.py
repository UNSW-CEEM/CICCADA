"""
Phase 3b: the circuit -> signal map, and the sign convention it depends on.
============================================================================

`build_structured_data.py` applies `circuit_polarity` and sums, on the
documented assumption that it "makes PV generation positive" -- true for the
PV half, because that is the only half the existing pipeline ever consumed.
This module checks whether the same fixed-sign assumption holds for load
circuits too, and separately flags circuits that might not have a fixed sign
at all (battery / EV), before either is trusted for `gross_load`.

`build_signal_map` is where a census turns into
`ami_config.CIRCUIT_SIGNAL_MAP` -- proven aggregates and storage circuits are
routed out of the sum, everything else becomes its own named signal.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = [
    "STORAGE_KEYWORDS", "classify_storage_circuits",
    "verify_polarity_makes_positive", "build_signal_map",
    "night_window_stats", "classify_pv_night_behaviour", "compare_pv_night_to_load",
    "sites_with_storage_circuits", "reconstruct_gross_load", "evaluate_load_reconstruction",
]

#: Name fragments that suggest a circuit might be bidirectional (charges AND
#: discharges), and therefore cannot be sign-corrected with one fixed
#: `circuit_polarity`. A HYPOTHESIS GENERATOR ONLY --
#: `verify_polarity_makes_positive`'s `bidirectional` column is the actual
#: evidence; a name here does not by itself resolve `ami_config.STORAGE_HANDLING`.
STORAGE_KEYWORDS = ("batt", "ev_", "_ev", "charger", "stor")


def classify_storage_circuits(circuit_types, *, keywords=STORAGE_KEYWORDS) -> dict:
    """{circuit_type: bool} -- does the name suggest battery/EV? Pure, name-only."""
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    return {str(t): bool(pattern.search(str(t))) for t in circuit_types}


def verify_polarity_makes_positive(
    sample: pd.DataFrame, *,
    type_column: str = "circuit_type",
    power_column: str = "power",
    polarity_column: str = "circuit_polarity",
    bidirectional_share_threshold: float = 0.05,
    noise_floor_w: float = 50.0,
) -> pd.DataFrame:
    """
    Does `power * circuit_polarity` come out consistently one-signed per
    circuit_type, as the Stage 1 pipeline assumes for PV? Pure.

    Takes real `ts` rows already joined to `meta_up23c` for `circuit_polarity`
    and `circuit_type`. Groups by circuit_type and reports what share of
    intervals are negative vs positive after correction. A circuit_type that
    is (near-)always one sign confirms the fixed-sign assumption holds for it.
    One with a real share on BOTH sides of zero (`bidirectional=True`) is a
    genuine candidate for `ami_config.STORAGE_HANDLING`: it needs a
    charge/discharge split, not a sign flip, because no single fixed
    `circuit_polarity` value can be "correct" for both directions.

    `share_negative`/`share_positive`/`bidirectional` only count a reading
    once its magnitude clears `noise_floor_w` -- a real-fleet run without this
    floor flagged 17 of 24 load types (lighting, hot water, a fridge...) as
    "bidirectional", which is not a plausible fleet of reversible appliances.
    A circuit idling near zero (CT/power-factor measurement noise, or simply
    "off") produces small negative readings by chance; at fleet scale (tens
    of thousands of intervals per type) even a couple of percent of such
    noise clears a bare sign-count threshold easily, without the circuit ever
    genuinely reversing power flow. `share_negative_raw`/`share_positive_raw`
    keep the old sign-only computation alongside the new one, so the
    difference this floor makes stays visible rather than silently changing
    history. This mirrors `ami_signal.classify_pv_night_behaviour`'s
    `net_like_threshold_w`, which exists for the identical reason.
    """
    if sample is None or not len(sample):
        return pd.DataFrame()
    frame = sample.copy()
    frame["power_signed"] = frame[power_column] * frame[polarity_column]

    def _share_negative_raw(s):
        return float((s < 0).mean())

    def _share_positive_raw(s):
        return float((s > 0).mean())

    def _share_negative(s):
        return float((s < -noise_floor_w).mean())

    def _share_positive(s):
        return float((s > noise_floor_w).mean())

    grouped = frame.groupby(type_column)["power_signed"]
    out = pd.DataFrame({
        "n": grouped.count(),
        "share_negative_raw": grouped.apply(_share_negative_raw),
        "share_positive_raw": grouped.apply(_share_positive_raw),
        "share_negative": grouped.apply(_share_negative),
        "share_positive": grouped.apply(_share_positive),
    }).reset_index()
    out["bidirectional"] = (
        (out["share_negative"] > bidirectional_share_threshold)
        & (out["share_positive"] > bidirectional_share_threshold)
    )
    return out.sort_values(type_column).reset_index(drop=True)


def build_signal_map(
    census: pd.DataFrame, *,
    aggregate_types,
    storage_types=frozenset(),
    type_column: str = "circuit_type",
    is_pv_column: str = "is_pv",
) -> dict:
    """
    Assign every circuit_type a signal role. Pure.

    Rule, applied in this order, per circuit_type:
      1. In `aggregate_types` (proven by `ami_taxonomy.check_aggregation`,
         never by name alone) -> excluded entirely. Summing a proven aggregate
         in would double-count every one of its own components.
      2. In `storage_types` (proven bidirectional by
         `verify_polarity_makes_positive`) -> excluded from `gross_load`.
         A battery is a second controllable resource, not a load; folding its
         discharge into `gross_load` would make the ground truth wrong in a
         way disaggregation cannot recover.
      3. `is_pv=True` -> `"pv_generation"`.
      4. Everything else -> its own name (`load_pool` stays `load_pool`,
         rather than every sub-load being flattened into one undifferentiated
         bucket here). `ami_build` (Phase 5) sums the named sub-loads into
         `gross_load`; keeping them named up to that point means a bug in one
         sub-load's handling doesn't have to be found by re-deriving the whole
         sum from scratch.
    """
    aggregate_types = set(aggregate_types)
    storage_types = set(storage_types)
    mapping: dict[str, str] = {}
    for _, row in census.drop_duplicates(subset=[type_column]).iterrows():
        t = str(row[type_column])
        if t in aggregate_types:
            mapping[t] = "EXCLUDE (proven aggregate -- see AggregationCheck)"
        elif t in storage_types:
            mapping[t] = "EXCLUDE from gross_load (storage -- see STORAGE_HANDLING)"
        elif bool(row[is_pv_column]):
            mapping[t] = "pv_generation"
        else:
            mapping[t] = t
    return mapping


def night_window_stats(
    sample: pd.DataFrame, *,
    time_column: str = "t_stamp",
    power_column: str = "power_signed",
    night_hour_start: int = 1,
    night_hour_end: int = 4,
) -> dict:
    """
    Mean/median of `power_column` inside a local-time night window. Pure,
    PV-agnostic -- this is just "what did this circuit read between these two
    hours", usable for a PV circuit or a co-located load circuit alike.

    `time_column` must already be local time (e.g. AEST) -- this function
    reads hour-of-day directly and does no timezone conversion itself; the
    caller converts once, via `ami_plots.to_aest`, for every signal that
    needs local hours, rather than each function doing its own conversion
    and risking them drifting out of sync.
    """
    if sample is None or not len(sample) or time_column not in sample.columns:
        return {
            "n_night_intervals": 0, "night_mean": None, "night_median": None,
            "reason": "no sample, or missing time column",
        }
    hours = sample[time_column].dt.hour
    night = sample[(hours >= night_hour_start) & (hours < night_hour_end)]
    if not len(night):
        return {
            "n_night_intervals": 0, "night_mean": None, "night_median": None,
            "reason": f"no intervals in the {night_hour_start}:00-{night_hour_end}:00 window",
        }
    return {
        "n_night_intervals": int(len(night)),
        "night_mean": float(night[power_column].mean()),
        "night_median": float(night[power_column].median()),
        "reason": None,
    }


def classify_pv_night_behaviour(
    sample: pd.DataFrame, *,
    time_column: str = "t_stamp",
    power_column: str = "power_signed",
    night_hour_start: int = 1,
    night_hour_end: int = 4,
    net_like_threshold_w: float = 100.0,
) -> dict:
    """
    Is a PV-side circuit's reading pure generation, or net-of-load? Pure.

    True PV generation is ~0 W in the dead of night, wherever the site is --
    there is no sunlight to produce it, at most a small negative draw from
    the inverter's own standby power (a few W to a few tens of W). A NET
    (generation - load) signal instead reads a substantial negative value at
    night, because it is showing real household consumption with nothing to
    offset it. `net_like_threshold_w` is the line between "small enough to be
    standby draw" and "too big to be anything but load" -- a heuristic
    threshold, not a law of physics, so treat a verdict near it
    (`abs(night_mean)` within a factor of ~2 of the threshold) as worth a
    second look rather than final.

    This tests ONE circuit in isolation -- `compare_pv_night_to_load` is the
    companion check that asks whether a co-located load circuit's own
    night-time reading corroborates a net-like verdict here.
    """
    stats = night_window_stats(
        sample, time_column=time_column, power_column=power_column,
        night_hour_start=night_hour_start, night_hour_end=night_hour_end,
    )
    if stats["night_mean"] is None:
        return {**stats, "verdict": None}

    night_mean = stats["night_mean"]
    if abs(night_mean) <= net_like_threshold_w:
        verdict = "generation-like"
        reason = (
            f"night mean {night_mean:.1f} W is within +/-{net_like_threshold_w:.0f} W of "
            "zero -- consistent with near-zero PV output overnight (at most inverter "
            "standby draw)"
        )
    else:
        verdict = "net-like"
        reason = (
            f"night mean {night_mean:.1f} W is more than {net_like_threshold_w:.0f} W from "
            "zero, with no sunlight available to produce it -- consistent with this signal "
            "including load consumption, not pure generation"
        )
    return {**stats, "verdict": verdict, "reason": reason}


def compare_pv_night_to_load(
    pv_stats: dict, load_stats: dict, *,
    corroboration_ratio_band: tuple = (0.3, 2.0),
) -> dict:
    """
    Does a co-located load circuit's night-time reading corroborate a
    `classify_pv_night_behaviour` "net-like" verdict? Pure -- consumes the
    dicts those two functions return, runs no query of its own.

    This is corroborating evidence only, not a second independent proof: when
    a site's "load" and "PV" circuits are really two different projections of
    the same underlying net meter, a match here doesn't distinguish "this IS
    net-of-load" from "these two signals happen to be similarly sized." What
    it DOES rule out is the opposite: a night-time PV deficit far bigger or
    smaller than the co-located load's own night consumption means that
    particular load circuit does not explain the deficit, whatever else is
    going on.
    """
    if pv_stats.get("verdict") != "net-like":
        return {
            "corroborated": None,
            "reason": "PV verdict is not net-like -- no net-of-load hypothesis to corroborate",
        }
    load_night_mean = load_stats.get("night_mean")
    if load_night_mean is None:
        return {"corroborated": None, "reason": "no co-located load sample available for comparison"}

    pv_deficit = abs(pv_stats["night_mean"])
    load_night = abs(load_night_mean)
    if load_night == 0:
        return {
            "corroborated": False,
            "reason": "co-located load's night mean is ~0 -- cannot corroborate a net-of-load deficit",
        }

    ratio = pv_deficit / load_night
    lo, hi = corroboration_ratio_band
    corroborated = lo <= ratio <= hi
    return {
        "corroborated": corroborated,
        "ratio": ratio,
        "reason": (
            f"PV night deficit / co-located load night mean = {ratio:.2f} "
            f"({'within' if corroborated else 'outside'} the {lo}-{hi} corroboration band)"
        ),
    }


def sites_with_storage_circuits(
    meta: pd.DataFrame, *,
    site_column: str = "site_id",
    type_column: str = "circuit_type",
    keywords=STORAGE_KEYWORDS,
) -> list:
    """
    Which site_ids have ANY circuit_type whose NAME suggests battery/EV
    (`classify_storage_circuits`)? Pure, name-only -- a hypothesis generator
    for which sites need a closer look before their `ac_load_net` is trusted
    as pure house consumption, not a proof that storage is actually present
    or actually affecting `ac_load_net`'s reading.

    This is only the EXPLICIT half of storage detection: a battery wired
    behind the SAME CT as `ac_load_net` (rather than separately metered
    under its own circuit_id) would be invisible here -- see
    `evaluate_load_reconstruction`'s night-time check for the other half.
    """
    if meta is None or not len(meta) or type_column not in meta.columns:
        return []
    flags = classify_storage_circuits(meta[type_column].unique())
    storage_types = {t for t, flagged in flags.items() if flagged}
    if not storage_types:
        return []
    return sorted(meta.loc[meta[type_column].isin(storage_types), site_column].unique().tolist())


def reconstruct_gross_load(
    interval_table: pd.DataFrame,
    circuit_polarity: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    type_column: str = "circuit_type",
    time_column: str = "t_stamp",
    power_column: str = "power",
    polarity_column: str = "circuit_polarity",
    load_type: str = "ac_load_net",
    pv_type: str = "pv_site_net",
) -> pd.DataFrame:
    """
    Candidate reconstruction of true (PV-independent) house load from
    Phase 4's interval-level table: `load_signed` (every `load_type` circuit
    at a site, sign-corrected and SUMMED across phases -- the on-demand
    grouping the settled Phase 4 decisions deliberately defer to a consumer,
    not performed in `ami_resolution.build_interval_table` itself) plus
    `pv_signed` (the analogous PV-side sum), one row per (site_id, t_stamp).

    This function only COMPUTES the candidate reconstruction -- it does not
    judge whether the result is trustworthy. `evaluate_load_reconstruction`
    is the companion that does that. Keeping them separate means a bad
    reconstruction formula and a bad plausibility threshold are two
    independently testable claims, not one conflated one.

    `circuit_polarity` is a separate small frame (`circuit_id`,
    `circuit_polarity`) rather than assumed already present on
    `interval_table` -- Phase 4's own output deliberately does not carry
    polarity or apply the sign convention (see `ami_resolution`'s module
    docstring and `ami_config.SIGN_CONVENTION_RESOLVED`), so this function is
    where that correction is actually applied, once, for this diagnostic
    specifically.
    """
    columns = [site_column, time_column, "load_signed", "pv_signed", "reconstructed_load"]
    if interval_table is None or not len(interval_table):
        return pd.DataFrame(columns=columns)

    frame = interval_table.merge(circuit_polarity, on=circuit_column, how="left")
    frame["power_signed"] = frame[power_column] * frame[polarity_column]

    def _side_sum(side_type, out_name):
        side = frame[frame[type_column] == side_type]
        if not len(side):
            return pd.DataFrame(columns=[site_column, time_column, out_name])
        summed = (
            side.groupby([site_column, time_column])["power_signed"]
            .sum()
            .reset_index()
            .rename(columns={"power_signed": out_name})
        )
        return summed

    load_side = _side_sum(load_type, "load_signed")
    pv_side = _side_sum(pv_type, "pv_signed")

    merged = load_side.merge(pv_side, on=[site_column, time_column], how="outer")
    merged["reconstructed_load"] = merged["load_signed"] + merged["pv_signed"]
    return merged[columns].sort_values([site_column, time_column]).reset_index(drop=True)


def evaluate_load_reconstruction(
    reconstructed: pd.DataFrame, *,
    site_column: str = "site_id",
    time_column: str = "t_stamp",
    reconstructed_column: str = "reconstructed_load",
    night_hour_start: int = 1,
    night_hour_end: int = 4,
    negative_threshold_w: float = -100.0,
) -> pd.DataFrame:
    """
    Per-site plausibility summary for `reconstruct_gross_load`'s output:
    true house load should not be meaningfully negative, so a share of
    intervals below `negative_threshold_w` is the sanity signal -- split
    into ALL intervals vs NIGHT-ONLY intervals specifically.

    The night split matters: PV is ~0 overnight (Section 9's own finding),
    so a negative reconstruction there cannot be explained by any
    near-simultaneous PV over-generation double-counted into the sum --
    a real night-time violation means either the sign convention is wrong
    for that site, or (per the module's storage-handling note) a battery/EV
    circuit is netted into `ac_load_net` invisibly, behind the same CT,
    with no separate circuit_id to detect it by name
    (`sites_with_storage_circuits` only catches the EXPLICIT case).

    `time_column` must already be LOCAL time (e.g. via `ami_plots.to_aest`)
    -- this function does no timezone conversion itself, mirroring
    `night_window_stats`'s own contract.
    """
    columns = [
        site_column, "n_intervals", "share_negative_all",
        "n_night_intervals", "share_negative_night", "min_reconstructed_load",
        "likely_storage_or_sign_issue",
    ]
    if reconstructed is None or not len(reconstructed):
        return pd.DataFrame(columns=columns)

    frame = reconstructed.dropna(subset=[reconstructed_column]).copy()
    if not len(frame):
        return pd.DataFrame(columns=columns)

    hours = frame[time_column].dt.hour
    frame["_is_night"] = (hours >= night_hour_start) & (hours < night_hour_end)
    frame["_is_negative"] = frame[reconstructed_column] < negative_threshold_w

    rows = []
    for site_id, group in frame.groupby(site_column):
        night = group[group["_is_night"]]
        rows.append({
            site_column: site_id,
            "n_intervals": int(len(group)),
            "share_negative_all": float(group["_is_negative"].mean()),
            "n_night_intervals": int(len(night)),
            "share_negative_night": float(night["_is_negative"].mean()) if len(night) else np.nan,
            "min_reconstructed_load": float(group[reconstructed_column].min()),
        })
    out = pd.DataFrame(rows)
    out["likely_storage_or_sign_issue"] = (
        out["share_negative_night"].fillna(0.0) > 0.0
    )
    return out[columns].sort_values(site_column).reset_index(drop=True)
