from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_MIN_SIGNATURE_KW",
    "DEFAULT_HIGH_MARGIN_RATIO",
    "DEFAULT_MEDIUM_MARGIN_RATIO",
    "DEFAULT_SUNRISE_HOUR",
    "DEFAULT_SUNSET_HOUR",
    "infer_pv_sign_convention",
    "score_circuits_by_solar_shape",
    "derive_circuit_pv_mapping",
]

#: A candidate's day/night delta must clear this floor before it counts as
#: evidence at all -- filters out ordinary measurement/quantization noise on
#: a circuit with no real day/night difference. Kilowatts, not a fraction of
#: capacity, since every circuit here is a small residential load/PV mix.
DEFAULT_MIN_SIGNATURE_KW = 0.05

#: Margin-ratio confidence bands, same shape as
#: `telemetry_profiles.py`'s `phase_mapping_high_margin_ratio` /
#: `phase_mapping_medium_margin_ratio` -- how much bigger the smallest
#: SELECTED candidate's score is than the largest EXCLUDED candidate's,
#: relative to the selected score itself.
DEFAULT_HIGH_MARGIN_RATIO = 0.5
DEFAULT_MEDIUM_MARGIN_RATIO = 0.15

#: Default daylight window for `score_circuits_by_solar_shape`'s reference
#: curve -- AEST, generous enough to cover shoulder-season sunrise/sunset;
#: the score is a correlation, not a hard cutoff, so a slightly-off window
#: still gives a real PV circuit a strong score.
DEFAULT_SUNRISE_HOUR = 6.0
DEFAULT_SUNSET_HOUR = 18.0


def infer_pv_sign_convention(
    circuit_profile: pd.DataFrame, *,
    min_signature_kw: float = DEFAULT_MIN_SIGNATURE_KW,
) -> int:
    """
    KNOWN UNRELIABLE -- kept for reference/diagnostics, not recommended for
    ranking. See `score_circuits_by_solar_shape` instead, which fixes the
    exact problem this function has.

    Attempts to empirically resolve the fleet-wide sign of a genuine PV
    circuit's daytime-vs-nighttime shape, by voting on sites where
    `n_pv_phases == n_available_circuits` (every available circuit forced
    onto the selected set -- "no ranking ambiguity" was the intended
    justification for treating these as unambiguous PV examples).

    That justification turned out to be wrong in practice: a
    forced-inclusion site's single circuit can ALSO be dominated by a large
    non-PV daytime load (an EV charger, a pool pump) with an even bigger
    day/night asymmetry than any PV behind it -- there is no ranking step
    at a forced-inclusion site to catch that, so a wrong-but-confident vote
    goes in. On this project's own fleet this flipped the fleet-wide sign
    away from the direction a real, visually-confirmed PV circuit
    (site 13683586, circuit 448410) actually shows, which then caused
    `derive_circuit_pv_mapping(..., pv_sign=...)` to select a non-PV
    circuit at that site with HIGH confidence -- worse than the original
    unsigned ambiguity it was meant to fix. The failure mode moved from the
    per-site ranking step up into this cross-site vote; it did not go away.

    `circuit_profile` needs the same columns `derive_circuit_pv_mapping`
    needs, plus the two raw medians `day_night_delta_kw` was built from:
    `daytime_median_p_kw`, `nighttime_median_p_kw`.
    """
    votes: list[float] = []
    for site_id, group in circuit_profile.groupby("site_id", sort=True):
        available = group.loc[group["power_measurement_available"].fillna(False)]
        target_raw = group["n_pv_phases"].iloc[0]
        target = (
            int(target_raw) if pd.notna(target_raw) and int(target_raw) > 0 else None
        )
        if target is None or len(available) != target:
            continue
        for _, row in available.iterrows():
            day = row.get("daytime_median_p_kw")
            night = row.get("nighttime_median_p_kw")
            if pd.isna(day) or pd.isna(night):
                continue
            delta = float(day) - float(night)
            if abs(delta) >= min_signature_kw:
                votes.append(delta)

    if not votes:
        return 1
    positive = sum(1 for v in votes if v > 0)
    negative = sum(1 for v in votes if v < 0)
    return 1 if positive >= negative else -1


def score_circuits_by_solar_shape(
    daily_profile: pd.DataFrame, *,
    power_column: str = "median_p_kw",
    local_hour_column: str = "local_hour",
    sunrise_hour: float = DEFAULT_SUNRISE_HOUR,
    sunset_hour: float = DEFAULT_SUNSET_HOUR,
) -> pd.DataFrame:
    """
    One row per (`site_id`, `circuit_id`) in `daily_profile` (the shape
    `ami_diagnostics.build_circuit_daily_profile_sql` returns, or anything
    with the same one-row-per-site/circuit/local_hour shape), scoring how
    closely that circuit's OWN by-local-hour median-power profile resembles
    a canonical solar-generation curve: zero outside
    [`sunrise_hour`, `sunset_hour`], one smooth bump peaking at solar noon
    in between.

    This replaces the two-point day/night delta this module used to rank
    on, to fix two related problems:
      - An UNSIGNED two-point delta can't tell a real midday PV bump apart
        from an oppositely-shaped night-heavy load with similar magnitude
        -- the two can land as a near-tied "ambiguous" pair purely by
        coincidence (this is what drove site 13683586's original "low"
        confidence result).
      - A single fleet-wide SIGN, voted on from other sites
        (`infer_pv_sign_convention`), does not fix that: it is
        contaminated by the identical confound one level up, since a
        "forced inclusion, therefore unambiguous" site can itself have a
        dominant non-PV daytime load. On this project's fleet that vote
        came out backwards and caused a NON-PV circuit to be selected with
        HIGH confidence at site 13683586 -- worse than doing nothing.

    Scoring each circuit's shape against a fixed, external, physically
    motivated reference curve uses no other circuit's or site's data at
    all, so neither failure mode above can happen here: a rectangular,
    non-solar load (flat while switched on, sharp on/off edges, or
    concentrated outside typical daylight hours -- an EV charger, a pool
    pump, overnight hot water) correlates poorly with a smooth midday bump
    in EITHER direction, while a genuine PV circuit correlates strongly in
    ONE of the two directions depending on which of `ac_load_net`'s two
    (unresolved, see module docstring) sign conventions applies. The
    returned `solar_shape_score` is that correlation's MAGNITUDE
    (`abs(corr)`, so both directions count as evidence); `solar_shape_sign`
    records which direction gave the higher correlation, per circuit --
    no fleet-wide assumption needed.

    Verified against site 13683586's real data: `|corr|` = 0.96 for
    circuit 448410 (the real, visually-confirmed PV circuit) vs. 0.30-0.39
    for its two load circuits -- not a near-tie the way the raw two-point
    delta was.

    Correlation needs >= 2 local-hour points with nonzero variance in both
    the observed profile and the reference curve; returns NaN (score and
    sign both) otherwise, e.g. for a circuit observed at only one hour, or
    a perfectly flat one. NaN sorts last in `derive_circuit_pv_mapping`'s
    ranking -- "no shape evidence", not "strong evidence either way".
    """
    def _reference(hour: float) -> float:
        if pd.isna(hour) or hour < sunrise_hour or hour > sunset_hour:
            return 0.0
        span = sunset_hour - sunrise_hour
        if span <= 0:
            return 0.0
        return float(np.sin(np.pi * (hour - sunrise_hour) / span))

    rows = []
    for (site_id, circuit_id), group in daily_profile.groupby(
        ["site_id", "circuit_id"], sort=True
    ):
        hours = pd.to_numeric(group[local_hour_column], errors="coerce")
        power = pd.to_numeric(group[power_column], errors="coerce")
        reference = hours.map(_reference)
        mask = hours.notna() & power.notna()

        score, sign = float("nan"), None
        if mask.sum() >= 2 and power[mask].std(ddof=0) > 0 and reference[mask].std(ddof=0) > 0:
            corr = power[mask].corr(reference[mask])
            if pd.notna(corr):
                score = abs(float(corr))
                sign = 1 if corr >= 0 else -1

        rows.append({
            "site_id": site_id,
            "circuit_id": circuit_id,
            "solar_shape_score": score,
            "solar_shape_sign": sign,
        })
    return pd.DataFrame(rows)


def derive_circuit_pv_mapping(
    circuit_profile: pd.DataFrame, *,
    min_signature_kw: float = DEFAULT_MIN_SIGNATURE_KW,
    high_margin_ratio: float = DEFAULT_HIGH_MARGIN_RATIO,
    medium_margin_ratio: float = DEFAULT_MEDIUM_MARGIN_RATIO,
    pv_sign: int | None = None,
    score_column: str | None = None,
) -> pd.DataFrame:
    """
    One row per `site_id` in `circuit_profile`, inferring which of that
    site's `circuit_id`s (in `ami_meter_residential`) most plausibly carry
    PV, using only what a real net-meter deployment could see -- never
    `ami_circuit_metadata.is_pv`/`circuit_type` or `ami_raw_phaseseparate`.

    `circuit_profile` must have one row per (site_id, circuit_id) with:
      - `n_pv_phases`: the site's target PV-circuit count (from
        `ami_site_metadata_residential` -- install-time metadata, not
        inferred from the signal, exactly like DNSP's
        `install_phase_count`).
      - `day_night_delta_kw`: `abs(daytime_median_P_kw - nighttime_median_P_kw)`
        for that circuit (see module docstring on why this is unsigned by
        default). ALWAYS required, regardless of `pv_sign`/`score_column`
        -- this is the kilowatt-denominated noise-floor check
        (`min_signature_kw`), kept separate from whatever column is used
        for ranking (see below), since only `day_night_delta_kw` is in
        physical units a "real signal at all" floor makes sense against.
      - `power_measurement_available`: whether this circuit has any
        non-null `P_kw` in scope at all.
      - `daytime_median_p_kw`, `nighttime_median_p_kw`: only required when
        `pv_sign` is given (see below).

    Pass AT MOST ONE of `pv_sign` / `score_column` (passing both raises
    `ValueError`) to change how candidates are RANKED against each other
    (the noise floor above always still applies):

      - `score_column` (RECOMMENDED): rank by this column's value instead
        of `day_night_delta_kw` -- e.g. `"solar_shape_score"` from
        `score_circuits_by_solar_shape`, which fixes this module's
        documented weak spot properly (see that function's docstring).
        Any numeric column works; higher is treated as more PV-like.

      - `pv_sign` (+1 or -1, from `infer_pv_sign_convention`): KNOWN
        UNRELIABLE, kept only for reference -- see that function's
        docstring for why voting on a single fleet-wide sign is
        contaminated by the same confound it's meant to fix, and prefer
        `score_column="solar_shape_score"` instead.

      - neither (default, `None`/`None`): rank by the unsigned
        `day_night_delta_kw` -- the original behaviour. Can't tell a real
        midday PV bump apart from an oppositely-shaped night-heavy load of
        similar magnitude, which is exactly the ambiguity `score_column`
        is meant to resolve.

    Returns columns: `site_id`, `n_target_pv_phases`, `n_available_circuits`,
    `inferred_pv_bearing_circuits` (`"|"`-joined circuit_id strings, empty
    when none selected), `phase_mapping_method`, `phase_mapping_confidence`
    (`high`/`medium`/`low`/`insufficient`/`not_applicable`),
    `phase_mapping_margin_ratio`, `top_day_night_delta_kw` (the winning
    candidate's RANKING score -- `day_night_delta_kw` in the default mode,
    or whatever `score_column`/`pv_sign` produced otherwise; the name is
    kept for continuity with the original unsigned mode).
    """
    if pv_sign is not None and score_column is not None:
        raise ValueError("pass at most one of `pv_sign` and `score_column`, not both")

    rows = []
    for site_id, group in circuit_profile.groupby("site_id", sort=True):
        group = group.sort_values("circuit_id")
        available = group.loc[group["power_measurement_available"].fillna(False)]
        target_raw = group["n_pv_phases"].iloc[0]
        target = (
            int(target_raw) if pd.notna(target_raw) and int(target_raw) > 0 else None
        )
        circuit_ids = sorted(available["circuit_id"].astype(str).tolist())

        selected: list[str] = []
        method = "not_applicable"
        confidence = "not_applicable"
        margin_ratio = None
        top_score = None

        if target is None:
            method = "no_pv_expected"
            confidence = "not_applicable"
        elif len(circuit_ids) < target:
            method = "insufficient_circuits"
            confidence = "insufficient"
        else:
            if score_column is not None:
                score = pd.to_numeric(available[score_column], errors="coerce")
            elif pv_sign is not None:
                signed_delta = pd.to_numeric(
                    available["daytime_median_p_kw"], errors="coerce"
                ) - pd.to_numeric(available["nighttime_median_p_kw"], errors="coerce")
                score = pv_sign * signed_delta
            else:
                score = pd.to_numeric(available["day_night_delta_kw"], errors="coerce")

            ranked = available.assign(_score=score).sort_values(
                ["_score", "circuit_id"], ascending=[False, True]
            )
            selected = sorted(ranked.head(target)["circuit_id"].astype(str).tolist())

            # The kW noise floor ALWAYS reads the unsigned day/night delta,
            # regardless of what `_score` was ranked on -- a correlation or
            # a signed delta isn't in kilowatts, so "does this candidate
            # have any real signal at all" has to be checked separately
            # from "is this candidate the most PV-shaped of the bunch".
            selected_floor_signal = pd.to_numeric(
                ranked.iloc[target - 1]["day_night_delta_kw"], errors="coerce"
            )
            selected_floor_score = ranked.iloc[target - 1]["_score"]
            next_score = ranked.iloc[target]["_score"] if len(ranked) > target else None
            top_score = ranked.iloc[0]["_score"] if pd.notna(ranked.iloc[0]["_score"]) else None
            method = (
                "all_available_circuits_selected"
                if len(circuit_ids) == target
                else "ranked_day_night_delta"
            )

            if pd.notna(selected_floor_signal) and float(selected_floor_signal) < min_signature_kw:
                confidence = "low"
            elif len(circuit_ids) == target:
                # Counts already match -- no ranking ambiguity, only whether
                # the evidence clears the noise floor (checked above).
                confidence = "high"
            elif next_score is not None and pd.notna(next_score):
                denom = max(abs(float(selected_floor_score)), 1e-6)
                margin_ratio = (float(selected_floor_score) - float(next_score)) / denom
                if margin_ratio >= high_margin_ratio:
                    confidence = "high"
                elif margin_ratio >= medium_margin_ratio:
                    confidence = "medium"
                else:
                    confidence = "low"
            else:
                confidence = "medium"

        rows.append({
            "site_id": site_id,
            "n_target_pv_phases": target,
            "n_available_circuits": len(circuit_ids),
            "inferred_pv_bearing_circuits": "|".join(selected),
            "phase_mapping_method": method,
            "phase_mapping_confidence": confidence,
            "phase_mapping_margin_ratio": margin_ratio,
            "top_day_night_delta_kw": top_score,
        })
    return pd.DataFrame(rows)
