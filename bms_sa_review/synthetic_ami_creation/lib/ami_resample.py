"""
Phase 3c: which source column is power, which is energy -- verified, not
assumed -- and the resample rule that follows from the answer.
============================================================================

`ami_config.SOURCE_COLUMN_UNITS` states what the Stage 1 code IMPLIES:
`power` is instantaneous W, `energy_reactive` is already a 5-minute kvarh
(Stage 1 multiplies it by 12 to render an average kvar). `verify_column_units`
checks that against the data itself rather than trusting the docstring it was
read off: a circuit's apparent power cannot sustainably exceed its own rated
capacity by much, so the unit hypothesis that produces materially fewer such
violations is the one that actually fits.

`resample_to_interval` makes `ami_config.RESAMPLE_RULE` executable: an
instantaneous column is converted to per-source-interval energy and then
summed; an already-energy column is summed directly. Getting this backwards
for the reactive column is wrong by a factor of `12 x (source interval count
per target interval)` -- exactly the mistake the config file's "THIS IS THE
ONE THAT WILL BITE" comment warns about.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "interval_energy_kwh", "verify_column_units", "confirm_energy_matches_power",
    "confirm_energy_matches_power_actual_interval", "resample_to_interval",
    "energy_granularity_and_implied_interval",
]


def interval_energy_kwh(power_w: pd.Series, source_interval_minutes: float) -> pd.Series:
    """kWh delivered in one source interval, from instantaneous power in W. Pure."""
    return power_w / 1000.0 * (source_interval_minutes / 60.0)


def confirm_energy_matches_power(
    sample: pd.DataFrame, *,
    power_column: str = "power",
    energy_column: str = "energy",
    interval_minutes: float = 5.0,
    tolerance_wh: float = 0.5,
    max_mismatch_share: float = 0.01,
    group_column: str = None,
) -> dict:
    """
    Does a native `energy` column already equal `power * interval_hours`, as
    its name promises? Pure.

    `verify_column_units` exists for a genuinely ambiguous column -- but it
    only had to guess at `energy_reactive`'s units because the queries that
    fed it never pulled `ts`'s own `energy` column (an oversight inherited
    from Stage 1's narrower column list, not a fact about the source data).
    Once `energy` -- and its siblings `energy_import`, `energy_export`,
    `energy_reactive_import`, `energy_reactive_export` -- are actually
    queried, there is nothing left to guess: this just confirms the
    relationship the column's own name promises, the same spot-check any
    other named column would get before being trusted at face value.
    `tolerance_wh` absorbs ordinary floating-point/rounding noise, not a
    real discrepancy; `max_mismatch_share` is how much of the sample is
    allowed to miss that tolerance before `confirmed` turns False.

    `group_column`, if given (e.g. `circuit_type`), adds
    `share_mismatched_by_group` -- worth checking if `confirmed` comes back
    False: a mismatch concentrated in one or two circuit_types (rather than
    spread evenly) points to something specific to that type, not a
    fleet-wide units problem.
    """
    if sample is None or not len(sample):
        return {
            "n_rows": 0, "n_mismatched": 0, "share_mismatched": None,
            "confirmed": None, "reason": "no sample available",
        }
    expected_wh = sample[power_column] * (interval_minutes / 60.0)
    diff = (sample[energy_column] - expected_wh).abs()
    mismatched = diff > tolerance_wh
    n_mismatched = int(mismatched.sum())
    share = n_mismatched / len(sample)
    confirmed = share <= max_mismatch_share
    result = {
        "n_rows": int(len(sample)),
        "n_mismatched": n_mismatched,
        "share_mismatched": float(share),
        "confirmed": confirmed,
        "reason": (
            f"{n_mismatched:,} of {len(sample):,} row(s) ({share:.2%}) differ from "
            f"`{power_column}` * interval_hours by more than {tolerance_wh} Wh -- "
            + ("within tolerance, `energy` is directly usable" if confirmed else
               "exceeds the allowed mismatch share, do not trust `energy` blindly here")
        ),
    }
    if group_column is not None and group_column in sample.columns:
        by_group = mismatched.groupby(sample[group_column]).mean().sort_values(ascending=False)
        result["share_mismatched_by_group"] = {str(k): float(v) for k, v in by_group.items()}
    return result


def confirm_energy_matches_power_actual_interval(
    sample: pd.DataFrame, *,
    circuit_column: str = "circuit_id",
    time_column: str = "t_stamp",
    power_column: str = "power",
    energy_column: str = "energy",
    tolerance_wh: float = 0.5,
    max_mismatch_share: float = 0.01,
    group_column: str = None,
) -> dict:
    """
    Same question as `confirm_energy_matches_power`, but computes each row's
    interval length from the ACTUAL gap to that circuit's own previous
    reading, instead of assuming every row is exactly
    `ami_config.SOURCE_INTERVAL_MINUTES` apart. Pure.

    Real AMI feeds miss polls: a circuit's next `ts` row can arrive 10 or 15
    minutes after the last one, not the nominal 5, whenever a report was
    dropped -- the same real-world fleet churn `sites_missing_day_data`
    already accounts for at the whole-day level. If `energy` is the meter's
    own reading for the time actually elapsed (which a genuinely-measured
    energy column should be), comparing it against
    `power * NOMINAL_interval_hours` mismatches on exactly those rows even
    though `energy` is correct for its real interval -- a false alarm from
    `confirm_energy_matches_power`, not a real units problem. This
    recomputes the expected value from each row's own measured gap to test
    that hypothesis directly instead of assuming it: if the mismatch share
    drops sharply here relative to the nominal-interval version, missed
    polls (not a units error) is the explanation.

    The first reading of every circuit has no earlier timestamp to diff
    against and is excluded from this check (not from the sample) -- there
    is no actual interval to compute for it.
    """
    if sample is None or not len(sample):
        return {
            "n_rows": 0, "n_mismatched": 0, "share_mismatched": None,
            "confirmed": None, "reason": "no sample available",
        }
    frame = sample.sort_values([circuit_column, time_column]).copy()
    if not pd.api.types.is_datetime64_any_dtype(frame[time_column]):
        frame[time_column] = pd.to_datetime(frame[time_column])
    frame["_actual_interval_minutes"] = (
        frame.groupby(circuit_column)[time_column].diff().dt.total_seconds() / 60.0
    )
    frame = frame.dropna(subset=["_actual_interval_minutes"])
    if not len(frame):
        return {
            "n_rows": 0, "n_mismatched": 0, "share_mismatched": None,
            "confirmed": None,
            "reason": "no row has an earlier reading on its own circuit to diff against",
        }

    expected_wh = frame[power_column] * (frame["_actual_interval_minutes"] / 60.0)
    diff = (frame[energy_column] - expected_wh).abs()
    mismatched = diff > tolerance_wh
    n_mismatched = int(mismatched.sum())
    share = n_mismatched / len(frame)
    confirmed = share <= max_mismatch_share
    result = {
        "n_rows": int(len(frame)),
        "n_mismatched": n_mismatched,
        "share_mismatched": float(share),
        "confirmed": confirmed,
        "reason": (
            f"{n_mismatched:,} of {len(frame):,} row(s) ({share:.2%}) differ from "
            f"`{power_column}` * ACTUAL-interval-hours (from consecutive `{time_column}` "
            f"gaps on the same circuit, not the nominal interval) by more than "
            f"{tolerance_wh} Wh"
        ),
    }
    if group_column is not None and group_column in frame.columns:
        by_group = mismatched.groupby(frame[group_column]).mean().sort_values(ascending=False)
        result["share_mismatched_by_group"] = {str(k): float(v) for k, v in by_group.items()}
    return result


def energy_granularity_and_implied_interval(
    sample: pd.DataFrame, *,
    circuit_column: str = "circuit_id",
    power_column: str = "power",
    energy_column: str = "energy",
    min_abs_power_for_ratio: float = 200.0,
) -> pd.DataFrame:
    """
    Per circuit_id: is `energy` logged as a whole-Wh integer, and what
    interval does `energy / power` actually imply? Pure.

    `confirm_energy_matches_power`/`..._actual_interval` both assume every
    circuit's `energy` is consistent with ONE interval (nominal, or actual
    from `t_stamp` gaps). A device/meter-model effect can slip past both: a
    circuit's `t_stamp` gaps can land exactly on the nominal grid (its
    LOGGING cadence isn't drifting) while its internal energy-accumulation
    window is slightly shorter or longer than that -- invisible to a
    gap-based check, but visible here as `implied_interval_minutes` (the
    median of `energy / power * 60`, restricted to rows with
    `abs(power) >= min_abs_power_for_ratio` so small-power quantization
    noise doesn't swamp the ratio) differing from the nominal interval.

    `share_integer_energy` close to 1.0 for one circuit while its siblings
    sit near 0.0 is itself informative -- a different logging convention
    (e.g. a whole-watt-hour energy register vs a continuously computed
    value) for that specific circuit, independent of its `circuit_type`.
    Cross-referencing flagged circuit_ids against `device_type` (a
    `meta_up23c` column, not present in `sample` -- join it in the caller)
    is the natural next step: two documented AMI hardware brands
    (`CATCH Power`, `Watt Watcher`) could plausibly differ in exactly this
    way.
    """
    columns = [
        "circuit_id", "n_rows", "share_integer_energy",
        "implied_interval_minutes", "n_rows_used_for_ratio",
    ]
    if sample is None or not len(sample):
        return pd.DataFrame(columns=columns)

    frame = sample.copy()
    is_integer = (frame[energy_column].astype(float) % 1.0) == 0.0
    granularity = is_integer.groupby(frame[circuit_column]).agg(["mean", "count"])
    granularity.columns = ["share_integer_energy", "n_rows"]

    ratio_rows = frame[frame[power_column].abs() >= min_abs_power_for_ratio]
    implied = (
        (ratio_rows[energy_column] / ratio_rows[power_column] * 60.0)
        .groupby(ratio_rows[circuit_column])
        .agg(["median", "count"])
    )
    implied.columns = ["implied_interval_minutes", "n_rows_used_for_ratio"]

    result = granularity.join(implied, how="left").reset_index()
    result = result.rename(columns={circuit_column: "circuit_id"})
    result["n_rows_used_for_ratio"] = (
        result["n_rows_used_for_ratio"].fillna(0).astype(int)
    )
    return result[columns].sort_values("circuit_id").reset_index(drop=True)


def verify_column_units(
    sample: pd.DataFrame, *,
    power_column: str = "power",
    reactive_column: str = "energy_reactive",
    plausible_reactive_ratio: tuple = (0.02, 5.0),
) -> dict:
    """
    Which unit hypothesis for `energy_reactive` actually fits the data? Pure.

    Two hypotheses, both consistent with a column literally named
    "energy_reactive":

      A (documented -- Stage 1's assumption): 5-minute kvarh; convert to an
         average kvar with `* 12`.
      B (the naive alternative): already an instantaneous kvar reading; no `* 12`.

    "Does apparent power exceed rated capacity" cannot distinguish these: `* 12`
    only ever scales `|Q|` UP, so hypothesis A can only accumulate the same or
    MORE such violations than B, never fewer -- that comparison is one-sided
    and can never make A look preferable. Instead this compares the implied
    reactive-to-active ratio `|Q| / |P|` against a generously wide but still
    physically bounded band (default 0.02..5.0, roughly power factor 0.9998
    down to 0.196): whichever hypothesis pushes most rows' ratio outside that
    band is scaling the reactive column by roughly the wrong order of
    magnitude, in either direction -- a genuinely symmetric test. Returns both
    implausible-share values and a plain-language verdict; a thin margin is
    surfaced (`margin_is_thin`) rather than silently decided.
    """
    if sample is None or not len(sample):
        return {"verdict": None, "margin_is_thin": None, "reason": "no sample provided"}

    frame = sample.copy()
    valid = frame[power_column] != 0
    if not valid.any():
        return {"verdict": None, "margin_is_thin": None,
                "reason": f"no rows with nonzero {power_column} to form a ratio from"}

    # The /1000 kW/kvar conversion would apply identically to numerator and
    # denominator under either hypothesis, so it cancels out of the ratio --
    # left out here, not forgotten.
    p_abs = frame.loc[valid, power_column].abs()
    ratio_a = (frame.loc[valid, reactive_column] * 12).abs() / p_abs
    ratio_b = frame.loc[valid, reactive_column].abs() / p_abs

    lo, hi = plausible_reactive_ratio
    share_a = float(((ratio_a < lo) | (ratio_a > hi)).mean())
    share_b = float(((ratio_b < lo) | (ratio_b > hi)).mean())

    if share_a < share_b:
        verdict = "A (5-minute kvarh -- x12 needed)"
    elif share_b < share_a:
        verdict = "B (already instantaneous kvar -- no x12)"
    else:
        verdict = None

    return {
        "share_implausible_hypothesis_A": share_a,
        "share_implausible_hypothesis_B": share_b,
        "verdict": verdict,
        "margin_is_thin": bool(verdict is not None and abs(share_a - share_b) < 0.01),
        "reason": (
            f"hypothesis A: {share_a:.1%} of rows have an implausible reactive/active "
            f"ratio (outside {lo}..{hi}); hypothesis B: {share_b:.1%} -- "
            + ("A fits better" if verdict and verdict.startswith("A") else
               "B fits better" if verdict else "tie -- inconclusive, do not decide from this alone")
        ),
    }


def resample_to_interval(
    frame: pd.DataFrame, *,
    time_column: str,
    group_columns: list,
    energy_like_columns: dict,
    source_interval_minutes: float,
    target_interval_minutes: float,
) -> pd.DataFrame:
    """
    Resample a tidy source-interval frame to the target AMI interval. Pure.

    `energy_like_columns` maps column name -> `True` if the column is already
    an energy value per source interval (summed directly, e.g. a raw energy
    field) or `False` if it is an instantaneous rate in kW/W (converted to
    kWh per source interval via `interval_energy_kwh`, THEN summed). This is
    `ami_config.RESAMPLE_RULE` made executable, so the power/energy asymmetry
    cannot be gotten wrong by a bucket forgetting which column is which.

    `target_interval_minutes` must be a whole multiple of
    `source_interval_minutes`; anything else would need interpolation this
    function deliberately does not attempt.
    """
    if target_interval_minutes % source_interval_minutes != 0:
        raise ValueError(
            f"target_interval_minutes ({target_interval_minutes}) must be a whole "
            f"multiple of source_interval_minutes ({source_interval_minutes})"
        )

    out = frame.copy()
    # Coerce rather than assume: `time_column` can arrive without a proper
    # datetime64 dtype (e.g. a caller concatenating several Athena pulls,
    # one of which came back with zero rows for a real data gap -- see
    # `ami_plots.to_aest`'s docstring for the same underlying cause), and
    # `.dt.floor` raises "Can only use .dt accessor with datetimelike
    # values" on that rather than degrading gracefully.
    if not pd.api.types.is_datetime64_any_dtype(out[time_column]):
        out[time_column] = pd.to_datetime(out[time_column])

    for column, is_energy_already in energy_like_columns.items():
        if not is_energy_already:
            out[column] = interval_energy_kwh(out[column], source_interval_minutes)

    out["_bucket"] = out[time_column].dt.floor(f"{int(target_interval_minutes)}min")
    agg = {column: "sum" for column in energy_like_columns}
    result = out.groupby(group_columns + ["_bucket"], as_index=False).agg(agg)
    return result.rename(columns={"_bucket": time_column})
