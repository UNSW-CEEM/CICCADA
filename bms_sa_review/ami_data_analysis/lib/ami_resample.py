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
    "interval_energy_kwh", "verify_column_units", "resample_to_interval",
]


def interval_energy_kwh(power_w: pd.Series, source_interval_minutes: float) -> pd.Series:
    """kWh delivered in one source interval, from instantaneous power in W. Pure."""
    return power_w / 1000.0 * (source_interval_minutes / 60.0)


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
