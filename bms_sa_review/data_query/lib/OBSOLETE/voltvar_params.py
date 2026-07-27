"""
voltvar_params.py  —  Detection parameters for the Volt-VAr curtailment analysis
================================================================================

Lifted out of `03_voltvar_curtailment_detection.ipynb` cell 8 so the notebook
stays a pure orchestrator.

Import the default and override per-run if you need to:

    from voltvar_params import PARAMS
    p = {**PARAMS, "GHI_CS_RATIO_MIN": 0.85}     # one-off sweep value

NEW IN PHASE 4A
---------------
`EXCLUDE_FLEX` (R18). Sites with `flex_export_detected = True` are under a
DNSP-imposed dynamic export limit. Their active power is reduced by an EXTERNAL
instruction, not by Volt-VAr reactive absorption eating into the S-circle — so
including them contaminates the curtailment attribution. Default True (exclude).
Set False to reproduce the un-filtered numbers for the sensitivity comparison
you'll report in the paper.
"""

from bms_sa_review.data_query.OBSOLETEciccada_config import AS4777

PARAMS = {
    # ── Scan scope ──────────────────────────────────────────────────────────
    "YEARS":              [2024, 2025],  # years to scan in `ts` for Method A
    "GHI_YEARS":          [2024, 2025],  # years with structured_data / all_uncurtailedpv coverage
    "MAX_AC_CAPACITY_KW": 30.0,          # residential-scale systems only

    # ── Volt-VAr-only voltage band ──────────────────────────────────────────
    # Lower bound: below 240 V the standard requires Q = 0 (deadband), so any
    # S-limit hit there is NOT Volt-VAr curtailment.
    # Upper bound: at 253 V Volt-Watt begins, and the two mechanisms overlap in
    # 253-258 V with no published way to disaggregate them. Restricting to
    # 240-253 V is the defensible choice.
    "V_LOW":  240.0,
    "V_HIGH": 253.0,

    # ── Peak-solar window (AEST) ────────────────────────────────────────────
    "PEAK_HOUR_START": 11,
    "PEAK_HOUR_END":   13,

    # ── Clear-sky filter ────────────────────────────────────────────────────
    # ghi/ghi_cs >= this is treated as "clear enough". NOTE: not established
    # elsewhere in the literature — tunable. Sweep 0.80-0.95 for the paper.
    "GHI_CS_RATIO_MIN": 0.95,

    # ── "At the apparent-power limit" tolerance ─────────────────────────────
    # Consistent with the 4%-of-nameplate convention used everywhere else
    # (replaces the old, arbitrary S_LIMIT_FRAC = 0.97).
    "S_TOL_FRAC": AS4777["TOL_FRAC"],

    # ── Apparent-power limit basis ──────────────────────────────────────────
    # True : use S_99 (empirical 99th-pct apparent power) as the S-circle radius
    # False: use ac_capacity_kw (nameplate)
    # S_99 is the physically correct choice — it's what the inverter demonstrably
    # delivers. See R10.
    "USE_S99": True,

    # ── Cohort filters ──────────────────────────────────────────────────────
    # "all" matches the established 16,148-site cohort definition
    # (1 inverter per site, up to 3 circuits). Also: "single" | "three".
    "PHASE_FILTER": "all",

    # R18 — exclude DNSP flexible-export sites (see module docstring).
    "EXCLUDE_FLEX": True,
}


def describe(params=None):
    """Print the active parameter set. Call from the notebook after import."""
    p = params or PARAMS
    print("Volt-VAr detection parameters:")
    for k, v in p.items():
        print(f"  {k:20s} = {v}")
    return p
