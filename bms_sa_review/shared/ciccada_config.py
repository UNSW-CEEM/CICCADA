"""
Shared constants for CICCADA analysis modules.

These are standard-defined or project-wide values used by Volt-VAr, Volt-Watt, anti-islanding, sustained-operation, and future analysis modes.
These are refering to Australia A only for now.
"""

import pytz

# Database names 
SA  = "solar_analytics"          # legacy Hive tables
SAI = "solar_analytics_iceberg"  # primary Iceberg tables

# Timezone
# AEST = UTC+10, fixed, no DST. Solar physics doesn't observe daylight saving,
# and a fixed offset avoids the discontinuity at DST transitions.
FIXED_OFFSET = pytz.FixedOffset(600)   # AEST = UTC+10, no DST

# AS/NZS 4777.2:2020 Australia A set-points
AS4777 = {
    # Volt-Watt: 100% S_rated at/below V1, ramp to 20% at V2
    "VW": {
        "V1": 253.0,
        "V2": 260.0,
        "P1": 1.00,
        "P2": 0.20,
    },
    # Volt-VAr: +Q1 supplying <=V1, 0 across deadband V2..V3, -Q4 absorbing >=V4
    "VVAR": {
        "V1": 207.0,
        "V2": 220.0,
        "V3": 240.0,
        "V4": 258.0,
        "Q1": 0.44,   # fraction of S_rated, supplying
        "Q4": 0.60,   # fraction of S_rated, absorbing
    },

    # AS/NZS 4777.2:2020 Figure 2.1 minimum reactive-power capability.
    # P and Q are fractions of rated apparent power S_rated.
    # Below 20% S_rated the standard specifies no quantified minimum Q.
    # From >60% S_rated the inverter must operate down to PF 0.8.
    "QCAP": {
        "P_MIN":        0.20,
        "P_FLAT_MAX":   0.60,
        "Q_FLAT":       0.44,
        "PF_MIN":       0.80,
        "P_CIRCLE":     0.80,
    },

    # Volt-VAr Q_impact category thresholds (signed measured/required ratio).
    #   < thr1        : Q_adverse                 wrong direction
    #   thr1 .. thr2  : Q_inactive                no response
    #   thr2 .. thr3  : Q_significant_shortfall   responded, but far short
    #   thr3 .. thr4  : Q_near_conformant         near conformant
    #   > thr4        : Q_major_surplus           over-response
    #
    # R4 (RESOLVED in Stage 2): the two middle bands were named backwards in the
    # original (`Q_minor_deviation` held the 0.1-0.9 shortfall, `Q_major_deficit`
    # held the 0.9-1.1 near-conformant band). The _v2 tables use the corrected
    # names above. The thresholds themselves never changed.
    "QIMP": {"thr1": -0.1, "thr2": 0.1, "thr3": 0.9, "thr4": 1.1},

    "TOL_FRAC":         0.04,        # +/-4% of NAMEPLATE, applied additively in kW
    "SITE_CONF_THRESH": 0.10,        # site conformant if nonconf fraction <= 10%
    "PV_ACTIVE_FRAC":   0.16,        # 'solar generating' = P_pv > 16% S_rated
    "AI_V_BAND":        (260, 265),  # anti-islanding / cease-generate voltage sweep
    "SUSTOP_V_BAND":    (253, 258),  # sustained-operation voltage sweep
    "INTERVAL_H":       5 / 60,      # 5-min interval -> hours (kW sum -> kWh)
}

# ═══════════════════════════════════════════════════════════════════════════
# TABLE NAMES. NOTE: ONLY place these are written down.
# ═══════════════════════════════════════════════════════════════════════════
#
# !! NOT ALL TABLES WERE REBUILT !!
##
# NOT rebuilt (still originals):
#     conformance_sust_op_3w
#     conformance_antiisland
#
# Those two are still queried by notebook 02. They therefore still carry the
# UTC date-extraction bug (R3/R9), avg(voltage) (R1), and no flex-export filter
# (R2). Any conformance rate drawn from them is NOT comparable to the Volt-VAr /
# Volt-Watt rates. `conformance_queries.table_provenance()` prints this warning
# so it cannot be forgotten mid-analysis.
#
TABLES = {
    # Stage 1 rebuilt
    "structured_data":         "structured_data_v2_flex_included",
    "all_uncurtailedpv":       "all_uncurtailedpv_v2_flex_included",

    # Stage 2 rebuilt
    "conformance_voltvar":     "conformance_voltvar_v2_flex_included",
    "conformance_voltwatt":    "conformance_voltwatt_v2_flex_included",
    "conformance_voltwattghi": "conformance_voltwattghi_v2_flex_included",

    # NOT rebuilt. legacy, use with care
    "conformance_sust_op_3w":  "conformance_sust_op_3w",
    "conformance_antiisland":  "conformance_antiisland",
}

# Which of the above are rebuilt. Drives the provenance warning.
REBUILT = {
    "structured_data", 
    "all_uncurtailedpv",
    "conformance_voltvar", 
    "conformance_voltwatt", 
    "conformance_voltwattghi",
}

# Flip the whole analysis back to originals for an A/B comparison.
TABLES_LEGACY = {k: k for k in TABLES}