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
# TBD -> Review
FIXED_OFFSET = pytz.FixedOffset(600)   # AEST = UTC+10, no DST

# AS/NZS 4777.2:2020 Australia A set-points
AS4777 = {
    "VVAR": {
        "V1": 207.0,
        "V2": 220.0,
        "V3": 240.0,
        "V4": 258.0,
        "Q1": 0.44,    # fraction of S_rated supplying
        "Q4": 0.60,    # fraction of S_rated absorbing
    },
    "VW": {
        "V1": 253.0,
        "V2": 260.0,
        "P2": 0.20,
    },
    "TOL_FRAC":   0.04,    # established 4% AC-capacity tolerance
    "INTERVAL_H": 5 / 60,  # 5-minute intervals in hours
}