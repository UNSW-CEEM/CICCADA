"""Tracked scientific and workflow settings for conformance analysis."""

from datetime import time


PHASE_B_METHODS = ("default", "original", "tier_based", "old_sweep", "blended")
PRIMARY_PHASE_B_METHOD = "tier_based"

VALID_VOLTAGE_MIN = 80.0
VALID_VOLTAGE_MAX = 300.0
VOLTAGE_ROLLING_WINDOW = "10m"

LOCAL_TIMEZONE = "Australia/Adelaide"
SAPN2022_EVENT_DAYS = (13, 14, 15, 16, 17, 19)
SAPN2022_DAY_START = time(6, 0)
SAPN2022_DAY_END = time(18, 0)
SAPN2022_DAY_COVERAGE_THRESHOLD = 0.80

SAPN2022_REQUIRED_SITE_METADATA_ROWS = 1
SAPN2022_MAX_PV_SITE_NET_CIRCUITS = 3

PHASE_A_TAU = 0.3
PHASE_A_EPS = 0.02
PHASE_A_OV1_FLOOR_OFFSET = 0.5
PHASE_B_COMPLIANCE_THRESHOLD_PCT = 90.0

# Plot the primary Phase B method during the main conformance run.
GENERATE_SITE_PLOTS_DEFAULT = True

# Optionally generate comparison plots after the by-method CSVs are written.
GENERATE_METHOD_COMPARISON_PLOTS = True
METHOD_COMPARISON_METHODS = ("tier_based", "blended")
