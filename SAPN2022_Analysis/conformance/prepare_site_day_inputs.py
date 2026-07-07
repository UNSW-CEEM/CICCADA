import polars as pl

from checkPVBehaviour import CheckPVBehaviour
from funcs import mapCircuitDataToSite


DAY_COVERAGE_THRESHOLD = 0.80


def collect_site_days(site_number, circuit_details, all_data, days_to_check):
    day_behaviours = []

    for day in days_to_check:
        start_day = pl.datetime(2022, 11, day, 6, 0, 0, time_zone="Australia/Adelaide")
        end_day = pl.datetime(2022, 11, day, 18, 0, 0, time_zone="Australia/Adelaide")

        has_data, wide, _ = mapCircuitDataToSite(
            all_data, circuit_details, site_number, start_day, end_day
        )
        if not has_data:
            continue

        behaviour = CheckPVBehaviour(wide, volCol="voltage_valid")
        day_behaviours.append(
            {
                "day": day,
                "behaviour": behaviour,
                "eligibility": behaviour.day_eligibility_summary(
                    coverage_threshold=DAY_COVERAGE_THRESHOLD
                ),
            }
        )

    return day_behaviours
