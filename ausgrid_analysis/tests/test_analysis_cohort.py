from __future__ import annotations

import pandas as pd

from ausgrid_analysis.analysis_cohort import CohortRules, derive_site_eligibility


def _site_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "serial": "1",
                "analysis_cohort": "solar_only",
                "has_battery": False,
                "phase_mapping_confidence": "high",
                "inferred_der_phases": "B",
                "solar_capacity_kw": 6.6,
            },
            {
                "serial": "2",
                "analysis_cohort": "solar_only",
                "has_battery": False,
                "phase_mapping_confidence": "high",
                "inferred_der_phases": "A",
                "solar_capacity_kw": 5.0,
            },
        ]
    )


def _phase_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "serial": "1", "phase": "A", "n_rows": 100,
                "n_active_power": 100, "n_reactive_power": 100,
                "n_p_q_missingness_mismatch": 0,
            },
            {
                "serial": "1", "phase": "B", "n_rows": 100,
                "n_active_power": 99, "n_reactive_power": 98,
                "n_p_q_missingness_mismatch": 1,
            },
            {
                "serial": "2", "phase": "A", "n_rows": 100,
                "n_active_power": 100, "n_reactive_power": 100,
                "n_p_q_missingness_mismatch": 0,
            },
        ]
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "serial": "1", "controlled_load": "No",
                "sub_lat": -33.0, "sub_long": 151.0,
                "s_rated_kva": None, "s_rated_source": "unavailable",
            },
            {
                "serial": "2", "controlled_load": "Yes",
                "sub_lat": -33.1, "sub_long": 151.1,
                "s_rated_kva": None, "s_rated_source": "unavailable",
            },
        ]
    )


def test_eligibility_requires_no_controlled_load_and_p_q_coverage() -> None:
    result = derive_site_eligibility(
        _site_profiles(),
        _phase_profiles(),
        _metadata(),
        CohortRules(minimum_power_coverage=0.95),
    ).set_index("serial")

    assert result.loc["1", "minimum_joint_power_coverage"] == 0.98
    assert bool(result.loc["1", "eligible_for_irradiance_assessment"])
    assert not bool(result.loc["2", "eligible_for_irradiance_assessment"])
    assert "controlled_load_or_unknown" in result.loc["2", "exclusion_reasons"]


def test_low_coverage_is_retained_with_reason() -> None:
    phase = _phase_profiles()
    phase.loc[(phase.serial == "1") & (phase.phase == "B"), "n_reactive_power"] = 80
    result = derive_site_eligibility(
        _site_profiles(), phase, _metadata()
    ).set_index("serial")
    assert not bool(result.loc["1", "gate_power_coverage"])
    assert "insufficient_p_q_coverage" in result.loc["1", "exclusion_reasons"]

