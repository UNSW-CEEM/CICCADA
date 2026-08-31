"""Structural and methodological validation for Structured telemetry outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import FoundationConfig, SourceScope
from .db import (
    canonical_output_path,
    connect,
    structured_validation_path,
    site_profile_path,
    structured_phase_output_path,
    structured_site_output_path,
)
from .logging_utils import write_json
from .schemas import sql_string


def _glob(path) -> str:
    return str(path / "**" / "*.parquet")


def validate_structured_telemetry(
    config: FoundationConfig,
    scope: SourceScope,
) -> dict[str, Any]:
    """Validate row/key accounting and the safeguards needed before modelling."""

    canonical = canonical_output_path(config, scope)
    phase = structured_phase_output_path(config, scope)
    site = structured_site_output_path(config, scope)
    profile = site_profile_path(config, scope)
    for path in (canonical, phase, site):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not profile.is_file():
        raise FileNotFoundError(profile)

    connection = connect(config)
    try:
        canonical_glob = _glob(canonical)
        phase_glob = _glob(phase)
        site_glob = _glob(site)
        buckets = [
            int(row[0])
            for row in connection.execute(
                f"""SELECT DISTINCT site_bucket
                FROM read_parquet(
                    {sql_string(canonical_glob)}, hive_partitioning=true
                ) ORDER BY site_bucket"""
            ).fetchall()
        ]
        canonical_totals = [0, 0, 0]
        phase_totals = [0, 0, 0, 0, 0, 0]
        site_totals = [0, 0, 0, 0, 0]
        for bucket in buckets:
            canonical_row = connection.execute(
                f"""WITH x AS (
                    SELECT serial, timestamp_utc
                    FROM read_parquet(
                        {sql_string(canonical_glob)}, hive_partitioning=true
                    ) WHERE site_bucket={bucket}
                )
                SELECT
                    (SELECT count(*) FROM x),
                    (SELECT count(*) FROM (
                        SELECT serial, timestamp_utc FROM x GROUP BY ALL
                    )),
                    (SELECT count(DISTINCT serial) FROM x)"""
            ).fetchone()
            phase_row = connection.execute(
                f"""WITH x AS (
                    SELECT *
                    FROM read_parquet(
                        {sql_string(phase_glob)}, hive_partitioning=true
                    ) WHERE site_bucket={bucket}
                )
                SELECT count(*),
                    (SELECT coalesce(sum(n - 1), 0) FROM (
                        SELECT count(*) n FROM x
                        GROUP BY serial, timestamp_utc, phase HAVING n > 1
                    )),
                    count_if(voltage_valid_for_analysis
                        AND (voltage_v <= {config.quality.voltage_min_v}
                             OR voltage_v > {config.quality.voltage_max_v})),
                    count_if(formal_inverter_conformance_assessable),
                    count_if(measurement_basis <> 'net_meter'),
                    count_if(utc_offset_minutes NOT IN (600, 660))
                FROM x"""
            ).fetchone()
            site_row = connection.execute(
                f"""WITH x AS (
                    SELECT *
                    FROM read_parquet(
                        {sql_string(site_glob)}, hive_partitioning=true
                    ) WHERE site_bucket={bucket}
                )
                SELECT count(*),
                    (SELECT coalesce(sum(n - 1), 0) FROM (
                        SELECT count(*) n FROM x
                        GROUP BY serial, timestamp_utc HAVING n > 1
                    )),
                    count_if(NOT der_phase_power_complete
                        AND (p_export_der_phase_net_complete_w IS NOT NULL
                          OR q_absorbing_der_phase_net_complete_var IS NOT NULL)),
                    count_if(formal_inverter_conformance_assessable),
                    count_if(measurement_basis <> 'net_meter')
                FROM x"""
            ).fetchone()
            canonical_totals = [
                total + int(value)
                for total, value in zip(canonical_totals, canonical_row, strict=True)
            ]
            phase_totals = [
                total + int(value)
                for total, value in zip(phase_totals, phase_row, strict=True)
            ]
            site_totals = [
                total + int(value)
                for total, value in zip(site_totals, site_row, strict=True)
            ]
        profile_row = connection.execute(
            f"""SELECT count(*), count_if(has_battery),
                count_if(solar_only_mapped_cohort),
                count_if(phase_mapping_confidence='high'),
                count_if(phase_mapping_confidence='medium'),
                count_if(phase_mapping_confidence='low'),
                count_if(phase_mapping_confidence IN ('unknown','insufficient'))
                FROM read_parquet({sql_string(profile)})"""
        ).fetchone()
        monthly = connection.execute(
            f"""SELECT year(timestamp_utc) AS year_utc,
                month(timestamp_utc) AS month_utc,
                count(*) AS n_site_intervals,
                count(DISTINCT serial) AS n_sites,
                count_if(der_phase_power_complete) AS n_complete_der_power
                FROM read_parquet({sql_string(site_glob)},
                                  hive_partitioning=true)
                GROUP BY 1,2 ORDER BY 1,2"""
        ).fetchdf().to_dict(orient="records")
    finally:
        connection.close()

    failures: list[str] = []
    canonical_rows, canonical_site_keys, canonical_sites = canonical_totals
    phase_rows, phase_duplicate_keys, invalid_valid_voltage, phase_formal, phase_basis = (
        phase_totals[:5]
    )
    unexpected_offsets = phase_totals[5]
    site_rows, site_duplicate_keys, incomplete_nonnull, site_formal, site_basis = (
        site_totals
    )
    if phase_rows != canonical_rows:
        failures.append("structured phase row count differs from canonical")
    if phase_duplicate_keys:
        failures.append("structured phase keys are not unique")
    if site_rows != canonical_site_keys:
        failures.append("structured site row count differs from canonical site-time keys")
    if site_duplicate_keys:
        failures.append("structured site keys are not unique")
    if invalid_valid_voltage:
        failures.append("invalid voltage entered valid-voltage fields")
    if incomplete_nonnull:
        failures.append("incomplete DER-phase sums were converted to numbers")
    if phase_formal or site_formal:
        failures.append("Structured telemetry incorrectly marks formal conformance assessable")
    if phase_basis or site_basis:
        failures.append("Structured telemetry contains a non-net-meter measurement basis")
    if unexpected_offsets:
        failures.append("local timestamp offset is not AEST/AEDT")

    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope.label,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "canonical_rows": canonical_rows,
        "canonical_site_time_keys": canonical_site_keys,
        "canonical_sites": canonical_sites,
        "structured_phase_rows": phase_rows,
        "structured_phase_duplicate_keys": phase_duplicate_keys,
        "structured_site_rows": site_rows,
        "structured_site_duplicate_keys": site_duplicate_keys,
        "invalid_voltage_marked_valid": invalid_valid_voltage,
        "incomplete_power_with_complete_value": incomplete_nonnull,
        "unexpected_utc_offsets": unexpected_offsets,
        "formal_conformance_rows": phase_formal + site_formal,
        "site_profiles": int(profile_row[0]),
        "battery_sites": int(profile_row[1]),
        "primary_cohort_sites": int(profile_row[2]),
        "mapping_high_sites": int(profile_row[3]),
        "mapping_medium_sites": int(profile_row[4]),
        "mapping_low_sites": int(profile_row[5]),
        "mapping_unknown_or_insufficient_sites": int(profile_row[6]),
        "monthly_coverage": monthly,
        "methodology_state": {
            "load_pv_decomposition": "not_yet_performed",
            "battery_handling": "flagged_and_excluded_from_primary_cohort",
            "sign_convention": "working_assumption_not_verified",
            "phase_mapping": "candidate_mapping_with_confidence",
            "timezone": "UTC_and_Australia/Sydney_retained",
            "voltage_location": "revenue_meter_not_corrected",
            "uncurtailed_pv": "not_yet_estimated",
            "comparison_basis": "net_meter_only_no_formal_curve_comparison",
        },
    }
    write_json(structured_validation_path(config, scope), payload)
    return payload
