"""Build small site/phase diagnostics and conservative DER-phase mappings."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FoundationConfig, SourceScope
from .db import (
    canonical_output_path,
    connect,
    prepare_output_file,
    site_phase_profile_path,
    site_profile_path,
)
from .logging_utils import get_logger
from .metadata import metadata_output_path
from .schemas import sql_string


def _parquet_glob(path: Path) -> str:
    return str(path / "**" / "*.parquet")


def _phase_profile_query(config: FoundationConfig, scope: SourceScope) -> str:
    source = canonical_output_path(config, scope)
    if not source.is_dir():
        raise FileNotFoundError(f"Canonical dataset is missing: {source}")
    d2 = config.structured_telemetry
    return f"""
        WITH c AS (
            SELECT *
            FROM read_parquet(
                {sql_string(_parquet_glob(source))},
                hive_partitioning = true
            )
        ),
        p AS (
            SELECT
                serial,
                phase,
                count(*) AS n_rows,
                min(timestamp_utc) AS first_timestamp_utc,
                max(timestamp_utc) AS last_timestamp_utc,
                count_if(voltage_v IS NOT NULL) AS n_voltage,
                count_if(voltage_physical_ok) AS n_valid_voltage,
                count_if(voltage_v <= 0) AS n_voltage_at_or_below_zero,
                count_if(current_a IS NOT NULL) AS n_current,
                count_if(p_export_w IS NOT NULL) AS n_active_power,
                count_if(q_absorbing_var IS NOT NULL) AS n_reactive_power,
                count_if((p_export_w IS NULL) <> (q_absorbing_var IS NULL))
                    AS n_p_q_missingness_mismatch,
                approx_quantile(p_export_w, 0.5) FILTER (
                    WHERE hour(timestamp_local) >= {d2.daytime_start_hour}
                      AND hour(timestamp_local) < {d2.daytime_end_hour}
                ) AS daytime_median_p_export_w,
                approx_quantile(p_export_w, 0.5) FILTER (
                    WHERE hour(timestamp_local) >= {d2.nighttime_start_hour}
                      AND hour(timestamp_local) < {d2.nighttime_end_hour}
                ) AS nighttime_median_p_export_w,
                count_if(p_export_w > 0) AS n_export_rows,
                count_if(p_export_w < 0) AS n_import_rows
            FROM c
            GROUP BY serial, phase
        )
        SELECT
            p.*,
            p.n_active_power > 0 AS power_measurement_available,
            p.n_active_power = 0 AS active_power_always_null,
            p.n_reactive_power = 0 AS reactive_power_always_null,
            p.daytime_median_p_export_w - p.nighttime_median_p_export_w
                AS solar_signature_w,
            m.serial IS NOT NULL AS metadata_available,
            m.analysis_cohort,
            coalesce(m.has_battery, false) AS has_battery,
            try_cast(m.install_phase_count AS INTEGER) AS install_phase_count,
            m.solar_capacity_kw,
            m.approved_capacity_kw,
            m.battery_inverter_capacity_kw
        FROM p
        LEFT JOIN read_parquet({sql_string(metadata_output_path(config))}) m
            USING (serial)
        ORDER BY serial, phase
    """.strip()


def derive_site_profiles(
    phase_profiles: pd.DataFrame,
    config: FoundationConfig,
) -> pd.DataFrame:
    """Infer candidate DER phases without silently promoting them to truth."""

    rows: list[dict[str, Any]] = []
    d2 = config.structured_telemetry
    for serial, group in phase_profiles.groupby("serial", sort=True):
        group = group.sort_values("phase").copy()
        available = group.loc[group["power_measurement_available"].fillna(False)]
        first = group.iloc[0]
        install_raw = first.get("install_phase_count")
        install_count = (
            int(install_raw)
            if pd.notna(install_raw) and int(install_raw) in (1, 2, 3)
            else None
        )
        observed = sorted(group["phase"].astype(str).tolist())
        power_phases = sorted(available["phase"].astype(str).tolist())
        selected: list[str] = []
        method = "not_inferred"
        confidence = "unknown"
        margin_ratio: float | None = None
        top_signature: float | None = None

        if install_count is None:
            method = "missing_install_phase_count"
        elif len(power_phases) < install_count:
            method = "insufficient_power_phases"
            confidence = "insufficient"
        elif len(power_phases) == install_count:
            selected = power_phases
            method = "all_power_available_phases"
            confidence = "high"
        else:
            ranked = available.assign(
                _score=pd.to_numeric(available["solar_signature_w"], errors="coerce")
            ).sort_values(["_score", "phase"], ascending=[False, True])
            selected = sorted(ranked.head(install_count)["phase"].astype(str).tolist())
            selected_floor = ranked.iloc[install_count - 1]["_score"]
            next_score = ranked.iloc[install_count]["_score"]
            top_signature = (
                float(ranked.iloc[0]["_score"])
                if pd.notna(ranked.iloc[0]["_score"])
                else None
            )
            if pd.notna(selected_floor) and pd.notna(next_score):
                denominator = max(abs(float(selected_floor)), 1.0)
                margin_ratio = (float(selected_floor) - float(next_score)) / denominator
            method = "ranked_local_day_night_export_signature"
            if (
                pd.notna(selected_floor)
                and float(selected_floor) >= d2.phase_mapping_min_signature_w
                and margin_ratio is not None
            ):
                if margin_ratio >= d2.phase_mapping_high_margin_ratio:
                    confidence = "high"
                elif margin_ratio >= d2.phase_mapping_medium_margin_ratio:
                    confidence = "medium"
                else:
                    confidence = "low"
            else:
                confidence = "low"

        mapping_assessable = confidence in {"high", "medium"}
        metadata_available = bool(first.get("metadata_available", False))
        cohort = first.get("analysis_cohort")
        has_battery = bool(first.get("has_battery", False))
        rows.append(
            {
                "serial": serial,
                "metadata_available": metadata_available,
                "analysis_cohort": cohort,
                "has_battery": has_battery,
                "install_phase_count": install_count,
                "observed_phase_count": len(observed),
                "observed_phases": "|".join(observed),
                "power_phase_count": len(power_phases),
                "power_available_phases": "|".join(power_phases),
                "inferred_der_phases": "|".join(selected),
                "phase_mapping_method": method,
                "phase_mapping_confidence": confidence,
                "phase_mapping_sign_dependency": (
                    "uses_working_active_sign"
                    if method == "ranked_local_day_night_export_signature"
                    else "not_sign_ranked"
                ),
                "phase_mapping_requires_review": True,
                "phase_mapping_margin_ratio": margin_ratio,
                "top_solar_signature_w": top_signature,
                "phase_mapping_assessable": mapping_assessable,
                "solar_only_mapped_cohort": bool(
                    metadata_available
                    and cohort == "solar_only"
                    and not has_battery
                    and mapping_assessable
                    and len(selected) > 0
                ),
                "measurement_basis": "net_meter",
                "voltage_measurement_location": "revenue_meter",
                "formal_inverter_conformance_assessable": False,
            }
        )
    return pd.DataFrame(rows)


def build_site_profiles(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write phase diagnostics and one conservative mapping row per site."""

    logger = get_logger()
    phase_path = prepare_output_file(
        config, site_phase_profile_path(config, scope), overwrite=overwrite
    )
    site_path = prepare_output_file(
        config, site_profile_path(config, scope), overwrite=overwrite
    )
    connection = connect(config)
    try:
        query = _phase_profile_query(config, scope)
        connection.execute(
            f"""COPY ({query}) TO {sql_string(phase_path)}
            (FORMAT PARQUET, COMPRESSION {config.processing.parquet_compression})"""
        )
        phase_frame = connection.execute(
            f"SELECT * FROM read_parquet({sql_string(phase_path)})"
        ).fetchdf()
        site_frame = derive_site_profiles(phase_frame, config)
        connection.register("_site_profile", site_frame)
        connection.execute(
            f"""COPY (SELECT * FROM _site_profile) TO {sql_string(site_path)}
            (FORMAT PARQUET, COMPRESSION {config.processing.parquet_compression})"""
        )
        connection.unregister("_site_profile")
    finally:
        connection.close()

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope.label,
        "site_phase_rows": int(len(phase_frame)),
        "sites": int(len(site_frame)),
        "mapping_confidence": {
            str(k): int(v)
            for k, v in site_frame["phase_mapping_confidence"].value_counts().items()
        },
        "primary_cohort_sites": int(site_frame["solar_only_mapped_cohort"].sum()),
        "battery_sites": int(site_frame["has_battery"].sum()),
        "site_phase_profile": str(phase_path),
        "site_profile": str(site_path),
    }
    logger.info("Structured telemetry profiles written: %s sites", summary["sites"])
    return summary
