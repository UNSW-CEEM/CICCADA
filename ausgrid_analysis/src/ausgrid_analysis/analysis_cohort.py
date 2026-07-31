"""Build a transparent site-eligibility table for downstream modelling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FoundationConfig, SourceScope
from .db import (
    connect,
    prepare_output_file,
    site_phase_profile_path,
    site_profile_path,
)
from .logging_utils import write_json
from .metadata import metadata_output_path
from .schemas import sql_string


@dataclass(frozen=True)
class CohortRules:
    minimum_power_coverage: float = 0.95
    accepted_mapping_confidence: tuple[str, ...] = ("high", "medium")
    require_explicit_no_controlled_load: bool = True
    require_location: bool = True

    def validate(self) -> CohortRules:
        if not 0 < self.minimum_power_coverage <= 1:
            raise ValueError("minimum_power_coverage must be in (0, 1]")
        if not self.accepted_mapping_confidence:
            raise ValueError("accepted_mapping_confidence cannot be empty")
        return self


def site_eligibility_path(config: FoundationConfig) -> Path:
    return config.paths.derived_root / "analysis_cohort" / "site_eligibility.parquet"


def site_eligibility_summary_path(config: FoundationConfig) -> Path:
    return config.paths.derived_root / "audit" / "site_eligibility_summary.json"


def _normalise_yes_no(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    return "unknown"


def derive_site_eligibility(
    site_profiles: pd.DataFrame,
    phase_profiles: pd.DataFrame,
    metadata: pd.DataFrame,
    rules: CohortRules = CohortRules(),
) -> pd.DataFrame:
    """Return one row per telemetry site with independent eligibility gates."""

    rules.validate()
    sites = site_profiles.copy()
    sites["serial"] = sites["serial"].astype("string")
    phase = phase_profiles.copy()
    phase["serial"] = phase["serial"].astype("string")
    meta = metadata.copy()
    meta["serial"] = meta["serial"].astype("string")

    metadata_candidates = [
        "controlled_load",
        "sub_lat",
        "sub_long",
        "solar_capacity_kw",
        "approved_capacity_kw",
        "s_rated_kva",
        "s_rated_source",
    ]
    meta_columns = ["serial", *[
        column
        for column in metadata_candidates
        if column not in sites.columns and column in meta.columns
    ]]
    sites = sites.merge(meta[meta_columns], on="serial", how="left", validate="one_to_one")
    sites["controlled_load_status"] = sites["controlled_load"].map(_normalise_yes_no)
    sites["has_controlled_load"] = sites["controlled_load_status"].eq("yes")

    selected = sites[["serial", "inferred_der_phases"]].merge(
        phase,
        on="serial",
        how="left",
        validate="one_to_many",
    )
    selected["_selected"] = selected.apply(
        lambda row: str(row["phase"]) in str(row["inferred_der_phases"]).split("|")
        if pd.notna(row["inferred_der_phases"]) and str(row["inferred_der_phases"])
        else False,
        axis=1,
    )
    selected = selected.loc[selected["_selected"]].copy()
    selected["active_power_coverage"] = (
        selected["n_active_power"] / selected["n_rows"].where(selected["n_rows"].gt(0))
    )
    selected["reactive_power_coverage"] = (
        selected["n_reactive_power"] / selected["n_rows"].where(selected["n_rows"].gt(0))
    )
    selected["joint_power_coverage"] = selected[
        ["active_power_coverage", "reactive_power_coverage"]
    ].min(axis=1)
    coverage = (
        selected.groupby("serial", as_index=False)
        .agg(
            inferred_der_phase_count_observed=("phase", "nunique"),
            minimum_active_power_coverage=("active_power_coverage", "min"),
            minimum_reactive_power_coverage=("reactive_power_coverage", "min"),
            minimum_joint_power_coverage=("joint_power_coverage", "min"),
            p_q_missingness_mismatch_rows=("n_p_q_missingness_mismatch", "sum"),
        )
    )
    sites = sites.merge(coverage, on="serial", how="left", validate="one_to_one")
    for column in (
        "minimum_active_power_coverage",
        "minimum_reactive_power_coverage",
        "minimum_joint_power_coverage",
    ):
        sites[column] = sites[column].fillna(0.0)
    sites["inferred_der_phase_count_observed"] = (
        sites["inferred_der_phase_count_observed"].fillna(0).astype(int)
    )
    sites["p_q_missingness_mismatch_rows"] = (
        sites["p_q_missingness_mismatch_rows"].fillna(0).astype(int)
    )

    sites["gate_solar_only"] = sites["analysis_cohort"].eq("solar_only")
    sites["gate_no_battery"] = ~sites["has_battery"].fillna(False)
    sites["gate_no_controlled_load"] = sites["controlled_load_status"].eq("no")
    sites["gate_mapping"] = sites["phase_mapping_confidence"].isin(
        rules.accepted_mapping_confidence
    )
    sites["gate_power_coverage"] = sites["minimum_joint_power_coverage"].ge(
        rules.minimum_power_coverage
    )
    sites["gate_location"] = sites["sub_lat"].notna() & sites["sub_long"].notna()
    sites["gate_capacity_available"] = sites["solar_capacity_kw"].gt(0)

    required = [
        "gate_solar_only",
        "gate_no_battery",
        "gate_mapping",
        "gate_power_coverage",
    ]
    if rules.require_explicit_no_controlled_load:
        required.append("gate_no_controlled_load")
    if rules.require_location:
        required.append("gate_location")
    sites["eligible_for_irradiance_assessment"] = sites[required].all(axis=1)
    sites["eligible_for_decomposition_experiment"] = (
        sites["eligible_for_irradiance_assessment"]
        & sites["gate_capacity_available"]
    )

    reason_map = {
        "gate_solar_only": "not_solar_only",
        "gate_no_battery": "battery",
        "gate_no_controlled_load": "controlled_load_or_unknown",
        "gate_mapping": "mapping_not_accepted",
        "gate_power_coverage": "insufficient_p_q_coverage",
        "gate_location": "location_missing",
        "gate_capacity_available": "solar_capacity_missing",
    }
    sites["exclusion_reasons"] = sites.apply(
        lambda row: "|".join(
            reason for gate, reason in reason_map.items() if not bool(row[gate])
        ),
        axis=1,
    )
    return sites.sort_values("serial").reset_index(drop=True)


def build_site_eligibility(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    rules: CohortRules = CohortRules(),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the small eligibility parquet from completed structured telemetry."""

    profile_path = site_profile_path(config, scope)
    phase_path = site_phase_profile_path(config, scope)
    meta_path = metadata_output_path(config)
    for path in (profile_path, phase_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    connection = connect(config)
    try:
        site_profiles = connection.execute(
            f"SELECT * FROM read_parquet({sql_string(profile_path)})"
        ).fetchdf()
        phase_profiles = connection.execute(
            f"SELECT * FROM read_parquet({sql_string(phase_path)})"
        ).fetchdf()
        metadata = connection.execute(
            f"SELECT * FROM read_parquet({sql_string(meta_path)})"
        ).fetchdf()
        result = derive_site_eligibility(site_profiles, phase_profiles, metadata, rules)
        output = prepare_output_file(
            config, site_eligibility_path(config), overwrite=overwrite
        )
        connection.register("_site_eligibility", result)
        connection.execute(
            f"""COPY (SELECT * FROM _site_eligibility)
            TO {sql_string(output)}
            (FORMAT PARQUET, COMPRESSION {config.processing.parquet_compression})"""
        )
        connection.unregister("_site_eligibility")
    finally:
        connection.close()

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope.label,
        "minimum_power_coverage": rules.minimum_power_coverage,
        "accepted_mapping_confidence": list(rules.accepted_mapping_confidence),
        "sites": int(len(result)),
        "solar_only_sites": int(result["gate_solar_only"].sum()),
        "controlled_load_yes_sites": int(result["has_controlled_load"].sum()),
        "eligible_for_irradiance_assessment": int(
            result["eligible_for_irradiance_assessment"].sum()
        ),
        "eligible_for_decomposition_experiment": int(
            result["eligible_for_decomposition_experiment"].sum()
        ),
        "output": str(output),
    }
    write_json(site_eligibility_summary_path(config), payload)
    return payload
