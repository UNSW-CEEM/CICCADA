"""Typed, explicit methodology choices for mechanism-result tables."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


SIGN_REVIEW_STATES = {
    "unverified",
    "empirically_supported_pending_provider_confirmation",
    "provider_confirmed",
    "contradicted",
    "inconclusive",
}
SIGN_READY_STATES = {
    "empirically_supported_pending_provider_confirmation",
    "provider_confirmed",
}


@dataclass(frozen=True)
class MechanismAnalysisConfig:
    """Method choices that must never be redefined inside a query or plot."""

    voltage_aggregate: str = "max"
    capacity_basis: str = "s_rated_kva"
    tolerance_fraction: float = 0.04
    voltage_bin_width_v: float = 1.0
    sign_audit_month: str = "2025-01"
    sign_audit_site_count: int = 3
    minimum_response_intervals: int = 12
    minimum_response_voltage_span_v: float = 3.0
    active_sign_review_state: str = "unverified"
    reactive_sign_review_state: str = "unverified"

    def validate(self) -> MechanismAnalysisConfig:
        if self.voltage_aggregate not in {"avg", "max"}:
            raise ValueError("voltage_aggregate must be 'avg' or 'max'")
        if self.capacity_basis != "s_rated_kva":
            raise ValueError(
                "Delivery 4 permits only verified s_rated_kva. "
                "A proxy/sensitivity basis requires a separate user decision."
            )
        if self.tolerance_fraction <= 0:
            raise ValueError("tolerance_fraction must be positive")
        if self.voltage_bin_width_v <= 0:
            raise ValueError("voltage_bin_width_v must be positive")
        if self.sign_audit_site_count not in {2, 3}:
            raise ValueError("sign_audit_site_count must be 2 or 3")
        if self.minimum_response_intervals < 2:
            raise ValueError("minimum_response_intervals must be at least 2")
        if self.minimum_response_voltage_span_v <= 0:
            raise ValueError("minimum_response_voltage_span_v must be positive")
        for name in ("active_sign_review_state", "reactive_sign_review_state"):
            state = getattr(self, name)
            if state not in SIGN_REVIEW_STATES:
                raise ValueError(f"{name} is not a recognised review state: {state}")
        return self

    @property
    def comparison_voltage_column(self) -> str:
        return {
            "avg": "der_voltage_mean_valid_v",
            "max": "der_voltage_max_valid_v",
        }[self.voltage_aggregate]

    @property
    def voltage_basis_label(self) -> str:
        return {
            "avg": "mean_inferred_der_phase_revenue_meter_voltage",
            "max": "maximum_inferred_der_phase_revenue_meter_voltage",
        }[self.voltage_aggregate]

    @property
    def active_sign_ready(self) -> bool:
        return self.active_sign_review_state in SIGN_READY_STATES

    @property
    def reactive_sign_ready(self) -> bool:
        return self.reactive_sign_review_state in SIGN_READY_STATES

    @property
    def methodology_id(self) -> str:
        return (
            "net_meter_proxy"
            f"__voltage_{self.voltage_aggregate}"
            f"__capacity_{self.capacity_basis}"
            f"__tol_{self.tolerance_fraction:g}"
        )


def load_mechanism_config(path: str | Path) -> MechanismAnalysisConfig:
    """Read the optional ``[mechanism_analysis]`` section from analysis TOML."""

    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    section = data.get("mechanism_analysis", {})
    if not isinstance(section, dict):
        raise ValueError("[mechanism_analysis] must be a TOML table")
    return MechanismAnalysisConfig(**section).validate()
