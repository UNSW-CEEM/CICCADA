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


PHASE_SCOPE_BASES = {"der_inferred", "all_phases"}


@dataclass(frozen=True)
class MechanismAnalysisConfig:
    """Method choices that must never be redefined inside a query or plot."""

    voltage_aggregate: str = "max"
    capacity_basis: str = "s_rated_kva"
    phase_scope_basis: str = "der_inferred"
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
        if self.phase_scope_basis not in PHASE_SCOPE_BASES:
            raise ValueError(
                f"phase_scope_basis must be one of {sorted(PHASE_SCOPE_BASES)}"
            )
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
        """structured_site_intervals column for the configured voltage basis.

        ``der_inferred`` restricts to phases site_profile/site_eligibility
        inferred as DER-connected (may be 1, 2 or 3 phases, per site).
        ``all_phases`` averages/maxes every phase regardless of DER mapping --
        a deliberate sensitivity view, not a claim the standard requires it
        (see docs/MECHANISM_RESULTS.md).
        """
        prefix = "der_" if self.phase_scope_basis == "der_inferred" else ""
        suffix = {"avg": "mean_valid_v", "max": "max_valid_v"}[self.voltage_aggregate]
        return f"{prefix}voltage_{suffix}"

    @property
    def comparison_p_column(self) -> str:
        """structured_site_intervals column for net export under this scope."""
        return {
            "der_inferred": "p_export_der_phase_net_complete_w",
            "all_phases": "p_export_net_observed_w",
        }[self.phase_scope_basis]

    @property
    def comparison_q_absorbing_column(self) -> str:
        """structured_site_intervals column for absorbing-convention net Q."""
        return {
            "der_inferred": "q_absorbing_der_phase_net_complete_var",
            "all_phases": "q_absorbing_net_observed_var",
        }[self.phase_scope_basis]

    @property
    def power_scope_complete_sql(self) -> str:
        """SQL predicate: were all phases in this scope actually measured?

        ``der_inferred`` reuses the persisted ``der_phase_power_complete``
        flag.  ``all_phases`` has no equivalent persisted flag --
        ``p_export_net_observed_w`` sums whatever phases were available
        without requiring completeness -- so this recomputes the same
        all-or-nothing guarantee from ``observed_phase_rows`` /
        ``measured_power_phase_rows`` rather than silently accepting a
        partial-phase sum as if it were a verified total.
        """
        if self.phase_scope_basis == "der_inferred":
            return "s.der_phase_power_complete"
        return (
            "(s.measured_power_phase_rows = s.observed_phase_rows "
            "AND s.observed_phase_rows > 0)"
        )

    @property
    def phase_scope_sql(self) -> str:
        """SQL expression for the ``phase_scope`` output column."""
        if self.phase_scope_basis == "der_inferred":
            return "coalesce(e.inferred_der_phases, 'unmapped')"
        return "'all_phases'"

    @property
    def voltage_basis_label(self) -> str:
        aggregate_word = {"avg": "mean", "max": "maximum"}[self.voltage_aggregate]
        scope_word = {
            "der_inferred": "inferred_der_phase",
            "all_phases": "all_phase",
        }[self.phase_scope_basis]
        return f"{aggregate_word}_{scope_word}_revenue_meter_voltage"

    @property
    def active_sign_ready(self) -> bool:
        return self.active_sign_review_state in SIGN_READY_STATES

    @property
    def reactive_sign_ready(self) -> bool:
        return self.reactive_sign_review_state in SIGN_READY_STATES

    @property
    def methodology_id(self) -> str:
        # The phase segment is only appended for the non-default basis, so
        # every methodology_id already stamped on der_inferred (the default
        # since Delivery 4's first build) keeps its exact original string --
        # existing full/sample outputs stay reusable and comparable, and only
        # the new all_phases track gets a distinctly labelled id.
        phase_segment = (
            "" if self.phase_scope_basis == "der_inferred"
            else f"__phase_{self.phase_scope_basis}"
        )
        return (
            "net_meter_proxy"
            f"__voltage_{self.voltage_aggregate}"
            f"{phase_segment}"
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
