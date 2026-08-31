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


# ``s_rated_kva`` is the only *verified* basis -- it is null for this fleet
# (see DATA_CONTRACT.md), so magnitude assessment stays honestly unassessable
# under it today. The two proxy bases below are a separate, explicit user
# decision (2026-08-03) to run a full sweep despite that, with the bias of
# each proxy clearly named and documented rather than silently accepted:
#   solar_capacity_kw_proxy -> DC panel nameplate from metadata. Known to
#       systematically OVERSTATE true inverter S_rated (residential DC:AC
#       oversizing is common), which widens every tolerance band/ceiling and
#       biases the assessment lenient.
#   p99_net_export_proxy    -> the configured percentile (default 99th) of
#       observed net-export power per site, computed empirically from the
#       full structured_site_intervals history by capacity_proxy.py. Known
#       to UNDERSTATE true inverter S_rated (net export is generation minus
#       house load, so it is always <= true generation), which biases the
#       assessment conservative. Never confused with a verified rating --
#       see capacity_proxy.py's module docstring for exactly how it is
#       computed and joined in.
# Building both brackets the true (unknown) S_rated between a lenient and a
# conservative estimate; neither is treated as ground truth.
CAPACITY_METADATA_COLUMNS = {
    "s_rated_kva": "s_rated_kva",
    "solar_capacity_kw_proxy": "solar_capacity_kw",
}
CAPACITY_EMPIRICAL_BASES = {"p99_net_export_proxy"}
CAPACITY_BASES = set(CAPACITY_METADATA_COLUMNS) | CAPACITY_EMPIRICAL_BASES


@dataclass(frozen=True)
class MechanismAnalysisConfig:
    """Method choices that must never be redefined inside a query or plot."""

    voltage_aggregate: str = "max"
    capacity_basis: str = "s_rated_kva"
    phase_scope_basis: str = "der_inferred"
    capacity_proxy_percentile: float = 0.99
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
        if self.capacity_basis not in CAPACITY_BASES:
            raise ValueError(
                f"capacity_basis must be one of {sorted(CAPACITY_BASES)}. "
                "Delivery 4's original default is only verified s_rated_kva; "
                "any proxy/sensitivity basis must be one of these explicitly "
                "named, separately-decided tracks -- never a silent stand-in."
            )
        if not 0.0 < self.capacity_proxy_percentile < 1.0:
            raise ValueError("capacity_proxy_percentile must be strictly between 0 and 1")
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
    def capacity_is_empirical(self) -> bool:
        """True for a capacity_basis computed from telemetry (capacity_proxy.py),
        as opposed to a direct metadata pass-through (s_rated_kva,
        solar_capacity_kw_proxy).
        """
        return self.capacity_basis in CAPACITY_EMPIRICAL_BASES

    @property
    def capacity_metadata_column(self) -> str | None:
        """site_eligibility column supplying capacity_reference_va, for the
        two metadata pass-through bases. ``None`` for an empirical basis --
        that value comes from capacity_proxy.py's output table instead, via
        a separate join (see mechanism_results.py's _base_site_sql).
        """
        return CAPACITY_METADATA_COLUMNS.get(self.capacity_basis)

    @property
    def capacity_basis_label(self) -> str:
        """Short human-readable description of the configured capacity basis,
        for methodology tables/subtitles -- never silently implies 'verified'
        for a proxy basis.
        """
        if self.capacity_basis == "s_rated_kva":
            return "verified s_rated_kva (null for this fleet today)"
        if self.capacity_basis == "solar_capacity_kw_proxy":
            return "DC solar nameplate proxy (known lenient bias vs true S_rated)"
        if self.capacity_basis == "p99_net_export_proxy":
            pct = self.capacity_proxy_percentile * 100
            return (
                f"empirical P{pct:g} observed net-export proxy "
                "(known conservative bias vs true S_rated)"
            )
        raise ValueError(f"no label defined for capacity_basis={self.capacity_basis!r}")

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
        # capacity_basis was always part of this id (default 's_rated_kva'),
        # so every existing string is unchanged. The percentile segment only
        # appears for the empirical basis, so two different percentiles
        # (e.g. p99 vs p95) never collide on the same id/path.
        capacity_segment = self.capacity_basis
        if self.capacity_basis in CAPACITY_EMPIRICAL_BASES:
            capacity_segment += f"_p{self.capacity_proxy_percentile * 100:g}"
        return (
            "net_meter_proxy"
            f"__voltage_{self.voltage_aggregate}"
            f"{phase_segment}"
            f"__capacity_{capacity_segment}"
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
