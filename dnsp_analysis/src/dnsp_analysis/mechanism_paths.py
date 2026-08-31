"""Generated paths for mechanism results.

There is deliberately no curtailment path in this module.  Gate 7 is unmet.
"""

from __future__ import annotations

from pathlib import Path

from .config import FoundationConfig, SourceScope
from .db import scope_root
from .mechanism_config import MechanismAnalysisConfig


def _namespace_segments(mechanism: MechanismAnalysisConfig | None) -> tuple[str, ...]:
    """Ordered namespace segments for a non-default mechanism, else ().

    ``mechanism=None`` or a mechanism at every default value (``der_inferred``
    phase scope, ``s_rated_kva`` capacity) resolves to no segments at all --
    the original, unnamespaced paths every result built before either option
    existed stays exactly where it is and stays reusable. Each independent
    non-default choice adds its own subdirectory level, so
    phase_scope_basis and capacity_basis can be varied independently without
    ever colliding with each other or with the default build.
    """
    segments: list[str] = []
    if mechanism is not None and mechanism.phase_scope_basis != "der_inferred":
        segments.append(f"phase_scope_{mechanism.phase_scope_basis}")
    if mechanism is not None and mechanism.capacity_basis != "s_rated_kva":
        capacity_segment = mechanism.capacity_basis
        if mechanism.capacity_is_empirical:
            capacity_segment += f"_p{mechanism.capacity_proxy_percentile * 100:g}"
        segments.append(f"capacity_{capacity_segment}")
    return tuple(segments)


def _apply_segments(root: Path, mechanism: MechanismAnalysisConfig | None) -> Path:
    for segment in _namespace_segments(mechanism):
        root = root / segment
    return root


def mechanism_results_root(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig | None = None,
) -> Path:
    return _apply_segments(scope_root(config, scope) / "mechanism_results", mechanism)


def sign_diagnostics_root(
    config: FoundationConfig,
    mechanism: MechanismAnalysisConfig | None = None,
) -> Path:
    root = config.paths.derived_root / "mechanism_results" / "sign_diagnostics"
    return _apply_segments(root, mechanism)


def sign_candidate_days_path(
    config: FoundationConfig,
    mechanism: MechanismAnalysisConfig | None = None,
) -> Path:
    return sign_diagnostics_root(config, mechanism) / "candidate_days.parquet"


def sign_site_intervals_path(
    config: FoundationConfig,
    mechanism: MechanismAnalysisConfig | None = None,
) -> Path:
    return sign_diagnostics_root(config, mechanism) / "candidate_site_intervals.parquet"


def sign_phase_intervals_path(
    config: FoundationConfig,
    mechanism: MechanismAnalysisConfig | None = None,
) -> Path:
    return sign_diagnostics_root(config, mechanism) / "candidate_phase_intervals.parquet"


def voltvar_results_path(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig | None = None,
) -> Path:
    return mechanism_results_root(config, scope, mechanism) / "voltvar_proxy_results.parquet"


def voltwatt_results_path(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig | None = None,
) -> Path:
    return mechanism_results_root(config, scope, mechanism) / "voltwatt_proxy_results.parquet"


def response_observability_path(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig | None = None,
) -> Path:
    return mechanism_results_root(config, scope, mechanism) / "response_observability.parquet"


def mechanism_validation_path(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig | None = None,
) -> Path:
    root = _apply_segments(scope_root(config, scope) / "audit", mechanism)
    return root / "mechanism_results_validation.json"


def capacity_proxy_path(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
) -> Path:
    """Path for capacity_proxy.py's empirical per-site capacity table.

    Only meaningful when ``mechanism.capacity_is_empirical``. Lives under
    ``analysis_cohort/`` (alongside ``site_eligibility.parquet``, which it
    joins against) rather than ``mechanism_results/``, since it is itself an
    eligibility-adjacent input, not a mechanism result. Namespaced by
    phase_scope_basis (the P export column it percentiles depends on that
    choice) and by the configured percentile, so p99 and p95 runs, or
    der_inferred and all_phases runs, never collide.
    """
    if not mechanism.capacity_is_empirical:
        raise ValueError(
            f"capacity_proxy_path is only defined for an empirical capacity_basis, "
            f"got {mechanism.capacity_basis!r}"
        )
    pct = mechanism.capacity_proxy_percentile * 100
    root = scope_root(config, scope) / "analysis_cohort"
    return (
        root
        / f"capacity_proxy_{mechanism.capacity_basis}_p{pct:g}_{mechanism.phase_scope_basis}.parquet"
    )
