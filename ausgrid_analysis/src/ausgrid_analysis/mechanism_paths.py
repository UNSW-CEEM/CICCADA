"""Generated paths for mechanism results.

There is deliberately no curtailment path in this module.  Gate 7 is unmet.
"""

from __future__ import annotations

from pathlib import Path

from .config import FoundationConfig, SourceScope
from .db import scope_root
from .mechanism_config import MechanismAnalysisConfig


def _phase_scope_dir(mechanism: MechanismAnalysisConfig | None) -> str | None:
    """Namespace segment for a non-default phase_scope_basis, else None.

    ``mechanism=None`` or the default ``der_inferred`` basis resolves to the
    original, unnamespaced paths -- every path already built before this
    option existed stays exactly where it is and stays reusable. Only the
    ``all_phases`` sensitivity track gets its own subdirectory, so the two
    can never collide or be silently mixed together.
    """
    if mechanism is None or mechanism.phase_scope_basis == "der_inferred":
        return None
    return f"phase_scope_{mechanism.phase_scope_basis}"


def mechanism_results_root(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig | None = None,
) -> Path:
    root = scope_root(config, scope) / "mechanism_results"
    segment = _phase_scope_dir(mechanism)
    return root / segment if segment else root


def sign_diagnostics_root(
    config: FoundationConfig,
    mechanism: MechanismAnalysisConfig | None = None,
) -> Path:
    root = config.paths.derived_root / "mechanism_results" / "sign_diagnostics"
    segment = _phase_scope_dir(mechanism)
    return root / segment if segment else root


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
    root = scope_root(config, scope) / "audit"
    segment = _phase_scope_dir(mechanism)
    name = "mechanism_results_validation.json"
    return (root / segment / name) if segment else (root / name)
