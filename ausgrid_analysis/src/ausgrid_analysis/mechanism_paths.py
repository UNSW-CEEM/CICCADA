"""Generated paths for mechanism results.

There is deliberately no curtailment path in this module.  Gate 7 is unmet.
"""

from __future__ import annotations

from pathlib import Path

from .config import FoundationConfig, SourceScope
from .db import scope_root


def mechanism_results_root(
    config: FoundationConfig,
    scope: SourceScope,
) -> Path:
    return scope_root(config, scope) / "mechanism_results"


def sign_diagnostics_root(config: FoundationConfig) -> Path:
    return config.paths.derived_root / "mechanism_results" / "sign_diagnostics"


def sign_candidate_days_path(config: FoundationConfig) -> Path:
    return sign_diagnostics_root(config) / "candidate_days.parquet"


def sign_site_intervals_path(config: FoundationConfig) -> Path:
    return sign_diagnostics_root(config) / "candidate_site_intervals.parquet"


def sign_phase_intervals_path(config: FoundationConfig) -> Path:
    return sign_diagnostics_root(config) / "candidate_phase_intervals.parquet"


def voltvar_results_path(
    config: FoundationConfig,
    scope: SourceScope,
) -> Path:
    return mechanism_results_root(config, scope) / "voltvar_proxy_results.parquet"


def voltwatt_results_path(
    config: FoundationConfig,
    scope: SourceScope,
) -> Path:
    return mechanism_results_root(config, scope) / "voltwatt_proxy_results.parquet"


def response_observability_path(
    config: FoundationConfig,
    scope: SourceScope,
) -> Path:
    return mechanism_results_root(config, scope) / "response_observability.parquet"


def mechanism_validation_path(
    config: FoundationConfig,
    scope: SourceScope,
) -> Path:
    return scope_root(config, scope) / "audit" / "mechanism_results_validation.json"
