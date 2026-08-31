"""DuckDB connection and output-path safety helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb

from .config import FoundationConfig, SourceScope


def connect(config: FoundationConfig) -> duckdb.DuckDBPyConnection:
    """Open the file-backed foundation database with bounded resources."""

    config.paths.database_path.parent.mkdir(parents=True, exist_ok=True)
    config.paths.temp_directory.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(config.paths.database_path))
    connection.execute(f"SET threads = {int(config.processing.threads)}")
    connection.execute(f"SET memory_limit = '{config.processing.memory_limit}'")
    escaped_tmp = str(config.paths.temp_directory).replace("'", "''")
    connection.execute(f"SET temp_directory = '{escaped_tmp}'")
    connection.execute("SET TimeZone = 'UTC'")
    connection.execute("SET preserve_insertion_order = false")
    return connection


def scope_root(config: FoundationConfig, scope: SourceScope) -> Path:
    if scope.is_full:
        return config.paths.derived_root
    return config.paths.derived_root / "samples" / scope.label


def duplicate_audit_path(config: FoundationConfig, scope: SourceScope) -> Path:
    return scope_root(config, scope) / "audit" / "duplicate_key_audit.parquet"


def duplicate_summary_path(config: FoundationConfig, scope: SourceScope) -> Path:
    return scope_root(config, scope) / "audit" / "duplicate_summary.json"


def canonical_output_path(config: FoundationConfig, scope: SourceScope) -> Path:
    return scope_root(config, scope) / "canonical_phase"


def validation_output_path(config: FoundationConfig, scope: SourceScope) -> Path:
    return scope_root(config, scope) / "audit" / "canonical_validation.json"


def structured_telemetry_root(config: FoundationConfig, scope: SourceScope) -> Path:
    return scope_root(config, scope) / "structured_telemetry"


def site_phase_profile_path(config: FoundationConfig, scope: SourceScope) -> Path:
    return structured_telemetry_root(config, scope) / "site_phase_profile.parquet"


def site_profile_path(config: FoundationConfig, scope: SourceScope) -> Path:
    return structured_telemetry_root(config, scope) / "site_profile.parquet"


def structured_phase_output_path(config: FoundationConfig, scope: SourceScope) -> Path:
    return structured_telemetry_root(config, scope) / "structured_phase_intervals"


def structured_site_output_path(config: FoundationConfig, scope: SourceScope) -> Path:
    return structured_telemetry_root(config, scope) / "structured_site_intervals"


def structured_validation_path(config: FoundationConfig, scope: SourceScope) -> Path:
    return scope_root(config, scope) / "audit" / "structured_telemetry_validation.json"


def ensure_within_derived(config: FoundationConfig, path: Path) -> Path:
    """Resolve a generated path and prove that it is below derived_root."""

    derived = config.paths.derived_root.resolve()
    target = path.resolve()
    try:
        target.relative_to(derived)
    except ValueError as exc:
        raise ValueError(f"Refusing output operation outside derived_root: {target}") from exc
    if target == derived:
        raise ValueError("Refusing to operate on derived_root itself")
    return target


def prepare_output_directory(
    config: FoundationConfig,
    path: Path,
    *,
    overwrite: bool,
) -> Path:
    target = ensure_within_derived(config, path)
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {target}\n"
                "Use --overwrite only after reviewing the existing validation report."
            )
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def prepare_output_file(
    config: FoundationConfig,
    path: Path,
    *,
    overwrite: bool,
) -> Path:
    target = ensure_within_derived(config, path)
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {target}\n"
                "Use --overwrite only after reviewing the existing output."
            )
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
