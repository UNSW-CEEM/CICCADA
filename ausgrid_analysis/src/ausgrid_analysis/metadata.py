"""Metadata workbook normalisation and telemetry-ID reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FoundationConfig, SourceScope
from .db import connect, prepare_output_file, scope_root
from .logging_utils import get_logger, write_json
from .schemas import (
    METADATA_RENAME_MAP,
    NUMERIC_METADATA_COLUMNS,
    scoped_source_sql,
    sql_string,
)


def metadata_output_path(config: FoundationConfig) -> Path:
    return config.paths.derived_root / "metadata" / "metadata_canonical.parquet"


def reconciliation_output_path(
    config: FoundationConfig,
    scope: SourceScope | None = None,
) -> Path:
    root = config.paths.derived_root if scope is None else scope_root(config, scope)
    return root / "metadata" / "id_reconciliation.csv"


def metadata_summary_path(
    config: FoundationConfig,
    scope: SourceScope | None = None,
) -> Path:
    root = config.paths.derived_root if scope is None else scope_root(config, scope)
    return root / "_manifests" / "metadata_summary.json"


def canonicalize_metadata_frame(
    source: pd.DataFrame,
    *,
    id_column: str = "Unique Number ID",
) -> pd.DataFrame:
    """Return one canonical metadata row per provider ID."""

    missing = sorted(set(METADATA_RENAME_MAP) - set(source.columns))
    if missing:
        raise ValueError(f"Metadata sheet is missing required columns: {missing}")
    if id_column not in source.columns:
        raise ValueError(f"Metadata ID column is missing: {id_column}")

    frame = source.rename(columns=METADATA_RENAME_MAP).copy()
    frame["serial"] = (
        frame["serial"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
    )
    if frame["serial"].isna().any() or frame["serial"].eq("").any():
        raise ValueError("Metadata contains null or blank Unique Number IDs")

    duplicate_ids = frame.loc[frame["serial"].duplicated(keep=False), "serial"]
    if not duplicate_ids.empty:
        sample = sorted(duplicate_ids.unique())[:10]
        raise ValueError(f"Metadata IDs are not unique; examples: {sample}")

    for column in NUMERIC_METADATA_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in (
        "controlled_load",
        "der_type",
        "solar_manufacturer",
        "solar_model",
        "battery_manufacturer",
        "battery_model",
    ):
        frame[column] = frame[column].astype("string").str.strip()

    frame["has_battery"] = (
        frame["der_type"].str.contains("battery", case=False, na=False)
        | frame["battery_kwh"].fillna(0).gt(0)
        | frame["battery_inverter_capacity_kw"].fillna(0).gt(0)
    )
    frame["analysis_cohort"] = frame["has_battery"].map(
        {True: "solar_battery", False: "solar_only"}
    )
    frame["s_rated_kva"] = float("nan")
    frame["s_rated_source"] = "unavailable"

    ordered = list(METADATA_RENAME_MAP.values()) + [
        "has_battery",
        "analysis_cohort",
        "s_rated_kva",
        "s_rated_source",
    ]
    return frame[ordered].sort_values("serial").reset_index(drop=True)


def _write_metadata_parquet(
    connection,
    frame: pd.DataFrame,
    path: Path,
    compression: str,
) -> None:
    connection.register("_metadata_frame", frame)
    try:
        connection.execute(
            f"""
            COPY (SELECT * FROM _metadata_frame)
            TO {sql_string(path)}
            (FORMAT PARQUET, COMPRESSION {compression})
            """
        )
    finally:
        connection.unregister("_metadata_frame")


def prepare_metadata(
    config: FoundationConfig,
    scope: SourceScope | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Normalise metadata, write parquet, and reconcile against telemetry IDs."""

    logger = get_logger()
    logger.info("Reading metadata workbook: %s", config.paths.metadata_workbook)
    source = pd.read_excel(
        config.paths.metadata_workbook,
        sheet_name=config.metadata.sheet_name,
        dtype={config.metadata.id_column: "string"},
        engine="openpyxl",
    )
    frame = canonicalize_metadata_frame(source, id_column=config.metadata.id_column)

    metadata_path = metadata_output_path(config)
    metadata_status = "reused"
    if overwrite or not metadata_path.exists():
        metadata_path = prepare_output_file(
            config,
            metadata_path,
            overwrite=overwrite,
        )
        metadata_status = "written"
    reconciliation_path = prepare_output_file(
        config,
        reconciliation_output_path(config, scope),
        overwrite=overwrite,
    )

    connection = connect(config)
    try:
        if metadata_status == "written":
            _write_metadata_parquet(
                connection,
                frame,
                metadata_path,
                config.processing.parquet_compression,
            )
        source_scope = scope or SourceScope(
            bucket_count=config.processing.site_bucket_count
        )
        telemetry_source = scoped_source_sql(config, source_scope)
        reconciliation = connection.execute(
            f"""
            WITH telemetry_ids AS (
                SELECT DISTINCT serial
                FROM ({telemetry_source}) AS telemetry
            ),
            metadata_ids AS (
                SELECT serial
                FROM read_parquet({sql_string(metadata_path)})
            )
            SELECT
                coalesce(t.serial, m.serial) AS serial,
                t.serial IS NOT NULL AS in_telemetry,
                m.serial IS NOT NULL AS in_metadata,
                CASE
                    WHEN t.serial IS NOT NULL AND m.serial IS NOT NULL THEN 'matched'
                    WHEN t.serial IS NOT NULL THEN 'telemetry_only'
                    ELSE 'metadata_only'
                END AS reconciliation_status
            FROM telemetry_ids t
            FULL OUTER JOIN metadata_ids m USING (serial)
            ORDER BY reconciliation_status, serial
            """
        ).fetchdf()
    finally:
        connection.close()

    reconciliation.to_csv(reconciliation_path, index=False)
    status_counts = {
        str(key): int(value)
        for key, value in reconciliation["reconciliation_status"].value_counts().items()
    }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "label": source_scope.label,
            "month": source_scope.month,
            "site_bucket": source_scope.site_bucket,
            "bucket_count": source_scope.bucket_count,
        },
        "source_rows": int(len(source)),
        "canonical_rows": int(len(frame)),
        "unique_metadata_ids": int(frame["serial"].nunique()),
        "cohort_counts": {
            str(key): int(value)
            for key, value in frame["analysis_cohort"].value_counts().items()
        },
        "install_phase_counts": {
            str(key): int(value)
            for key, value in frame["install_phase_count"]
            .value_counts(dropna=False)
            .items()
        },
        "reconciliation_counts": status_counts,
        "metadata_parquet_status": metadata_status,
        "metadata_parquet": str(metadata_path),
        "reconciliation_csv": str(reconciliation_path),
    }
    write_json(metadata_summary_path(config, scope), payload)
    logger.info("Canonical metadata written to %s", metadata_path)
    logger.info("ID reconciliation: %s", status_counts)
    return payload
