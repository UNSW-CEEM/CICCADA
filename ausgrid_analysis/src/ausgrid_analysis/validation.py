"""Post-build validation and exact row-accounting checks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import FoundationConfig, SourceScope
from .db import (
    canonical_output_path,
    connect,
    duplicate_audit_path,
    validation_output_path,
)
from .logging_utils import get_logger, write_json
from .schemas import scoped_source_sql, sql_string


def validate_canonical_phase(
    config: FoundationConfig,
    scope: SourceScope,
) -> dict[str, Any]:
    """Validate uniqueness, accounting, timestamps, quality and metadata coverage."""

    logger = get_logger()
    canonical_dir = canonical_output_path(config, scope)
    audit_path = duplicate_audit_path(config, scope)
    if not canonical_dir.is_dir():
        raise FileNotFoundError(f"Canonical output directory is missing: {canonical_dir}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"Duplicate audit is missing: {audit_path}")

    canonical_glob = str(canonical_dir / "**" / "*.parquet")
    source_sql = scoped_source_sql(config, scope)
    connection = connect(config)
    try:
        source_rows = int(
            connection.execute(f"SELECT count(*) FROM ({source_sql}) s").fetchone()[0]
        )
        duplicate_accounting = connection.execute(
            f"""
            SELECT
                coalesce(sum(
                    CASE WHEN duplicate_class = 'identical_duplicate'
                        THEN row_count - 1 ELSE 0 END
                ), 0) AS identical_rows_collapsed,
                coalesce(sum(
                    CASE WHEN duplicate_class = 'conflicting_duplicate'
                        THEN row_count ELSE 0 END
                ), 0) AS conflicting_rows_quarantined,
                count_if(duplicate_class = 'conflicting_duplicate')
                    AS conflicting_duplicate_keys
            FROM read_parquet({sql_string(audit_path)})
            """
        ).fetchone()
        identical_collapsed = int(duplicate_accounting[0])
        conflicting_quarantined = int(duplicate_accounting[1])
        conflicting_keys = int(duplicate_accounting[2])
        expected_rows = source_rows - identical_collapsed - conflicting_quarantined

        row = connection.execute(
            f"""
            WITH canonical AS (
                SELECT *
                FROM read_parquet(
                    {sql_string(canonical_glob)},
                    hive_partitioning = true
                )
            ),
            duplicate_keys AS (
                SELECT count(*) AS n
                FROM (
                    SELECT serial, timestamp_utc, phase
                    FROM canonical
                    GROUP BY serial, timestamp_utc, phase
                    HAVING count(*) > 1
                )
            )
            SELECT
                count(*) AS canonical_rows,
                count(DISTINCT serial) AS canonical_serials,
                (SELECT n FROM duplicate_keys) AS duplicate_keys_remaining,
                count_if(timestamp_utc IS NULL) AS null_timestamp_rows,
                count_if(phase IS NULL OR trim(phase) = '') AS blank_phase_rows,
                count_if(row_has_null_measurement) AS rows_with_null_measurement,
                count_if(NOT voltage_physical_ok) AS voltage_flagged_rows,
                count_if(NOT metadata_available) AS rows_without_metadata,
                count_if(p_export_w > 0) AS positive_export_rows,
                count_if(p_export_w < 0) AS negative_export_rows,
                count_if(q_absorbing_var > 0) AS positive_absorption_rows,
                count_if(q_absorbing_var < 0) AS negative_absorption_rows,
                min(timestamp_utc) AS first_timestamp_utc,
                max(timestamp_utc) AS last_timestamp_utc,
                min(timestamp_local) AS first_timestamp_local,
                max(timestamp_local) AS last_timestamp_local
            FROM canonical
            """
        ).fetchone()
        names = [item[0] for item in connection.description]
        metrics = dict(zip(names, row, strict=True))

        monthly = connection.execute(
            f"""
            SELECT
                year_utc,
                month_utc,
                count(*) AS n_rows,
                count(DISTINCT serial) AS n_serials
            FROM read_parquet(
                {sql_string(canonical_glob)},
                hive_partitioning = true
            )
            GROUP BY year_utc, month_utc
            ORDER BY year_utc, month_utc
            """
        ).fetchdf()
    finally:
        connection.close()

    accounting_difference = int(metrics["canonical_rows"]) - expected_rows
    failures: list[str] = []
    if accounting_difference != 0:
        failures.append("canonical row accounting does not reconcile")
    if int(metrics["duplicate_keys_remaining"]) != 0:
        failures.append("duplicate canonical keys remain")
    if int(metrics["null_timestamp_rows"]) != 0:
        failures.append("canonical timestamps contain nulls")
    if int(metrics["blank_phase_rows"]) != 0:
        failures.append("canonical phase labels contain nulls or blanks")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "label": scope.label,
            "month": scope.month,
            "site_bucket": scope.site_bucket,
            "bucket_count": scope.bucket_count,
        },
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "source_rows": source_rows,
        "identical_rows_collapsed": identical_collapsed,
        "conflicting_rows_quarantined": conflicting_quarantined,
        "conflicting_duplicate_keys": conflicting_keys,
        "expected_canonical_rows": expected_rows,
        "accounting_difference": accounting_difference,
        **metrics,
        "monthly_coverage": monthly.to_dict(orient="records"),
    }
    write_json(validation_output_path(config, scope), payload)
    logger.info(
        "Canonical validation %s: accounting difference=%s, duplicate keys=%s",
        payload["status"].upper(),
        accounting_difference,
        metrics["duplicate_keys_remaining"],
    )
    if failures:
        raise RuntimeError("Canonical validation failed: " + "; ".join(failures))
    return payload

