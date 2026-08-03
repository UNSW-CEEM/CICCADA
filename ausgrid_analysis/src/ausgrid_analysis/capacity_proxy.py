"""Empirical per-site capacity proxy, for magnitude assessment while
``s_rated_kva`` remains unverified (null) for this fleet.

This is one of two explicitly named, separately-decided proxy tracks (see
``mechanism_config.CAPACITY_BASES``) -- the other, ``solar_capacity_kw_proxy``,
is a plain metadata pass-through and needs no separate builder.

For each eligible site (the same core gate ``mechanism_results.py`` uses:
solar-only, no battery, no controlled load, DER-phase mapping assessable,
power-coverage gate), this computes the configured percentile (default 99th)
of the mechanism's own ``comparison_p_column`` -- i.e. observed net-export
power, in watts -- across every interval in ``scope``, and writes it as
``capacity_proxy_va`` (watts treated directly as VA; no unit conversion, since
the source column is already in watts, unlike ``s_rated_kva``/
``solar_capacity_kw`` which are in kW/kVA and need a factor of 1000 where they
are consumed in ``mechanism_results.py``).

This number is net export, i.e. generation minus house load, so it always
sits at or below true gross inverter output. Using it as a capacity proxy
therefore systematically UNDERSTATES true inverter ``S_rated`` -- more so for
sites with heavier daytime self-consumption. It is never written into
``site_eligibility.parquet`` and never treated as a verified rating; it lives
in its own namespaced file (``mechanism_paths.capacity_proxy_path``) and is
only joined into ``mechanism_results.py``'s SQL when
``capacity_basis in CAPACITY_EMPIRICAL_BASES``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis_cohort import site_eligibility_path
from .config import FoundationConfig, SourceScope
from .db import connect, prepare_output_file, structured_site_output_path
from .mechanism_config import MechanismAnalysisConfig
from .mechanism_paths import capacity_proxy_path
from .mechanism_results import core_site_gate_sql
from .schemas import sql_string


def _glob(path: Path) -> str:
    return str(path / "**" / "*.parquet")


def build_capacity_proxy(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the empirical per-site capacity proxy table.

    Requires ``mechanism.capacity_is_empirical`` (i.e.
    ``capacity_basis == 'p99_net_export_proxy'`` today). Percentile and
    source power column both come from ``mechanism``, so a different
    percentile or a different ``phase_scope_basis`` writes to its own
    namespaced path (``mechanism_paths.capacity_proxy_path``) rather than
    silently overwriting a different choice.
    """

    if not mechanism.capacity_is_empirical:
        raise ValueError(
            "build_capacity_proxy requires an empirical capacity_basis "
            f"(got {mechanism.capacity_basis!r}); solar_capacity_kw_proxy and "
            "s_rated_kva are plain metadata pass-throughs and need no builder."
        )
    mechanism.validate()

    site = structured_site_output_path(config, scope)
    eligibility = site_eligibility_path(config)
    if not site.is_dir():
        raise FileNotFoundError(site)
    if not eligibility.is_file():
        raise FileNotFoundError(eligibility)

    p_column = mechanism.comparison_p_column
    percentile = mechanism.capacity_proxy_percentile
    query = f"""
        SELECT
            s.serial,
            quantile_cont(s.{p_column}, {percentile}) AS capacity_proxy_va,
            count(*) AS n_intervals_in_scope,
            count_if(s.{p_column} IS NOT NULL) AS n_intervals_with_power,
            {sql_string(mechanism.capacity_basis)} AS capacity_source,
            {percentile} AS percentile,
            {sql_string(mechanism.phase_scope_basis)} AS phase_scope_basis,
            {sql_string(p_column)} AS source_power_column
        FROM read_parquet({sql_string(_glob(site))}, hive_partitioning=true) s
        LEFT JOIN read_parquet({sql_string(eligibility)}) e USING (serial)
        WHERE ({core_site_gate_sql("e")})
        GROUP BY s.serial
    """
    output = prepare_output_file(config, capacity_proxy_path(config, scope, mechanism), overwrite=overwrite)
    connection = connect(config)
    try:
        connection.execute(
            f"""COPY ({query}) TO {sql_string(output)}
            (FORMAT PARQUET, COMPRESSION {config.processing.parquet_compression})"""
        )
        rows = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({sql_string(output)})"
            ).fetchone()[0]
        )
        coverage = connection.execute(
            f"""
            SELECT
                count_if(capacity_proxy_va IS NULL) AS n_null_proxy,
                min(capacity_proxy_va), max(capacity_proxy_va)
            FROM read_parquet({sql_string(output)})
            """
        ).fetchone()
    finally:
        connection.close()

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope.label,
        "capacity_basis": mechanism.capacity_basis,
        "capacity_proxy_percentile": percentile,
        "phase_scope_basis": mechanism.phase_scope_basis,
        "source_power_column": p_column,
        "rows": rows,
        "n_null_proxy": int(coverage[0]),
        "min_capacity_proxy_va": coverage[1],
        "max_capacity_proxy_va": coverage[2],
        "bias_note": (
            "empirical net-export percentile; systematically understates true "
            "inverter S_rated (net export excludes self-consumed generation)"
        ),
        "output": str(output),
    }
