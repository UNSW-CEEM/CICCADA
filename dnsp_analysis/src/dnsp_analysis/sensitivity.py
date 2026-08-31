"""Sensitivity analysis over already-made empirical decisions.

Every mechanism-result track rests on a handful of explicit, named
threshold/percentile choices (DER-phase mapping confidence cut points, the
capacity-proxy percentile, the Volt-VAr Q_impact bucket boundaries, the
tolerance fraction). This module re-examines each of those choices against
the real fleet data without silently treating any of them as ground truth --
mirroring the project's broader "never silently accept a proxy as verified"
posture (see ``mechanism_config.py``, ``capacity_proxy.py``).

Three cost tiers, by how expensive it is to test one more candidate value:

- FREE -- ``phase_mapping_sensitivity`` reruns
  ``telemetry_profiles.derive_site_profiles`` (pure pandas) over the
  already-built ``site_phase_profile.parquet``; ``q_impact_bucket_sensitivity``
  bins the non-conformant assessable population by Q_impact exactly once,
  then resums those cached bins under any number of candidate threshold
  sets -- no additional DuckDB query per candidate.
- MODERATE -- ``capacity_percentile_sensitivity`` and
  ``tolerance_fraction_sensitivity`` each need one fresh DuckDB query per
  candidate value (a different percentile changes which rows the
  ``quantile_cont`` aggregate reads; a different tolerance fraction changes
  the required band itself, hence Q_impact's denominator), but never a full
  per-serial/month/voltage-bin mechanism-result rebuild -- every query here
  is fleet-aggregated to a handful of summary rows.

Nothing in this module writes to a track's persisted result files
(``voltvar_proxy_results.parquet`` etc). ``capacity_percentile_sensitivity``
is the one exception that writes anything at all -- it calls
``capacity_proxy.build_capacity_proxy``, which writes to its own namespaced,
percentile-specific path and never touches the production p99 build.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import pandas as pd

from .as4777_curves import Q_IMPACT_THRESHOLDS
from .capacity_proxy import build_capacity_proxy
from .config import FoundationConfig, SourceScope, StructuredTelemetryConfig
from .db import connect, site_phase_profile_path
from .mechanism_config import MechanismAnalysisConfig
from .mechanism_results import _voltvar_classify_status_sql, _voltvar_scored_cte_sql
from .schemas import sql_string
from .telemetry_profiles import derive_site_profiles


def phase_mapping_sensitivity(
    config: FoundationConfig,
    scope: SourceScope,
    *,
    variants: Mapping[str, StructuredTelemetryConfig] | None = None,
) -> pd.DataFrame:
    """FREE: re-run DER-phase mapping confidence classification under
    alternative threshold constants, without requerying raw telemetry.

    ``telemetry_profiles.derive_site_profiles`` is a pure Python/pandas
    function over the already-computed ``site_phase_profile.parquet`` (built
    once by ``telemetry_profiles.build_site_profiles`` and never touched
    here) -- so every variant below costs only a few seconds of in-memory
    recomputation.

    If ``variants`` is not given, the default sweep perturbs each of the
    three ``[structured_telemetry]`` threshold constants
    (``phase_mapping_min_signature_w``, ``phase_mapping_high_margin_ratio``,
    ``phase_mapping_medium_margin_ratio``) independently by +/-25%, holding
    the other two at ``config.structured_telemetry``'s (i.e. analysis.toml's)
    values, alongside the production baseline itself.
    """
    path = site_phase_profile_path(config, scope)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} -- run telemetry_profiles.build_site_profiles(config, scope) "
            "first; phase_mapping_sensitivity never builds it implicitly."
        )
    connection = connect(config)
    try:
        phase_profiles = connection.execute(
            f"SELECT * FROM read_parquet({sql_string(str(path))})"
        ).fetchdf()
    finally:
        connection.close()

    base = config.structured_telemetry
    if variants is None:
        variants = _default_phase_mapping_variants(base)

    rows: list[dict[str, Any]] = []
    for name, variant in variants.items():
        variant_config = dataclasses.replace(config, structured_telemetry=variant)
        site_frame = derive_site_profiles(phase_profiles, variant_config)
        confidence_counts = site_frame["phase_mapping_confidence"].value_counts()
        rows.append(
            {
                "variant": name,
                "phase_mapping_min_signature_w": variant.phase_mapping_min_signature_w,
                "phase_mapping_high_margin_ratio": variant.phase_mapping_high_margin_ratio,
                "phase_mapping_medium_margin_ratio": variant.phase_mapping_medium_margin_ratio,
                "n_sites": int(len(site_frame)),
                "n_high": int(confidence_counts.get("high", 0)),
                "n_medium": int(confidence_counts.get("medium", 0)),
                "n_low": int(confidence_counts.get("low", 0)),
                "n_insufficient": int(confidence_counts.get("insufficient", 0)),
                "n_unknown": int(confidence_counts.get("unknown", 0)),
                "n_mapping_assessable": int(site_frame["phase_mapping_assessable"].sum()),
                "n_solar_only_mapped_cohort": int(
                    site_frame["solar_only_mapped_cohort"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _default_phase_mapping_variants(
    base: StructuredTelemetryConfig,
) -> dict[str, StructuredTelemetryConfig]:
    variants: dict[str, StructuredTelemetryConfig] = {"production": base}
    for field_name in (
        "phase_mapping_min_signature_w",
        "phase_mapping_high_margin_ratio",
        "phase_mapping_medium_margin_ratio",
    ):
        base_value = getattr(base, field_name)
        for factor, label in ((0.75, "minus25pct"), (1.25, "plus25pct")):
            variants[f"{field_name}__{label}"] = dataclasses.replace(
                base, **{field_name: base_value * factor}
            )
    return variants


def capacity_percentile_sensitivity(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
    *,
    percentiles: Sequence[float] = (0.90, 0.95, 0.97, 0.99, 0.995),
    overwrite: bool = False,
) -> pd.DataFrame:
    """MODERATE: rebuild the empirical capacity proxy at several percentiles.

    Only meaningful for an empirical ``capacity_basis`` (today,
    ``p99_net_export_proxy``) -- raises otherwise, matching
    ``capacity_proxy.build_capacity_proxy``'s own guard. Each percentile
    costs one fresh full-table ``quantile_cont`` scan (there is no cached
    intermediate to resample cheaply from, unlike
    ``q_impact_bucket_sensitivity``'s histogram), but every candidate writes
    to its own namespaced, percentile-specific path
    (``mechanism_paths.capacity_proxy_path``) -- this never overwrites the
    production build for ``mechanism.capacity_proxy_percentile``.
    """
    if not mechanism.capacity_is_empirical:
        raise ValueError(
            "capacity_percentile_sensitivity requires an empirical "
            f"capacity_basis (got {mechanism.capacity_basis!r}); "
            "s_rated_kva and solar_capacity_kw_proxy are plain metadata "
            "pass-throughs with no percentile to sweep."
        )

    rows: list[dict[str, Any]] = []
    for percentile in percentiles:
        variant = dataclasses.replace(mechanism, capacity_proxy_percentile=percentile)
        variant.validate()
        summary = build_capacity_proxy(config, scope, variant, overwrite=overwrite)
        connection = connect(config)
        try:
            mean_va, median_va = connection.execute(
                f"""
                SELECT
                    avg(capacity_proxy_va) AS mean_capacity_proxy_va,
                    median(capacity_proxy_va) AS median_capacity_proxy_va
                FROM read_parquet({sql_string(summary['output'])})
                """
            ).fetchone()
        finally:
            connection.close()
        rows.append(
            {
                "capacity_proxy_percentile": percentile,
                "sites": summary["rows"],
                "n_null_proxy": summary["n_null_proxy"],
                "min_capacity_proxy_va": summary["min_capacity_proxy_va"],
                "mean_capacity_proxy_va": mean_va,
                "median_capacity_proxy_va": median_va,
                "max_capacity_proxy_va": summary["max_capacity_proxy_va"],
                "output": summary["output"],
            }
        )
    return pd.DataFrame(rows)


def q_impact_bucket_sensitivity(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
    *,
    threshold_sets: Mapping[str, tuple[float, float, float, float]] | None = None,
    bin_width: float = 0.02,
    clip_min: float = -2.0,
    clip_max: float = 3.0,
) -> pd.DataFrame:
    """FREE: re-bucket the non-conformant assessable population under
    alternative Q_impact threshold sets, from a single cached histogram.

    The 'conformant' bucket is a literal band-membership check (see
    ``mechanism_results._voltvar_classify_status_sql``) and does not depend
    on Q_IMPACT_THRESHOLDS at all -- its count (and the overall
    ``n_assessable`` denominator) is fetched once, held fixed, and excluded
    from the histogrammed pool, so only the split among
    adverse/inactive/major_deficit/minor_deviation/major_surplus varies by
    threshold set. Bucket boundaries are applied to each bin's midpoint, so
    counts are exact up to a quantization error of at most
    ``bin_width / 2`` per boundary crossed -- acceptable for an exploratory
    sweep; narrow ``bin_width`` (default 0.02, vs. the production
    -0.1/0.1/0.9/1.1 spacing of 0.1-0.2) to tighten it further, or fall back
    to a full rebuild (``build_voltvar_results`` under a variant mechanism)
    for an exact count at one specific threshold set.

    Default candidate sets scale the production inactive band
    (``Q_IMPACT_THRESHOLDS[0:2]``, symmetric around 0) and minor_deviation
    band (``Q_IMPACT_THRESHOLDS[2:4]``, symmetric around 1.0) by a shared
    factor in ``(0.5, 0.75, 1.0, 1.25, 1.5)`` -- the ``1.0`` entry reproduces
    ``Q_IMPACT_THRESHOLDS`` exactly.
    """
    mechanism.validate()
    if threshold_sets is None:
        threshold_sets = _default_q_impact_threshold_sets()

    cte_chain = _voltvar_scored_cte_sql(config, scope, mechanism)
    conformant_check = (
        "q_generator_net_proxy_var BETWEEN q_min_final_var AND q_max_final_var"
    )
    clipped = f"least(greatest(q_impact, {clip_min}), {clip_max})"

    connection = connect(config)
    try:
        n_assessable, n_conformant = connection.execute(
            f"""
            WITH {cte_chain}
            SELECT
                count_if(denominator_status = 'assessable') AS n_assessable,
                count_if(denominator_status = 'assessable' AND {conformant_check})
                    AS n_conformant
            FROM scored
            """
        ).fetchone()
        histogram = connection.execute(
            f"""
            WITH {cte_chain}
            SELECT
                floor({clipped} / {bin_width}) * {bin_width} AS bin_lower_q_impact,
                count(*) AS n
            FROM scored
            WHERE denominator_status = 'assessable' AND NOT ({conformant_check})
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()
    finally:
        connection.close()

    n_assessable = int(n_assessable)
    n_conformant = int(n_conformant)
    histogram["bin_mid_q_impact"] = histogram["bin_lower_q_impact"] + bin_width / 2.0

    rows: list[dict[str, Any]] = []
    for name, (t1, t2, t3, t4) in threshold_sets.items():
        counts = {
            "n_adverse": 0,
            "n_inactive": 0,
            "n_major_deficit": 0,
            "n_minor_deviation": 0,
            "n_major_surplus": 0,
        }
        for mid, n in zip(histogram["bin_mid_q_impact"], histogram["n"]):
            counts[_q_impact_bucket(mid, t1, t2, t3, t4)] += int(n)
        row: dict[str, Any] = {
            "threshold_set": name,
            "adverse_cutoff": t1,
            "inactive_cutoff": t2,
            "major_deficit_cutoff": t3,
            "minor_deviation_cutoff": t4,
            "n_assessable": n_assessable,
            "n_conformant": n_conformant,
            **counts,
        }
        if n_assessable:
            row["non_conformance_fraction"] = (
                row["n_adverse"] + row["n_inactive"] + row["n_major_deficit"]
            ) / n_assessable
            row["conformance_fraction"] = (
                row["n_conformant"] + row["n_minor_deviation"] + row["n_major_surplus"]
            ) / n_assessable
        else:
            row["non_conformance_fraction"] = None
            row["conformance_fraction"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def _default_q_impact_threshold_sets() -> dict[str, tuple[float, float, float, float]]:
    t1, t2, t3, t4 = Q_IMPACT_THRESHOLDS
    sets: dict[str, tuple[float, float, float, float]] = {}
    for factor in (0.5, 0.75, 1.0, 1.25, 1.5):
        sets[f"band_x{factor:g}"] = (
            t1 * factor,
            t2 * factor,
            1 - (1 - t3) * factor,
            1 + (t4 - 1) * factor,
        )
    return sets


def _q_impact_bucket(mid: float, t1: float, t2: float, t3: float, t4: float) -> str:
    if mid < t1:
        return "n_adverse"
    if mid <= t2:
        return "n_inactive"
    if mid < t3:
        return "n_major_deficit"
    if mid <= t4:
        return "n_minor_deviation"
    return "n_major_surplus"


def tolerance_fraction_sensitivity(
    config: FoundationConfig,
    scope: SourceScope,
    mechanism: MechanismAnalysisConfig,
    *,
    tolerance_fractions: Sequence[float] = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10),
) -> pd.DataFrame:
    """MODERATE: fleet-wide Volt-VAr classification totals under alternative
    tolerance fractions.

    Unlike the Q_impact bucket thresholds, ``tolerance_fraction`` changes
    the required band itself (``curves``/``bands`` in
    ``_voltvar_scored_cte_sql``) -- both the 'conformant' literal-membership
    check and Q_impact's own denominator (nearest band edge) shift, so
    ``q_impact_bucket_sensitivity``'s single cached histogram cannot answer
    this question; each candidate value needs its own fresh query. Every
    query here is aggregated straight to fleet totals (no per-serial/month/
    voltage-bin breakdown), so it is still far cheaper than a full
    ``build_voltvar_results`` rebuild.
    """
    mechanism.validate()
    rows: list[dict[str, Any]] = []
    connection = connect(config)
    try:
        for tolerance_fraction in tolerance_fractions:
            variant = dataclasses.replace(
                mechanism, tolerance_fraction=tolerance_fraction
            )
            variant.validate()
            cte_chain = _voltvar_scored_cte_sql(config, scope, variant)
            classify = _voltvar_classify_status_sql(variant)
            result = connection.execute(
                f"""
                WITH {cte_chain},
                classified AS (
                    SELECT *, {classify} AS proxy_curve_status
                    FROM scored
                )
                SELECT
                    count_if(denominator_status = 'assessable') AS n_assessable,
                    count_if(proxy_curve_status = 'conformant') AS n_conformant,
                    count_if(proxy_curve_status = 'adverse') AS n_adverse,
                    count_if(proxy_curve_status = 'inactive') AS n_inactive,
                    count_if(proxy_curve_status = 'major_deficit') AS n_major_deficit,
                    count_if(proxy_curve_status = 'minor_deviation')
                        AS n_minor_deviation,
                    count_if(proxy_curve_status = 'major_surplus') AS n_major_surplus
                FROM classified
                """
            ).fetchdf()
            row = result.iloc[0].to_dict()
            row = {key: int(value) for key, value in row.items()}
            row["tolerance_fraction"] = tolerance_fraction
            rows.append(row)
    finally:
        connection.close()

    frame = pd.DataFrame(rows)
    denom = frame["n_assessable"].where(frame["n_assessable"] > 0)
    frame["non_conformance_fraction"] = (
        frame["n_adverse"] + frame["n_inactive"] + frame["n_major_deficit"]
    ) / denom
    frame["conformance_fraction"] = (
        frame["n_conformant"] + frame["n_minor_deviation"] + frame["n_major_surplus"]
    ) / denom
    columns = [
        "tolerance_fraction",
        "n_assessable",
        "n_conformant",
        "n_adverse",
        "n_inactive",
        "n_major_deficit",
        "n_minor_deviation",
        "n_major_surplus",
        "non_conformance_fraction",
        "conformance_fraction",
    ]
    return frame[columns]
