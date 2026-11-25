"""
Volt-VAr results for a chosen table, metric and tolerance (notebook 04).
========================================================================

One entry point, `site_results`, so a notebook can switch source table,
metric, tolerance and failure definition from a single parameters cell.

Sources (keys of config.tables):
    "conformance_voltvar_zones"  Stage 2 zone x category counts (phase 5):
                                 all three metrics, tol 0% or 4%,
                                 flex-export sites removed.
    "conformance_voltvar"        the original Stage 2 table: legacy
                                 conformance rate only, tol 4% only.

Metrics: "remote_conformance_rate" (default), "nonconformance_frequency",
"legacy_conformance_rate".

Zones: A response required (P >= 20%, V < 220 or > 240 V);
B sufficient power, voltage deadband (P >= 20%, 220-240 V);
C low power, voltage exposed (P < 20%, V < 220 or > 240 V);
D low power, voltage deadband (P < 20%, 220-240 V).
"""

from __future__ import annotations

import conformance_metrics as cm
import conformance_queries as cq

SOURCES = ("conformance_voltvar_zones", "conformance_voltvar")


def site_results(aq, config, source="conformance_voltvar_zones", metric=None, tol=None,
                 failure_definition=None, **metric_options):
    """
    Returns (sites, all_metrics):
      sites        one row per site for `metric`, in the generic columns
                   (nonconf_count, denominator_count, nonconf_frac, conformant,
                   any_nonconformance) used by fleet_result / group_breakdown;
                   sites not assessable under `metric` are dropped
                   (count in sites.attrs['n_not_assessable']).
      all_metrics  one row per site with all three metrics and tags (zone
                   table only; None for the original table).
    """
    from bms_sa_review.shared.ciccada_config import CONFORMANCE

    metric = metric or CONFORMANCE["METRIC_DEFAULT"]
    if source == "conformance_voltvar_zones":
        zc = cq.fetch_vvar_zone_counts(aq, config, tol=tol)
        sites = cm.prepare_vvar_metrics(zc, metric=metric,
                                        failure_definition=failure_definition, **metric_options)
        everything = cm.vvar_site_metrics(zc, failure_definition, **metric_options)
        return sites, everything
    if source == "conformance_voltvar":
        if metric != "legacy_conformance_rate":
            raise ValueError(
                "the original conformance_voltvar table only gives the legacy "
                "conformance rate; use source='conformance_voltvar_zones' for "
                f"{metric!r}"
            )
        if tol not in (None, 0.04):
            raise ValueError("the original conformance_voltvar table was built at tol 4% only")
        if failure_definition not in (None, "reduced"):
            raise ValueError("the original table has the legacy failure buckets only")
        sites = cm.prepare_vvar_conformance(cq.fetch_vvar_site_year(aq, config), config)
        sites.attrs.update(metric=metric, n_not_assessable=0, tol=0.04)
        return sites, None
    raise ValueError(f"source must be one of {SOURCES}, got {source!r}")


def describe(config, source, metric, tol, failure_definition) -> str:
    """One line for figure titles and printouts."""
    table = config.tables[source]
    tol_txt = "4%" if tol in (None, 0.04) else f"{tol:.0%}"
    return (f"{metric} · tol {tol_txt} · failure definition {failure_definition or 'reduced'}"
            f" · {table} · years {', '.join(map(str, config.years))} · {config.day_night}")


# ---------------------------------------------------------------------------
# One site's telemetry for the figures (standard_plots tidy frame + info)
# ---------------------------------------------------------------------------
def _load(fn_local, fn_aws, args, allow_athena):
    try:
        return fn_local(*args)
    except (FileNotFoundError, ValueError) as exc:
        if not allow_athena:
            raise LookupError(
                f"{args} is not in the local Solar Analytics store ({exc}). Set "
                "ALLOW_ATHENA_SITE_LOADS = True to read it from Athena "
                "(one month partition scan, a few GB)."
            ) from exc
        return fn_aws(*args)


def site_day_frame(site_id, date, allow_athena=False):
    """One AEST day ('YYYY-MM-DD'): local store first, Athena only if allowed."""
    from standard_plots.lib import sources as S

    return _load(S.load_local, S.load_aws, (int(site_id), str(date)), allow_athena)


def site_month_frame(site_id, month, allow_athena=False):
    """One AEST month ('YYYY-MM'): local store first, Athena only if allowed."""
    from standard_plots.lib import sources as S

    return _load(lambda s, m: S.load_month(s, m, "local"),
                 lambda s, m: S.load_month(s, m, "aws"),
                 (int(site_id), str(month)), allow_athena)


def best_day(aq, config, site_id, tol=None):
    """The AEST date with the most zone A (response required) intervals."""
    d = cq.fetch_vvar_site_days(aq, config, site_id, tol)
    if d.empty:
        raise LookupError(f"site {site_id} has no rows in the zone table")
    r = d.iloc[0]
    return f"{int(r.year):04d}-{int(r.month):02d}-{int(r.day):02d}"
