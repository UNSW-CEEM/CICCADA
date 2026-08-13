"""
Analysis configuration and Volt-VAr parameters.
===============================================

Deliverable D5. The SolarEdge counterparts of
``data_query/lib/analysis_contract.AnalysisConfig`` and
``data_query/lib/voltvar_params.VoltVarParams``.

Frozen dataclasses, validated on construction, with ``with_changes()`` for
sensitivity sweeps. Every field is a methodological choice, and every choice
appears in ``se_contract.manifest()`` so it travels with the result.

What changed in the port, and why
---------------------------------
``AnalysisConfig`` offers ``rating_basis`` and ``empirical_limit_basis`` as a
choice between ``ac_capacity_kw`` (nameplate) and ``s_99`` (empirical). This
delivery has no nameplate, so both collapse to ``s_99``. The fields are KEPT
rather than removed, so that:

* the substitution is visible in the manifest instead of being invisible by
  absence, and
* nameplate slots straight back in if SolarEdge ever supplies it, without
  touching the query layer.

``flex_selection`` is likewise kept but defaults to ``"include"``. Solar Analytics
had a ``flex_export_detected`` flag from the plateau heuristic; nothing equivalent
exists here yet. ``derating_active`` is the closest signal and gets its own
selector, deliberately separate, because it is not the same thing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from solar_edge.config import se_config as C

__all__ = ["SEAnalysisConfig", "SEVoltVarParams", "CONFIG", "PARAMS", "describe"]

_AS4777 = C.as4777()

#: Capacity bases available in this dataset. Nameplate is absent, so the tuple has
#: one member -- but the axis is preserved so its absence is explicit.
CAPACITY_BASES = ("s_99", "s_95", "s_max")

_VOLTAGE_AGGREGATIONS = ("max", "mean")
_DAY_NIGHT = ("all", "day", "night")
_PHASE_COHORTS = ("all", "single", "three")
_DERATING_SELECTIONS = ("include", "exclude", "only")
_ANOMALY_SELECTIONS = ("exclude", "include")


@dataclass(frozen=True)
class SEAnalysisConfig:
    """Cohort definition and methodological bases shared by every notebook."""

    months: tuple = C.STUDY_MONTHS
    interval_h: float = C.INTERVAL_H
    site_nonconf_threshold: float = _AS4777["SITE_CONF_THRESH"]
    min_site_intervals: int = 1

    # --- capacity ---------------------------------------------------------
    #: Basis for the AS/NZS 4777.2 curves (the S_rated stand-in).
    rating_basis: str = "s_99"
    #: Basis for the apparent-limit test in Method A.
    empirical_limit_basis: str = "s_99"
    #: Basis the +/-4% tolerance is taken as a fraction of. Solar Analytics used
    #: nameplate; with none available this is re-anchored to s_99.
    tolerance_basis: str = "s_99"
    tolerance_fraction: float = _AS4777["TOL_FRAC"]

    # --- measurement choices ----------------------------------------------
    #: Which phase voltage represents the site. 'max' matches the legacy Method A
    #: (`max(voltage)` across circuits); 'mean' matches the corrected Stage 2.
    voltage_aggregation: str = "max"
    phase_cohort: str = "all"

    # --- cohort filters ---------------------------------------------------
    day_night: str = "all"
    derating_selection: str = "include"
    #: The 20 sites that report active power outside daylight hours. Excluded by
    #: default -- but note the reason changed once they were examined properly.
    #:
    #: They are NOT one phenomenon. Five report continuously with a flat overnight
    #: plateau and look like BATTERY STORAGE; fifteen report daylight-only with a
    #: handful of stray night rows and look like a timestamp fault. See
    #: `se_ingest.night_generation_anomaly`.
    #:
    #: Excluding both is defensible for a PV conformance study -- storage sites are
    #: not PV-only, their `s_99` absorbs battery discharge, and a fall in active
    #: power may be charging rather than curtailment -- but it is a JUDGEMENT, and
    #: the five storage sites are of independent interest given CICCADA's BESS
    #: scope. Sweep it in D15; consider a separate storage cohort rather than a
    #: silent exclusion.
    night_anomaly_selection: str = "exclude"
    #: Sites with fewer than this many observed days are dropped from the cohort.
    min_days_observed: int = 0
    max_capacity_kva: float = 30.0

    def validate(self) -> SEAnalysisConfig:
        if not self.months:
            raise ValueError("months cannot be empty")
        unknown = set(self.months) - set(C.STUDY_MONTHS)
        if unknown:
            raise ValueError(f"months outside the delivery: {sorted(unknown)}")
        for name in ("rating_basis", "empirical_limit_basis", "tolerance_basis"):
            if getattr(self, name) not in CAPACITY_BASES:
                raise ValueError(f"{name} must be one of {CAPACITY_BASES}")
        if self.voltage_aggregation not in _VOLTAGE_AGGREGATIONS:
            raise ValueError(f"voltage_aggregation must be one of {_VOLTAGE_AGGREGATIONS}")
        if self.phase_cohort not in _PHASE_COHORTS:
            raise ValueError(f"phase_cohort must be one of {_PHASE_COHORTS}")
        if self.day_night not in _DAY_NIGHT:
            raise ValueError(f"day_night must be one of {_DAY_NIGHT}")
        if self.derating_selection not in _DERATING_SELECTIONS:
            raise ValueError(f"derating_selection must be one of {_DERATING_SELECTIONS}")
        if self.night_anomaly_selection not in _ANOMALY_SELECTIONS:
            raise ValueError(
                f"night_anomaly_selection must be one of {_ANOMALY_SELECTIONS}"
            )
        if not 0.0 <= self.tolerance_fraction < 1.0:
            raise ValueError("tolerance_fraction must be in [0, 1)")
        if not 0.0 < self.site_nonconf_threshold <= 1.0:
            raise ValueError("site_nonconf_threshold must be in (0, 1]")
        if self.max_capacity_kva <= 0:
            raise ValueError("max_capacity_kva must be positive")
        return self

    def with_changes(self, **changes) -> SEAnalysisConfig:
        return replace(self, **changes).validate()


@dataclass(frozen=True)
class SEVoltVarParams:
    """
    Detection window and gates for the Volt-VAr curtailment analysis.

    Defaults reproduce the Solar Analytics run: the 240-253 V band, 11:00-14:00
    AEST, with the apparent-limit symptom gate on.

    The band matters. Below 240 V the standard requires no absorption; above 253 V
    Volt-Watt also engages, so any curtailment there cannot be attributed to
    Volt-VAr alone. Restricting to 240-253 V is what isolates the mechanism, and is
    the methodologically defensible choice recorded in the project notes.
    """

    v_low: float = _AS4777["VVAR"]["V3"]     # 240 V, absorption begins
    v_high: float = _AS4777["VW"]["V1"]      # 253 V, Volt-Watt begins
    peak_hour_start: int = 11
    peak_hour_end: int = 14                  # exclusive upper bound
    require_apparent_limit_symptom: bool = True
    minimum_flagged_intervals: int = 1

    # --- clear-sky gate (inert until D12 supplies irradiance) --------------
    #: Solar Analytics gates on GHI/GHI_cs >= 0.95 so that a shortfall can be
    #: blamed on curtailment rather than cloud. There is no irradiance in this
    #: delivery, so the gate is OFF and Method B is not yet runnable. Turning it
    #: on before D12 lands will raise rather than silently pass.
    apply_ghi_filter: bool = False
    ghi_cs_ratio_min: float = 0.95

    def validate(self) -> SEVoltVarParams:
        if self.v_low >= self.v_high:
            raise ValueError("v_low must be below v_high")
        if not 0 <= self.peak_hour_start < self.peak_hour_end <= 24:
            raise ValueError("require 0 <= peak_hour_start < peak_hour_end <= 24")
        if not 0.0 < self.ghi_cs_ratio_min <= 1.0:
            raise ValueError("ghi_cs_ratio_min must be in (0, 1]")
        if self.minimum_flagged_intervals < 1:
            raise ValueError("minimum_flagged_intervals must be at least 1")
        return self

    def with_changes(self, **changes) -> SEVoltVarParams:
        return replace(self, **changes).validate()


CONFIG = SEAnalysisConfig().validate()
PARAMS = SEVoltVarParams().validate()


def describe(obj) -> "pd.DataFrame":  # noqa: F821
    """Render any params dataclass as a two-column table."""
    import pandas as pd

    return pd.DataFrame(
        [(name, getattr(obj, name)) for name in obj.__dataclass_fields__],
        columns=["parameter", "value"],
    )


def config_dict(config) -> dict:
    out = asdict(config)
    for key, value in out.items():
        if isinstance(value, tuple):
            out[key] = list(value)
    return out
