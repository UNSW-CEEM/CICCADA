"""Explicit parameters for Volt-VAr curtailment attribution."""

from dataclasses import dataclass, replace

from bms_sa_review.shared.ciccada_config import AS4777


@dataclass(frozen=True)
class VoltVarParams:
    years: tuple = (2024, 2025)
    v_low: float = AS4777["VVAR"]["V3"]
    v_high: float = AS4777["VW"]["V1"]
    peak_hour_start: int = 11
    peak_hour_end: int = 14
    ghi_cs_ratio_min: float = 0.95
    apply_ghi_filter: bool = True
    empirical_limit_basis: str = "s_99"
    tolerance_basis: str = "ac_capacity_kw"
    tolerance_fraction: float = AS4777["TOL_FRAC"]
    rating_basis: str = "ac_capacity_kw"
    require_apparent_limit_symptom: bool = True
    flex_selection: str = "exclude"
    max_ac_capacity_kw: float = 30.0
    minimum_flagged_intervals: int = 1

    def validate(self):
        bases = {"s_99", "ac_capacity_kw"}
        if self.empirical_limit_basis not in bases:
            raise ValueError("invalid empirical_limit_basis")
        if self.tolerance_basis not in bases:
            raise ValueError("invalid tolerance_basis")
        if self.rating_basis not in bases:
            raise ValueError("invalid rating_basis")
        if self.flex_selection not in {"exclude", "include", "only"}:
            raise ValueError("invalid flex_selection")
        if self.v_low >= self.v_high:
            raise ValueError("v_low must be below v_high")
        return self

    def with_changes(self, **changes):
        return replace(self, **changes).validate()


PARAMS = VoltVarParams().validate()


def describe(params=PARAMS):
    import pandas as pd
    return pd.DataFrame(
        [(name, getattr(params, name)) for name in params.__dataclass_fields__],
        columns=["parameter", "value"],
    )
