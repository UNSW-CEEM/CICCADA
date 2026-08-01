"""AS/NZS 4777.2:2020 curve math for Australia Region A.

All reactive-power values use generator convention: ``+Q`` supplies reactive
power and ``-Q`` absorbs it.  In particular, the high-voltage Volt-VAr endpoint
is stored literally as ``Q4=-0.60``; callers must never flip the curve.

These functions are a straight port of the reviewed Solar Analytics helpers.
The SQL twins exist so fleet builders and Python checks share one source of
truth and can be parity-tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VoltWattCurve:
    v1: float = 253.0
    v2: float = 260.0
    p1: float = 1.00
    p2: float = 0.20


@dataclass(frozen=True)
class VoltVarCurve:
    v1: float = 207.0
    v2: float = 220.0
    v3: float = 240.0
    v4: float = 258.0
    q1: float = 0.44
    q4: float = -0.60


@dataclass(frozen=True)
class ReactiveCapability:
    p_min: float = 0.20
    p_flat_max: float = 0.60
    q_flat: float = 0.44
    pf_min: float = 0.80
    p_circle: float = 0.80


VOLT_WATT = VoltWattCurve()
VOLT_VAR = VoltVarCurve()
Q_CAPABILITY = ReactiveCapability()
TOLERANCE_FRACTION = 0.04
Q_IMPACT_THRESHOLDS = (-0.1, 0.1, 0.9, 1.1)


@dataclass(frozen=True)
class VoltVarIntervalResult:
    """One scalar interval classification in generator convention."""

    q_required: float | None
    q_capability_absorbing: float | None
    capability_assessable: bool
    q_min_final: float | None
    q_max_final: float | None
    q_impact: float | None
    status: str


def vw_max_p(
    voltage_v: float,
    capacity: float,
    curve: VoltWattCurve = VOLT_WATT,
) -> float:
    """Return the Volt-Watt ceiling in the same units as ``capacity``."""

    if voltage_v <= curve.v1:
        return curve.p1 * capacity
    if voltage_v < curve.v2:
        fraction = curve.p1 + (curve.p2 - curve.p1) * (
            voltage_v - curve.v1
        ) / (curve.v2 - curve.v1)
        return fraction * capacity
    return curve.p2 * capacity


def vw_max_p_sql(
    voltage_sql: str,
    capacity_sql: str,
    curve: VoltWattCurve = VOLT_WATT,
) -> str:
    """SQL expression equivalent to :func:`vw_max_p`."""

    return f"""(
        CASE
            WHEN {voltage_sql} <= {curve.v1} THEN {curve.p1} * ({capacity_sql})
            WHEN {voltage_sql} < {curve.v2} THEN
                ({curve.p1} + ({curve.p2} - {curve.p1})
                    * ({voltage_sql} - {curve.v1}) / ({curve.v2} - {curve.v1}))
                * ({capacity_sql})
            ELSE {curve.p2} * ({capacity_sql})
        END
    )"""


def vvar_required_q(
    voltage_v: float,
    capacity: float,
    curve: VoltVarCurve = VOLT_VAR,
) -> float:
    """Return required generator-convention Q in capacity units."""

    if voltage_v <= curve.v1:
        fraction = curve.q1
    elif voltage_v < curve.v2:
        fraction = curve.q1 * (curve.v2 - voltage_v) / (curve.v2 - curve.v1)
    elif voltage_v <= curve.v3:
        fraction = 0.0
    elif voltage_v < curve.v4:
        fraction = curve.q4 * (voltage_v - curve.v3) / (curve.v4 - curve.v3)
    else:
        fraction = curve.q4
    return fraction * capacity


def vvar_required_q_sql(
    voltage_sql: str,
    capacity_sql: str,
    curve: VoltVarCurve = VOLT_VAR,
) -> str:
    """SQL expression equivalent to :func:`vvar_required_q`."""

    return f"""(
        CASE
            WHEN {voltage_sql} <= {curve.v1} THEN {curve.q1} * ({capacity_sql})
            WHEN {voltage_sql} < {curve.v2} THEN
                {curve.q1} * ({curve.v2} - {voltage_sql})
                / ({curve.v2} - {curve.v1}) * ({capacity_sql})
            WHEN {voltage_sql} <= {curve.v3} THEN 0.0
            WHEN {voltage_sql} < {curve.v4} THEN
                {curve.q4} * ({voltage_sql} - {curve.v3})
                / ({curve.v4} - {curve.v3}) * ({capacity_sql})
            ELSE {curve.q4} * ({capacity_sql})
        END
    )"""


def q_cap_absorbing(
    active_power: float,
    capacity: float,
    capability: ReactiveCapability = Q_CAPABILITY,
) -> float:
    """Figure 2.1 absorbing-Q capability, in generator convention."""

    if capacity <= 0:
        return 0.0
    p_abs = abs(active_power)
    if p_abs < capability.p_min * capacity:
        return 0.0
    if p_abs <= capability.p_flat_max * capacity:
        return -capability.q_flat * capacity
    if p_abs <= capability.p_circle * capacity:
        return -p_abs * math.sqrt(1.0 / capability.pf_min**2 - 1.0)
    residual = capacity**2 - p_abs**2
    return -math.sqrt(residual) if residual > 0 else 0.0


def q_cap_absorbing_sql(
    active_power_sql: str,
    capacity_sql: str,
    capability: ReactiveCapability = Q_CAPABILITY,
) -> str:
    """SQL expression equivalent to :func:`q_cap_absorbing`."""

    ratio = math.sqrt(1.0 / capability.pf_min**2 - 1.0)
    return f"""(
        CASE
            WHEN ({capacity_sql}) <= 0 THEN 0.0
            WHEN abs({active_power_sql}) < {capability.p_min} * ({capacity_sql})
                THEN 0.0
            WHEN abs({active_power_sql}) <= {capability.p_flat_max} * ({capacity_sql})
                THEN -{capability.q_flat} * ({capacity_sql})
            WHEN abs({active_power_sql}) <= {capability.p_circle} * ({capacity_sql})
                THEN -abs({active_power_sql}) * {ratio}
            WHEN pow(({capacity_sql}), 2) - pow(abs({active_power_sql}), 2) > 0
                THEN -sqrt(
                    pow(({capacity_sql}), 2) - pow(abs({active_power_sql}), 2)
                )
            ELSE 0.0
        END
    )"""


def q_conformance_floor_absorbing(
    active_power: float,
    capacity: float,
    capability: ReactiveCapability = Q_CAPABILITY,
    curve: VoltVarCurve = VOLT_VAR,
) -> float:
    """Corrected conformance floor that does not relax above 80% capacity."""

    if capacity <= 0:
        return 0.0
    p_abs = abs(active_power)
    if p_abs < capability.p_min * capacity:
        return 0.0
    if p_abs <= capability.p_flat_max * capacity:
        return -capability.q_flat * capacity
    if p_abs <= capability.p_circle * capacity:
        return -p_abs * math.sqrt(1.0 / capability.pf_min**2 - 1.0)
    return curve.q4 * capacity


def q_conformance_floor_absorbing_sql(
    active_power_sql: str,
    capacity_sql: str,
    capability: ReactiveCapability = Q_CAPABILITY,
    curve: VoltVarCurve = VOLT_VAR,
) -> str:
    """SQL expression equivalent to :func:`q_conformance_floor_absorbing`."""

    ratio = math.sqrt(1.0 / capability.pf_min**2 - 1.0)
    return f"""(
        CASE
            WHEN ({capacity_sql}) <= 0 THEN 0.0
            WHEN abs({active_power_sql}) < {capability.p_min} * ({capacity_sql})
                THEN 0.0
            WHEN abs({active_power_sql}) <= {capability.p_flat_max} * ({capacity_sql})
                THEN -{capability.q_flat} * ({capacity_sql})
            WHEN abs({active_power_sql}) <= {capability.p_circle} * ({capacity_sql})
                THEN -abs({active_power_sql}) * {ratio}
            ELSE {curve.q4} * ({capacity_sql})
        END
    )"""


def add_tolerance(
    value: float,
    capacity: float,
    *,
    direction: int,
    tolerance_fraction: float = TOLERANCE_FRACTION,
) -> float:
    """Add an additive fraction of the explicitly selected capacity basis."""

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    return value + direction * tolerance_fraction * capacity


def add_tolerance_sql(
    value_sql: str,
    capacity_sql: str,
    *,
    direction: int,
    tolerance_fraction: float = TOLERANCE_FRACTION,
) -> str:
    """SQL expression equivalent to :func:`add_tolerance`."""

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    return (
        f"(({value_sql}) + ({direction}) * {tolerance_fraction} "
        f"* ({capacity_sql}))"
    )


def q_impact_nearest_edge(
    q_generator: float,
    q_min: float,
    q_max: float,
    *,
    assessable: bool = True,
    eps: float = 1e-9,
) -> float | None:
    """Normalise measured generator-Q against the nearest permitted band edge."""

    if not assessable:
        return None
    reference = q_max if abs(q_generator - q_max) <= abs(q_generator - q_min) else q_min
    if abs(q_max + q_min) <= eps:
        direction = 1.0
    else:
        direction = (
            (1.0 if reference > 0 else -1.0 if reference < 0 else 0.0)
            * (1.0 if q_generator > 0 else -1.0 if q_generator < 0 else 0.0)
        )
    return direction * abs(q_generator) / (abs(reference) + eps)


def q_impact_nearest_edge_sql(
    q_sql: str,
    q_min_sql: str,
    q_max_sql: str,
    *,
    assessable_sql: str = "true",
    eps: float = 1e-9,
) -> str:
    """SQL expression equivalent to :func:`q_impact_nearest_edge`."""

    reference = (
        f"(CASE WHEN abs(({q_sql}) - ({q_max_sql})) "
        f"<= abs(({q_sql}) - ({q_min_sql})) "
        f"THEN ({q_max_sql}) ELSE ({q_min_sql}) END)"
    )
    direction = f"""(
        CASE
            WHEN abs(({q_max_sql}) + ({q_min_sql})) <= {eps} THEN 1.0
            ELSE sign({reference}) * sign({q_sql})
        END
    )"""
    return f"""(
        CASE WHEN {assessable_sql} THEN
            {direction} * abs({q_sql}) / (abs({reference}) + {eps})
        END
    )"""


def classify_voltvar_interval(
    voltage_v: float | None,
    active_power: float | None,
    q_generator: float | None,
    capacity: float | None,
    *,
    capability_profile: str = "review_corrected",
    tolerance_fraction: float = TOLERANCE_FRACTION,
) -> VoltVarIntervalResult:
    """Classify one Volt-VAr interval against the clamped tolerance band.

    P, Q and ``capacity`` must share units. Q must already use generator
    convention. Missing/non-positive capacity is never replaced by another
    metadata field.
    """

    if (
        voltage_v is None
        or active_power is None
        or q_generator is None
        or capacity is None
        or capacity <= 0
    ):
        return VoltVarIntervalResult(
            None, None, False, None, None, None, "not_assessable"
        )
    if capability_profile == "review_corrected":
        q_capability = q_conformance_floor_absorbing(active_power, capacity)
        capability_assessable = abs(active_power) >= Q_CAPABILITY.p_min * capacity
    elif capability_profile == "figure_2_1_circle":
        q_capability = q_cap_absorbing(active_power, capacity)
        capability_assessable = abs(active_power) >= Q_CAPABILITY.p_min * capacity
    else:
        raise ValueError(
            "capability_profile must be 'review_corrected' or "
            "'figure_2_1_circle'"
        )

    q_required = vvar_required_q(voltage_v, capacity)
    tolerance = tolerance_fraction * capacity
    q_unclamped_min = q_required - tolerance
    q_unclamped_max = q_required + tolerance
    q_max_final = (
        max(q_unclamped_max, q_capability + tolerance)
        if q_unclamped_max < 0
        else q_unclamped_max
    )
    q_min_final = (
        min(q_unclamped_min, -q_capability - tolerance)
        if q_unclamped_min > 0
        else q_unclamped_min
    )
    impact = q_impact_nearest_edge(
        q_generator,
        q_min_final,
        q_max_final,
        assessable=capability_assessable,
    )
    if not capability_assessable:
        status = "not_assessable"
    elif q_min_final <= q_generator <= q_max_final:
        status = "conforming"
    else:
        assert impact is not None
        threshold_1, threshold_2, threshold_3, threshold_4 = Q_IMPACT_THRESHOLDS
        if impact < threshold_1:
            status = "Q_adverse"
        elif impact <= threshold_2:
            status = "Q_inactive"
        elif impact < threshold_3:
            status = "Q_significant_shortfall"
        elif impact <= threshold_4:
            status = "Q_near_conformant"
        else:
            status = "Q_major_surplus"
    return VoltVarIntervalResult(
        q_required,
        q_capability,
        capability_assessable,
        q_min_final,
        q_max_final,
        impact,
        status,
    )
