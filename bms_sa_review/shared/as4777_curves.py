"""
Source of truth for AS/NZS 4777.2:2020 (Only Australia A for now)
==============================================================================

Sign convention (generator convention, AS/NZS 4777.2 Fig 3.2):
  +Q = supplying / injecting reactive
  -Q = absorbing / consuming reactive
  In the 240-258 V band the standard REQUIRES Q < 0 (absorbing).
"""

import math

import numpy as np
from shared.ciccada_config import AS4777

# ---------------------------------------------------------------------------
# Pull set-points once
# ---------------------------------------------------------------------------
_VV = AS4777["VVAR"]  # V1..V4, Q1, Q4
_VW = AS4777["VW"]  # V1, V2, P2
_TOL = AS4777["TOL_FRAC"]  # i.e. 0.04
_QC = AS4777["QCAP"]  # From AS477.2.2020 Figure 2.1.


# ===========================================================================
# 1. VOLT-VAR REQUIRED-Q CURVE
# ===========================================================================
def vvar_required_q(
    v,
    s_rated,
    v1=_VV["V1"],
    v2=_VV["V2"],
    v3=_VV["V3"],
    v4=_VV["V4"],
    q1=_VV["Q1"],
    q4=_VV["Q4"],
):
    """
    Reactive power the AS/NZS 4777.2 Australia-A Volt-VAr curve REQUIRES at
    voltage `v`, expressed in the same units as `s_rated` (kvar if s_rated is kW).

    Piecewise:
        V <= V1 (207)          :  +q1 * s_rated           (full supply)
        V1 < V < V2 (207-220)  :  linear ramp +q1 -> 0
        V2 <= V <= V3 (220-240):  0                       (deadband)
        V3 < V < V4 (240-258)  :  linear ramp 0 -> -q4    (absorbing)
        V >= V4 (258)          :  -q4 * s_rated           (full absorb)

    Returns a float.
    Negative = absorbing.
    """
    if v <= v1:
        return q1 * s_rated
    elif v < v2:
        # ramp from +q1*s at v1 down to 0 at v2
        return q1 * s_rated * (v2 - v) / (v2 - v1)
    elif v <= v3:
        return 0.0
    elif v < v4:
        # ramp from 0 at v3 down to -q4*s at v4
        return -q4 * s_rated * (v - v3) / (v4 - v3)
    else:
        return -q4 * s_rated


def vvar_required_q_sql(
    v_col,
    s_col,
    v1=_VV["V1"],
    v2=_VV["V2"],
    v3=_VV["V3"],
    v4=_VV["V4"],
    q1=_VV["Q1"],
    q4=_VV["Q4"],
):
    """
    SQL CASE expression (as a string) that computes the same required-Q curve
    inside an Athena/Trino query. `v_col` and `s_col` are the column names to
    plug in (e.g. 'V' and 'ac_capacity_kw').
    """
    return f"""
        CASE
            WHEN {v_col} <= {v1} THEN {q1} * {s_col}
            WHEN {v_col} <  {v2} THEN {q1} * {s_col} * ({v2} - {v_col}) / ({v2} - {v1})
            WHEN {v_col} <= {v3} THEN 0
            WHEN {v_col} <  {v4} THEN -{q4} * {s_col} * ({v_col} - {v3}) / ({v4} - {v3})
            ELSE -{q4} * {s_col}
        END""".strip()


# ===========================================================================
# 2. VOLT-WATT MAX-P CURVE
# ===========================================================================
def vw_max_p(v, s_rated, v1=_VW["V1"], v2=_VW["V2"], p2=_VW["P2"]):
    """
    Maximum active power the Volt-Watt curve ALLOWS at voltage `v`, in the same
    units as `s_rated`.

        V <= V1 (253)          :  s_rated               (100%)
        V1 < V < V2 (253-260)  :  linear ramp 100% -> p2 (20%)
        V >= V2 (260)          :  p2 * s_rated          (20% floor)

    Note: the 4% tolerance is NOT included here.
    To be added separately with `add_tol_kw()` so the tolerance handling is explicit and auditable.
    """
    if v <= v1:
        return s_rated
    elif v < v2:
        # ramp from 100% at v1 down to p2 at v2
        return s_rated + (p2 * s_rated - s_rated) * (v - v1) / (v2 - v1)
    else:
        return p2 * s_rated


def vw_max_p_sql(v_col, s_col, v1=_VW["V1"], v2=_VW["V2"], p2=_VW["P2"]):
    """SQL CASE expression for the Volt-Watt max-P curve (tolerance NOT added)."""
    return f"""
        CASE
            WHEN {v_col} <= {v1} THEN {s_col}
            WHEN {v_col} <  {v2} THEN {s_col} + ({p2} * {s_col} - {s_col}) * ({v_col} - {v1}) / ({v2} - {v1})
            ELSE {p2} * {s_col}
        END""".strip()


# ===========================================================================
# 3. AS/NZS 4777.2:2020 FIGURE 2.1 MINIMUM REACTIVE-POWER CAPABILITY
# ===========================================================================
def q_cap_absorbing(p, s_rated):
    """
    Minimum absorbing reactive-power capability required by Figure 2.1. (Clause 2.6)

    `s_rated` must be rated apparent power.

    In the current CICCADA metadata form 3rd party monitoring devices,
    `ac_capacity_kw` is used as an explicitly documented proxy
    because aseparate rated-apparent-power field is unavailable.

    Returns the absorbing boundary using the generator convention:
        negative Q = absorbing.

        |P| < 0.20 S       :  0
        0.20 S to 0.60 S   : -0.44 S
        >0.60 S to 0.80 S  : PF 0.8 boundary
        >0.80 S to 1.00 S  : apparent-power circle
        |P| >= S           :  0

    Below 20% S_rated, zero means that Figure 2.1 specifies no quantified minimum capability.
    It does not mean that the inverter is physically incapable of producing reactive power.
    """

    if s_rated <= 0:
        return 0.0

    p_abs = abs(p)
    p_min = _QC["P_MIN"]
    p_flat_max = _QC["P_FLAT_MAX"]
    q_flat = _QC["Q_FLAT"]
    pf_min = _QC["PF_MIN"]
    p_circle = _QC["P_CIRCLE"]

    if p_abs < p_min * s_rated:
        return 0.0

    if p_abs <= p_flat_max * s_rated:
        return -q_flat * s_rated

    if p_abs <= p_circle * s_rated:
        q_to_p = math.sqrt(1.0 / pf_min**2 - 1.0)
        return -p_abs * q_to_p

    val = s_rated**2 - p_abs**2
    return -math.sqrt(val) if val > 0 else 0.0


def q_cap_absorbing_sql(p_col, s_col):
    """
    Athena/Trino equivalent of q_cap_absorbing().

    `s_col` must represent rated apparent power.
    It may be the documented `ac_capacity_kw` proxy,
    or `S_99` only for an explicitly labelled sensitivity run.
    `S_99` is empirical and must not be presented as verified manufacturer S_rated.

    """
    p_min = _QC["P_MIN"]
    p_flat_max = _QC["P_FLAT_MAX"]
    q_flat = _QC["Q_FLAT"]
    pf_min = _QC["PF_MIN"]
    p_circle = _QC["P_CIRCLE"]

    circle_val = f"power({s_col}, 2) - power(abs({p_col}), 2)"
    pf_ratio = f"sqrt(power(1.0 / {pf_min}, 2) - 1.0)"

    return f"""
        CASE
            WHEN {s_col} <= 0 THEN 0
            WHEN abs({p_col}) <  {p_min} * {s_col} THEN 0
            WHEN abs({p_col}) <= {p_flat_max} * {s_col}
                THEN -{q_flat} * {s_col}
            WHEN abs({p_col}) <= {p_circle} * {s_col}
                THEN -abs({p_col}) * {pf_ratio}
            WHEN ({circle_val}) <= 0 THEN 0
            ELSE -sqrt({circle_val})
        END""".strip()


def q_conformance_floor_absorbing(p, s_rated):
    """
    Minimum absorbing-Q response used by the field conformance assessment.

    Unlike q_cap_absorbing(), this does not relax the required Volt-VAr response above 80% S_rated.
    In that region, reactive-power priority requires active power to be reduced where necessary.

        |P| < 0.20 S       : no quantified minimum
        0.20 S to 0.60 S   : -0.44 S
        >0.60 S to 0.80 S  : PF 0.8 boundary
        >0.80 S            : -0.60 S absorbing response available by
                             reducing active power where necessary
    """

    if s_rated <= 0:
        return 0.0

    p_abs = abs(p)
    p_min = _QC["P_MIN"]
    p_flat_max = _QC["P_FLAT_MAX"]
    q_flat = _QC["Q_FLAT"]
    pf_min = _QC["PF_MIN"]
    p_circle = _QC["P_CIRCLE"]
    q_absorbing_max = _VV["Q4"]

    if p_abs < p_min * s_rated:
        return 0.0

    if p_abs <= p_flat_max * s_rated:
        return -q_flat * s_rated

    if p_abs <= p_circle * s_rated:
        q_to_p = math.sqrt(1.0 / pf_min**2 - 1.0)
        return -p_abs * q_to_p

    # Above 80%, do not use the falling edge of the fixed-P circle as an excuse for deficient Q.
    # In that region, reactive-power priority requires active power to be reduced where necessary!!!
    return -q_absorbing_max * s_rated


def q_conformance_floor_absorbing_sql(p_col, s_col):
    """Athena/Trino equivalent of q_conformance_floor_absorbing()."""
    p_min = _QC["P_MIN"]
    p_flat_max = _QC["P_FLAT_MAX"]
    q_flat = _QC["Q_FLAT"]
    pf_min = _QC["PF_MIN"]
    p_circle = _QC["P_CIRCLE"]
    q_absorbing_max = _VV["Q4"]

    pf_ratio = f"sqrt(power(1.0 / {pf_min}, 2) - 1.0)"

    return f"""
        CASE
            WHEN {s_col} <= 0 THEN 0
            WHEN abs({p_col}) < {p_min} * {s_col}
                THEN 0
            WHEN abs({p_col}) <= {p_flat_max} * {s_col}
                THEN -{q_flat} * {s_col}
            WHEN abs({p_col}) <= {p_circle} * {s_col}
                THEN -abs({p_col}) * {pf_ratio}
            ELSE -{q_absorbing_max} * {s_col}
        END""".strip()


# ===========================================================================
# 4. TOLERANCE HELPERS  (±4% of selected rating proxy, additive in kW)
# ===========================================================================
def add_tol_kw(value, capacity, tol_frac=_TOL, sign=+1):
    """Add/subtract 4% of the selected, explicitly labelled capacity basis."""
    return value + sign * tol_frac * capacity


def tol_kw_sql(capacity_col, tol_frac=_TOL):
    """Return the SQL fragment for 4% of the selected capacity-basis column."""
    return f"{tol_frac} * {capacity_col}"


# ===========================================================================
# 5. Volt-VAr CURVE ASSESSMENT: Q IMPACT RELATIVE TO NEAREST BAND EDGE
# ===========================================================================
def q_impact_nearest_edge(
    q_kvar,
    q_min_final,
    q_max_final,
    capability_assessable=True,
    eps=1e-9,
):
    """
    Normalise measured Q against the nearest permitted final band edge.

    A positive result means Q is in the expected direction.
    A negative result means Q is in the wrong direction.
    Ties select Q_max_final.
    """
    if not capability_assessable:
        return None

    if abs(q_kvar - q_max_final) <= abs(q_kvar - q_min_final):
        reference_edge = q_max_final
    else:
        reference_edge = q_min_final

    if abs(q_max_final + q_min_final) <= eps:
        direction = 1.0
    else:
        direction = ((reference_edge > 0) - (reference_edge < 0)) * (
            (q_kvar > 0) - (q_kvar < 0)
        )

    return direction * abs(q_kvar) / (abs(reference_edge) + eps)


def q_impact_nearest_edge_sql(
    q_col,
    q_min_col,
    q_max_col,
    assessable_col,
    eps=1e-9,
):
    """Athena/Trino equivalent of q_impact_nearest_edge()."""
    return f"""
        CASE
            WHEN {assessable_col} = 0 THEN NULL

            WHEN abs({q_col} - {q_max_col})
              <= abs({q_col} - {q_min_col})
            THEN
                (
                    CASE
                        WHEN abs({q_max_col} + {q_min_col}) <= {eps}
                            THEN 1
                        ELSE sign({q_max_col}) * sign({q_col})
                    END
                )
                * (
                    abs({q_col})
                    / (abs({q_max_col}) + {eps})
                )

            ELSE
                (
                    CASE
                        WHEN abs({q_max_col} + {q_min_col}) <= {eps}
                            THEN 1
                        ELSE sign({q_min_col}) * sign({q_col})
                    END
                )
                * (
                    abs({q_col})
                    / (abs({q_min_col}) + {eps})
                )
        END
    """.strip()


# ===========================================================================
# 7. ZONES, RESPONSE BAND AND INTERVAL CATEGORIES  (conformance rework, Nov 2026)
# ===========================================================================
#
# Interpretation of AS/NZS 4777.2:2020 used for field assessment:
# (BMS: See discussion thread from 18/11/2026. Also presentation on project folder)
#   * The Volt-VAr curve (Table 3.7, Australia A) is always the requirement.
#   * Figure 2.1 is the minimum hardware capability, read at MEASURED P.
#     P >= 20%: accept from min(curve, capability) up to the curve.
#     Capability = 44% (20-60%), 0.75 P (60-80%), none above 80% (full curve,
#     P reduced to make room, Cl. 3.3.2.3).
#   * P < 20%: no minimum. Fails only on wrong direction or over-response,
#     each by more than tol.
#   * Deadband 220-240 V: curve asks 0, Q must stay within +/-tol.
#
# Zones (P as fraction of S_rated; voltage boundaries are the curve's own):
#   A  response required                    |  P >= 20%,  V < 220 or V > 240
#   B  sufficient power, voltage deadband   |  P >= 20%,  220 <= V <= 240
#   C  low power, voltage exposed           |  P <  20%,  V < 220 or V > 240   (incl. night)
#   D  low power, voltage deadband          |  P <  20%,  220 <= V <= 240      (incl. night)
#
# Everything below is vectorised (numpy): scalars or arrays both work.
# Units follow `s_rated` (kW / kvar if s_rated is kW), generator convention.

ZONES = ("A", "B", "C", "D")

# Descriptive names; use these in printed tables, plot legends and reports.
ZONE_LABELS = {
    "A": "A - response required",
    "B": "B - sufficient power, voltage deadband",
    "C": "C - low power, voltage exposed",
    "D": "D - low power, voltage deadband",
}

VVAR_CATEGORIES = (
    "pass",
    "adverse",  # wrong direction
    "inactive",  # ~no response
    "significant_shortfall",  # responded, well short of the band
    "near_conformant",  # just short of the band (not a failure)
    "over_response",  # beyond the curve + tol
    "voltage_deadband_violation",  # zones B/D: |Q| > tol
)

_QIMP = AS4777["QIMP"]


def _arr(x):
    return np.asarray(x, dtype=float)


def _p_eff(p, s_rated, p_tol=0.0):
    """|P|, optionally reduced by p_tol*S_rated (in the inverter's favour)."""
    return np.maximum(np.abs(_arr(p)) - p_tol * _arr(s_rated), 0.0)


def vvar_required_q_np(v, s_rated):
    """Vectorised vvar_required_q() (same piecewise curve). NaN in -> NaN out."""
    v = _arr(v)
    s = _arr(s_rated)
    v1, v2, v3, v4 = _VV["V1"], _VV["V2"], _VV["V3"], _VV["V4"]
    q1, q4 = _VV["Q1"], _VV["Q4"]
    with np.errstate(invalid="ignore"):
        frac = np.select(
            [v <= v1, v < v2, v <= v3, v < v4],
            [
                np.full_like(v, q1),
                q1 * (v2 - v) / (v2 - v1),
                np.zeros_like(v),
                -q4 * (v - v3) / (v4 - v3),
            ],
            default=-q4,
        )
    return np.where(np.isnan(v), np.nan, frac * s)


def q_floor_mag(p, s_rated, p_tol=0.0):
    """
    Magnitude of the minimum response Figure 2.1 allows at measured P
    (= abs(q_conformance_floor_absorbing)), vectorised.
        < 20%  : 0 (no minimum)
        20-60% : 0.44 S
        60-80% : 0.75 P  (PF 0.8)
        > 80%  : 0.60 S  (full curve; P must be reduced to make room)
    The same magnitude is used on the supplying side (Figure 2.1 is symmetric).
    """
    s = _arr(s_rated)
    pe = _p_eff(p, s, p_tol)
    q_to_p = np.sqrt(1.0 / _QC["PF_MIN"] ** 2 - 1.0)
    with np.errstate(invalid="ignore"):
        mag = np.select(
            [
                s <= 0,
                pe < _QC["P_MIN"] * s,
                pe <= _QC["P_FLAT_MAX"] * s,
                pe <= _QC["P_CIRCLE"] * s,
            ],
            [
                np.zeros_like(pe),
                np.zeros_like(pe),
                _QC["Q_FLAT"] * s,
                pe * q_to_p,
            ],
            default=_VV["Q4"] * s,
        )
    return mag


def vvar_zone(v, p, s_rated, p_tol=0.0):
    """Zone label 'A'/'B'/'C'/'D' (object array; None where V, P or S is missing)."""
    v = _arr(v)
    s = _arr(s_rated)
    pe = _p_eff(p, s, p_tol)
    with np.errstate(invalid="ignore"):
        high = pe >= _QC["P_MIN"] * s
        exposed = (v > _VV["V3"]) | (v < _VV["V2"])
    zone = np.select(
        [high & exposed, high & ~exposed, ~high & exposed], ["A", "B", "C"], "D"
    ).astype(object)
    zone[np.isnan(v) | np.isnan(pe) | np.isnan(s)] = None
    return zone


def vvar_band(v, p, s_rated, tol=_TOL, p_tol=0.0):
    """
    Accepted Q band (q_min, q_max) under the agreed rules.

        curve < 0 (absorbing, V > 240): [curve - t,  -min(|curve|, floor) + t]
        curve > 0 (supplying, V < 220): [min(curve, floor) - t,  curve + t]
        curve = 0 (deadband)          : [-t, +t]
    with t = tol * S_rated and floor = q_floor_mag(P).

    Below 20% P floor = 0, which gives the low-power wide band
    [curve - t, +t] (absorbing side). Note: the Stage 2 builder's clamp gives
    curve + t (not +t) as the upper edge when |curve| < t at low power.
    This function follows the low-power rule (wrong direction only by more than t).
    """
    s = _arr(s_rated)
    curve = vvar_required_q_np(v, s)
    floor = q_floor_mag(p, s, p_tol)
    t = tol * s
    req = np.minimum(np.abs(curve), floor)
    q_min = np.where(curve > 0, req - t, curve - t)
    q_max = np.where(curve < 0, -req + t, curve + t)
    return q_min, q_max


def vvar_category(q, v, p, s_rated, tol=_TOL, p_tol=0.0, eps=1e-9):
    """
    One category per interval (object array, None where any input is missing).

    Inside the band -> 'pass'. Outside:
      deadband (curve = 0)                 -> 'voltage_deadband_violation'
      beyond the curve-side edge           -> 'over_response'
      short side, band reaches/passes 0    -> 'adverse' (wrong direction
                                              beyond the band, e.g. P < 20%)
      short side otherwise: r = response / near edge, with the existing
      Q_impact thresholds (ciccada_config QIMP):
          r <  thr1          -> 'adverse'
          thr1 <= r <= thr2  -> 'inactive'
          thr2 <  r <  thr3  -> 'significant_shortfall'
          r >= thr3          -> 'near_conformant'
    Where the band does not cross zero this is identical to the existing
    nearest-edge Q_impact buckets. It differs only where the band crosses zero
    (low power, 240-241.2 V at 4%), where the old Q_impact labels a wrong-
    direction response as 'surplus'.
    """
    q = _arr(q)
    s = _arr(s_rated)
    curve = vvar_required_q_np(v, s)
    q_min, q_max = vvar_band(v, p, s, tol, p_tol)
    d = np.sign(curve)
    with np.errstate(invalid="ignore", divide="ignore"):
        in_band = (q >= q_min - eps) & (q <= q_max + eps)
        over = np.where(d < 0, q < q_min - eps, q > q_max + eps)
        edge_mag = d * np.where(d < 0, q_max, q_min)
        r = np.where(
            edge_mag > eps, d * q / np.where(edge_mag > eps, edge_mag, 1.0), -np.inf
        )
        cat = np.select(
            [
                in_band,
                d == 0,
                over,
                r < _QIMP["thr1"],
                r <= _QIMP["thr2"],
                r < _QIMP["thr3"],
            ],
            [
                "pass",
                "voltage_deadband_violation",
                "over_response",
                "adverse",
                "inactive",
                "significant_shortfall",
            ],
            "near_conformant",
        ).astype(object)
    missing = (
        np.isnan(q)
        | np.isnan(curve)
        | np.isnan(q_min)
        | np.isnan(_arr(p))
        | np.isnan(s)
    )
    cat[missing] = None
    return cat


def vvar_binding_label(v, p, s_rated, p_tol=0.0):
    """Which requirement binds (labels as in binding_requirement_examples.xlsx)."""
    s = _arr(s_rated)
    zone = vvar_zone(v, p, s, p_tol)
    curve = vvar_required_q_np(v, s)
    floor = q_floor_mag(p, s, p_tol)
    pf = _p_eff(p, s, p_tol) / s
    cf = np.abs(curve) / s
    with np.errstate(invalid="ignore"):
        label = np.select(
            [
                (zone == "B") | (zone == "D"),
                zone == "C",
                floor < np.abs(curve),
                (pf > _QC["P_CIRCLE"]) & (pf**2 + cf**2 > 1.0),
            ],
            [
                "Voltage deadband (Table 3.7)",
                "Low power (wide band)",
                "Capability (Figure 2.1)",
                "Volt-VAr curve, with P reduced",
            ],
            "Volt-VAr curve",
        ).astype(object)
    label[np.array([z is None for z in np.atleast_1d(zone)]).reshape(label.shape)] = (
        None
    )
    return label


# SQL twins (Athena/Trino). Same rules, column names plugged in.
def _p_eff_sql(p_col, s_col, p_tol=0.0):
    if p_tol:
        return f"greatest(abs({p_col}) - {p_tol} * {s_col}, 0)"
    return f"abs({p_col})"


def q_floor_mag_sql(p_col, s_col, p_tol=0.0):
    """SQL twin of q_floor_mag()."""
    pe = _p_eff_sql(p_col, s_col, p_tol)
    q_to_p = f"sqrt(power(1.0 / {_QC['PF_MIN']}, 2) - 1.0)"
    return f"""
        CASE
            WHEN {s_col} <= 0 THEN 0
            WHEN {pe} <  {_QC["P_MIN"]} * {s_col} THEN 0
            WHEN {pe} <= {_QC["P_FLAT_MAX"]} * {s_col} THEN {_QC["Q_FLAT"]} * {s_col}
            WHEN {pe} <= {_QC["P_CIRCLE"]} * {s_col} THEN {pe} * {q_to_p}
            ELSE {_VV["Q4"]} * {s_col}
        END""".strip()


def vvar_zone_sql(v_col, p_col, s_col, p_tol=0.0):
    """SQL twin of vvar_zone(): 'A'/'B'/'C'/'D', NULL if V, P or S missing."""
    pe = _p_eff_sql(p_col, s_col, p_tol)
    high = f"{pe} >= {_QC['P_MIN']} * {s_col}"
    exposed = f"({v_col} > {_VV['V3']} OR {v_col} < {_VV['V2']})"
    return f"""
        CASE
            WHEN {v_col} IS NULL OR {p_col} IS NULL OR {s_col} IS NULL THEN NULL
            WHEN {high} AND {exposed} THEN 'A'
            WHEN {high} THEN 'B'
            WHEN {exposed} THEN 'C'
            ELSE 'D'
        END""".strip()


def vvar_band_sql(v_col, p_col, s_col, tol=_TOL, p_tol=0.0):
    """SQL twin of vvar_band(): returns (q_min_sql, q_max_sql)."""
    curve = f"({vvar_required_q_sql(v_col, s_col)})"
    floor = f"({q_floor_mag_sql(p_col, s_col, p_tol)})"
    t = f"{tol} * {s_col}"
    req = f"least(abs({curve}), {floor})"
    q_min = f"CASE WHEN {curve} > 0 THEN {req} - {t} ELSE {curve} - {t} END"
    q_max = f"CASE WHEN {curve} < 0 THEN -{req} + {t} ELSE {curve} + {t} END"
    return q_min, q_max


def vvar_category_sql(q_col, curve_col, q_min_col, q_max_col, eps=1e-9):
    """
    SQL twin of vvar_category(), on precomputed columns (curve from
    vvar_required_q_sql, band from vvar_band_sql), as Stage 2 does with CTEs.
    """
    edge = f"(CASE WHEN {curve_col} < 0 THEN {q_max_col} ELSE {q_min_col} END)"
    edge_mag = f"(sign({curve_col}) * {edge})"
    r = f"(sign({curve_col}) * {q_col} / {edge_mag})"
    return f"""
        CASE
            WHEN {q_col} IS NULL OR {curve_col} IS NULL
              OR {q_min_col} IS NULL OR {q_max_col} IS NULL THEN NULL
            WHEN {q_col} >= {q_min_col} - {eps} AND {q_col} <= {q_max_col} + {eps}
                THEN 'pass'
            WHEN {curve_col} = 0 THEN 'voltage_deadband_violation'
            WHEN {curve_col} < 0 AND {q_col} < {q_min_col} - {eps} THEN 'over_response'
            WHEN {curve_col} > 0 AND {q_col} > {q_max_col} + {eps} THEN 'over_response'
            WHEN {edge_mag} <= {eps} THEN 'adverse'
            WHEN {r} <  {_QIMP["thr1"]} THEN 'adverse'
            WHEN {r} <= {_QIMP["thr2"]} THEN 'inactive'
            WHEN {r} <  {_QIMP["thr3"]} THEN 'significant_shortfall'
            ELSE 'near_conformant'
        END""".strip()


# ===========================================================================
# 6. Quick self-test when run directly:  `python as4777_curves.py`
# ===========================================================================
if __name__ == "__main__":
    S = 5.0  # 5 kW test inverter
    print("Volt-VAr required Q (kvar) at key voltages, S_rated =", S, "kW")
    for v in (200, 207, 213.5, 220, 240, 249, 258, 265):
        print(f"  V={v:6.1f}  ->  Q={vvar_required_q(v, S):+7.3f}")
    print("\nVolt-Watt max P (kW):")
    for v in (250, 253, 256.5, 260, 263):
        print(f"  V={v:6.1f}  ->  Pmax={vw_max_p(v, S):6.3f}")
    print("\nCapability (max absorbing Q, kvar) vs P, S_rated =", S)
    for p in (0.5, 2.0, 3.5, 4.5, 5.0):
        print(f"  P={p:5.2f}  ->  Q_cap={q_cap_absorbing(p, S):+7.3f}")
