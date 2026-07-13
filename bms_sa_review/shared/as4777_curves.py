"""
Source of truth for AS/NZS 4777.2:2020 (Only Australia A for now)
==============================================================================

Sign convention (generator convention, AS/NZS 4777.2 Fig 3.2):
  +Q = supplying / injecting reactive
  -Q = absorbing / consuming reactive
  In the 240-258 V band the standard REQUIRES Q < 0 (absorbing).
"""

from ciccada_config import AS4777

# ---------------------------------------------------------------------------
# Pull set-points once
# ---------------------------------------------------------------------------
_VV = AS4777["VVAR"]      # V1..V4, Q1, Q4
_VW = AS4777["VW"]        # V1, V2, P2
_TOL = AS4777["TOL_FRAC"] # 0.04


# ===========================================================================
# 1. VOLT-VAR REQUIRED-Q CURVE
# ===========================================================================
def vvar_required_q(v, s_rated,
                    v1=_VV["V1"], v2=_VV["V2"], v3=_VV["V3"], v4=_VV["V4"],
                    q1=_VV["Q1"], q4=_VV["Q4"]):
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


def vvar_required_q_sql(v_col, s_col,
                        v1=_VV["V1"], v2=_VV["V2"], v3=_VV["V3"], v4=_VV["V4"],
                        q1=_VV["Q1"], q4=_VV["Q4"]):
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
# 3. INVERTER CAPABILITY CURVE (max reactive absorption given active power)
# ===========================================================================
def q_cap_absorbing(p, s_rated):
    """
    Maximum reactive power the inverter CAN absorb given it is currently
    producing active power `p`, respecting the apparent-power circle of radius
    `s_rated`. 
    
    Returns a negative number (absorbing) or 0.

        |P| < 0.2*S        : 0            (too little P to require VAr support)
        0.2 <= |P| <= 0.6  : -0.44*S      (fixed pf floor region)
        0.6 <  |P| <= 0.8  : -sqrt((|P|/0.8)^2 - |P|^2)   (pf=0.8 arc)
        |P| > 0.8          : -sqrt(S^2 - |P|^2)           (full S-circle)
    """
    import math
    ap = abs(p)
    if ap < 0.2 * s_rated:
        return 0.0
    elif ap <= 0.6 * s_rated:
        return -0.44 * s_rated
    elif ap <= 0.8 * s_rated:
        val = (ap / 0.8) ** 2 - ap ** 2
        return -math.sqrt(val) if val > 0 else 0.0
    else:
        val = s_rated ** 2 - ap ** 2
        return -math.sqrt(val) if val > 0 else 0.0


def q_cap_absorbing_sql(p_col, s_col):
    """
    SQL CASE expression for the capability curve.
    Pass `s_col = 's_99'` for the R10-correct behaviour (or 'ac_capacity_kw'
    to reproduce Hossein's original for comparison).
    """
    return f"""
        CASE
            WHEN abs({p_col}) <  0.2 * {s_col} THEN 0
            WHEN abs({p_col}) <= 0.6 * {s_col} THEN -0.44 * {s_col}
            WHEN abs({p_col}) <= 0.8 * {s_col} THEN
                CASE WHEN (power(abs({p_col})/0.8, 2) - power(abs({p_col}), 2)) < 0 THEN 0
                     ELSE -sqrt(power(abs({p_col})/0.8, 2) - power(abs({p_col}), 2)) END
            ELSE
                CASE WHEN (power({s_col}, 2) - power(abs({p_col}), 2)) < 0 THEN 0
                     ELSE -sqrt(power({s_col}, 2) - power(abs({p_col}), 2)) END
        END""".strip()


# ===========================================================================
# 4. TOLERANCE HELPERS  (±4% of nameplate, additive in kW)
# ===========================================================================
def add_tol_kw(value, ac_capacity_kw, tol_frac=_TOL, sign=+1):
    """Add (sign=+1) or subtract (sign=-1) the 4%-of-nameplate tolerance, in kW."""
    return value + sign * tol_frac * ac_capacity_kw


def tol_kw_sql(ac_col, tol_frac=_TOL):
    """Return the SQL fragment for the tolerance magnitude in kW (e.g. '0.04 * ac_capacity_kw')."""
    return f"{tol_frac} * {ac_col}"


# ===========================================================================
# 5. Quick self-test when run directly:  `python as4777_curves.py`
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
