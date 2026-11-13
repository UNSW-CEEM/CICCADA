"""
CICCADA plot palette: the single source for every colour used in a CICCADA figure.

Rule: one colour has one meaning across the project.
  - Measured quantities have fixed colours (V, P, Q, S, PV potential).
  - Requirements from AS/NZS 4777.2 are never coloured: black, with line style.
  - Conformance categories use one ordered scale (reds = too little response,
    blues = conforming, purple = too much).

Change a colour here and every figure follows. Nothing else in standard_plots
should contain a hex code.

Colour sources
  Okabe & Ito, "Color Universal Design" (colour-blind safe qualitative set).
  ColorBrewer RdBu / PRGn for the ordered conformance scale.
  Phases follow the Australian red / white / blue convention (white drawn as
  light grey so it shows on a white background).
"""
from __future__ import annotations

import matplotlib as mpl

# =============================================================================
# 1. Measured quantities
# =============================================================================
VOLTAGE_OKABE_ITO = "#D55E00"   # vermilion (current standard)
VOLTAGE_LEGACY = "#b45309"      # brown used in the pre-2026-11 plot modules
VOLTAGE = VOLTAGE_OKABE_ITO     # <- swap here to restyle every voltage line

QUANTITY = {
    "V":         VOLTAGE,
    "P":         "#009E73",     # active power  (Okabe-Ito bluish green)
    "Q":         "#0072B2",     # reactive power (Okabe-Ito blue)
    "S":         "#000000",     # apparent power, rating lines
    "potential": "#E69F00",     # PV potential / counterfactual / curtailed energy (Okabe-Ito orange)
}

# =============================================================================
# 2. Requirements, references and background
# =============================================================================
REFERENCE = {
    "required":   "#000000",    # AS/NZS 4777.2 curve, Volt-Watt ceiling, required slope (dashed)
    "band":       "#000000",    # tolerance / permitted band (drawn at BAND_ALPHA)
    "threshold":  "#7F7F7F",    # set-point voltages, gates (dotted)
    "zero":       "#000000",
    "night":      "#F2F2F2",
    "grid":       "#EBEBEB",
    "text":       "#555555",    # secondary text (metadata lines)
    "fit":        "#CC79A7",    # fitted lines (e.g. Q-V slope); Okabe-Ito reddish purple
    "hatch":      "#C8C8C8",    # hatching on blank / too-few-data cells
    "text_dark":  "#1F2937",    # dark text on light cells
}
BAND_ALPHA = 0.12

# =============================================================================
# 3. Phases (3-phase sites)
# =============================================================================
# Red / white / blue convention. Thin per-phase lines; the thick line on top is
# always the quantity colour (V vermilion, P green, Q blue), so line weight -- not
# hue alone -- separates L1 from V and L3 from Q.
PHASE = {
    "L1": "#C00000",            # red
    "L2": "#B0B0B0",            # "white", drawn light grey
    "L3": "#1F3A93",            # dark blue
}

# =============================================================================
# 4. Data source (when a panel already fixes the quantity)
# =============================================================================
SOURCE = {
    "pv_circuit": {"color": "#0072B2", "ls": "-",  "marker": "o", "fill": True},
    "net_meter":  {"color": "#E69F00", "ls": "--", "marker": "o", "fill": False},
    "sa_original": {"color": "#999999", "ls": ":", "marker": "s", "fill": True},
}

# =============================================================================
# 5. Conformance outcomes
# =============================================================================
STATUS = {
    "conformant":      "#2166AC",
    "nonconformant":   "#B2182B",
    "not_assessable":  "#999999",   # no data / below Figure 2.1 gate: draw hatched
    "not_exposed":     "#D9D9D9",   # inside the deadband / below the trigger voltage
}

# Volt-VAr Q_impact categories, worst -> best -> not judged.
# The Ausgrid slope classes map onto the same scale.
VOLTVAR_CATEGORY = {
    "Q_adverse":               "#67001F",
    "Q_inactive":              "#B2182B",   # = Ausgrid "flat"
    "Q_significant_shortfall": "#EF8A62",   # = Ausgrid "shallow"
    "Q_near_conformant":       "#92C5DE",
    "Q_conformant":            "#2166AC",   # = Ausgrid "conformant"
    "Q_major_surplus":         "#762A83",   # = Ausgrid "steep"
    "Q_not_exposed":           "#D9D9D9",
    "Q_not_assessable":        "#999999",
}
SLOPE_CLASS = {
    "flat":       VOLTVAR_CATEGORY["Q_inactive"],
    "shallow":    VOLTVAR_CATEGORY["Q_significant_shortfall"],
    "conformant": VOLTVAR_CATEGORY["Q_conformant"],
    "steep":      VOLTVAR_CATEGORY["Q_major_surplus"],
}
VOLTWATT_CATEGORY = {
    "conformant":        STATUS["conformant"],
    "nonconformant":     STATUS["nonconformant"],
    "curtailed":         QUANTITY["potential"],
    "sun_limited":       "#F0E442",
    "not_exposed":       STATUS["not_exposed"],
    "no_counterfactual": STATUS["not_assessable"],
}
JOINT = {
    "both conformant": "#2166AC",
    "Volt-VAr only":   "#92C5DE",
    "Volt-Watt only":  "#D1E5F0",
    "neither":         "#B2182B",
}

# =============================================================================
# 6. Continuous scales and groups
# =============================================================================
CMAP = {
    "rate":   "Blues",      # % conformant (heatmaps, confusion matrices)
    "count":  "cividis",    # densities, hexbin
    "year":   "viridis",    # install year and other ordered years
    "hour":   "twilight",   # hour of day (cyclic)
}
GROUP_CYCLE = "petroff10"   # matplotlib >= 3.10 colour cycle for DNSP / OEM / cohort
OTHER = "#999999"           # "Other" / "Unknown" in any grouping

# Curtailment mechanisms (family F)
# (as oem_analysis/notebooks/99_publication_figures.ipynb, Figure 4)
MECHANISM = {
    "voltvar":  "#7C3AED",   # Volt-VAr-induced curtailment (violet)
    "voltwatt": "#4709B2",   # Volt-Watt curtailment (indigo)
}


# =============================================================================
# 7. Global style
# =============================================================================
def use_style(context: str = "notebook") -> None:
    """Apply the CICCADA rcParams. context: 'notebook', 'paper' (9 pt) or 'slides' (11 pt)."""
    size = {"notebook": 9.5, "paper": 9.0, "slides": 11.0}[context]
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": size,
        "axes.titlesize": size + 1,
        "axes.labelsize": size,
        "legend.fontsize": size - 1.5,
        "xtick.labelsize": size - 1,
        "ytick.labelsize": size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": REFERENCE["grid"],
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
