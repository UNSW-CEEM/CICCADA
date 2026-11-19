# html visualisations

Interactive pages that explain how AS/NZS 4777.2:2020 (Australia A) works, built from CICCADA data.

## AS/NZS 4777.2 response explorer

Shows Volt-VAr, Volt-Watt and the Figure 2.1 minimum reactive power capability together, for one operating
point at a time: reactive power against voltage, active power against voltage, the power circle, a site day and
the site metadata.

| File | What it is |
|---|---|
| `as4777_explorer_internal.html` | Built page with real sites and synthetic sites. **Internal use only.** |
| `as4777_explorer_synthetic.html` | Built page with synthetic (invented) data only. Safe to share. |
| `build_visualisation.py` | Builds both pages. Edit `SOURCES` at the top to change the sites and days. |
| `template.html` | The page itself (layout, drawing, rules in JavaScript). |

Rebuild (conda env `ciccada`, from the repo root):

    python "html visualisations/build_visualisation.py"
    python "html visualisations/build_visualisation.py" --synthetic-only   # no data access needed

Nothing is duplicated from the rest of the repo:

- data: `standard_plots/lib/sources.py` (`load_local`, `load_aws`, `load_synthetic`, `load_month`)
- units and metadata lines: `standard_plots/lib/a_site_day.py`
- set-points: `bms_sa_review/shared/ciccada_config.py`
- colours: `standard_plots/lib/palette.py`
- rules: re-computed in the page, and checked on load against a table of answers from
  `bms_sa_review/shared/as4777_curves.py` (capability profile `review_corrected`). The footer reports the result.

Expected response used by the page, for a voltage and an available active power:

1. Volt-Watt limit at that voltage (Table 3.6); starting P = the lesser of available P and that limit.
2. Volt-VAr curve Q at that voltage (Table 3.7).
3. Figure 2.1 (Cl. 2.6) at the starting P: below 20% no minimum; 20-60% the lesser of the curve and 44%;
   60-80% the lesser of the curve and 0.75 x P; above 80% the curve, with P reduced to sqrt(100^2 - Q^2)
   where needed (Cl. 3.3.2.3).

In the three top charts the permitted band uses the expected active power. In the day plot it uses each
interval's measured active power, exactly as the Stage 2 conformance tables do.
