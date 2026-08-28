def build_ami_raw(
    interval_table: pd.DataFrame, circuit_polarity: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    type_column: str = "circuit_type",
    time_column: str = "t_stamp",
    power_column: str = "power",
    reactive_power_column: str = "reactive_power",
    voltage_column: str = "voltage",
    pv_type: str = "pv_site_net",
    load_type: str = "ac_load_net",
    site_capacity: pd.DataFrame | None = None,
    normalization_basis: str = "s_99",
) -> pd.DataFrame:
    """
    One row per (site_id, t_stamp) -- house-load and PV-system columns kept
    SEPARATE, at site level, in two symmetric trios:

    - Load side: `P_load_kw`, `Q_load_kvar`, `V_load`.
    - PV side:   `P_pv_kw`,   `Q_pv_kvar`,   `V_pv`.

    All four power/reactive-power columns are `circuit_polarity`-corrected
    and SIGNED -- `P_load_kw`/`Q_load_kvar` are `ac_load_net`, summed across
    the site's load phases; `P_pv_kw`/`Q_pv_kvar` are `pv_site_net`, summed
    across the site's PV circuits. Both reuse
    `ami_signal.reconstruct_gross_load`'s own `load_signed`/`pv_signed`
    output -- the same validated per-side sums the `gross_load`
    reconstruction is built from, just exposed directly instead of only
    being visible pre-summed.

    IMPORTANT sign-convention note (fixed here): earlier versions of this
    function summed PV power UNSIGNED (raw `power`, no `circuit_polarity`
    applied), on the theory that Phase 3 Section 9 already confirmed PV
    needs no correction. That reading of Section 9 was wrong: Section 9
    confirmed `pv_site_net` is a generation-only signal (no LOAD netted
    into it) -- it says nothing about SIGN. Raw `power` for a PV circuit
    reads NEGATIVE while generating (confirmed against real fleet data),
    so summing it unsigned made `P_pv_kw`/`P_kw_norm` come out negative
    during generation -- inconsistent with `Q_kvar_norm` and `gross_load`,
    which already applied `circuit_polarity` correctly. `P_pv_kw` (and
    therefore `P_kw_norm`, which divides it by capacity) now applies the
    same `circuit_polarity` correction as everything else in this table,
    so it reads POSITIVE while the site is generating, matching
    `Q_pv_kvar`. If you compared `P_kw_norm` against an earlier build,
    expect it to have flipped sign.

    `P_kw`/`Q_kvar` (`gross_load`/`gross_reactive_load`) are kept as
    convenience columns -- `P_kw = P_load_kw + P_pv_kw` by construction
    (the "what would this site have consumed with no solar" counterfactual
    demand), not a third independent quantity.

    `P_kw_norm`/`Q_kvar_norm` are `P_pv_kw`/`Q_pv_kvar` divided by site
    capacity -- need `site_capacity` (a frame with `site_id`, `S_99`,
    `ac_capacity_kw`, one row per site, already deduplicated by the
    caller); without it, those two columns (and `S_99`/`ac_capacity_kw`
    themselves) are present but entirely null, so the column shape stays
    stable whether or not capacity metadata was supplied.
    `normalization_basis` selects which of `S_99`/`ac_capacity_kw` is the
    denominator; a site with a zero or missing value there gets a null
    normalized value, not a divide-by-zero inf.

    Reactive-power columns (`Q_load_kvar`, `Q_pv_kvar`, `Q_kvar_norm`,
    `Q_kvar`) carry a validation caveat active power doesn't: only the
    active-power reconstruction has been checked against real plots
    (Phase 4). Treat reactive power as reasonable-by-construction, not
    independently confirmed.

    `V_load` is the site's load-side voltage, averaged across kept
    `ac_load_net` circuits at that timestamp (matching `structured_data`'s
    own aggregation convention, just applied to the load side instead of
    the PV side it uses). `V_pv` is the analogous average across the
    site's kept `pv_site_net` circuits -- the inverter/PV-side voltage,
    which can genuinely differ from `V_load` (local voltage rise from the
    site's own export).

    Inner-joined on (site_id, t_stamp): a timestamp where only one side
    reported contributes no row.
    """
    columns = [
        site_column, time_column, "year", "month",
        "V_load", "V_pv",
        "P_load_kw", "P_pv_kw", "P_kw_norm", "P_kw",
        "Q_load_kvar", "Q_pv_kvar", "Q_kvar_norm", "Q_kvar",
        "S_99", "ac_capacity_kw", "normalization_basis",
    ]
    if interval_table is None or not len(interval_table):
        return pd.DataFrame(columns=columns)

    reconstructed_p = Signal.reconstruct_gross_load(
        interval_table, circuit_polarity,
        site_column=site_column, circuit_column=circuit_column,
        type_column=type_column, time_column=time_column, power_column=power_column,
    )
    p_frame = reconstructed_p[
        [site_column, time_column, "load_signed", "pv_signed", "reconstructed_load"]
    ].rename(columns={
        "load_signed": "load_w", "pv_signed": "pv_generation_w",
        "reconstructed_load": "gross_load_w",
    })
    merged = p_frame.dropna(subset=["load_w", "pv_generation_w"])
    if not len(merged):
        return pd.DataFrame(columns=columns)

    if reactive_power_column in interval_table.columns:
        reconstructed_q = Signal.reconstruct_gross_load(
            interval_table, circuit_polarity,
            site_column=site_column, circuit_column=circuit_column,
            type_column=type_column, time_column=time_column, power_column=reactive_power_column,
        )
        q_frame = reconstructed_q[
            [site_column, time_column, "load_signed", "pv_signed", "reconstructed_load"]
        ].rename(columns={
            "load_signed": "load_var", "pv_signed": "pv_reactive_generation_var",
            "reconstructed_load": "gross_reactive_load_var",
        })
    else:
        q_frame = pd.DataFrame(columns=[
            site_column, time_column, "load_var", "pv_reactive_generation_var",
            "gross_reactive_load_var",
        ])
    merged = merged.merge(q_frame, on=[site_column, time_column], how="left")

    load_side = interval_table[interval_table[type_column] == load_type]
    if len(load_side) and voltage_column in load_side.columns:
        v_load_agg = (
            load_side.groupby([site_column, time_column])[voltage_column]
            .mean()
            .reset_index()
            .rename(columns={voltage_column: "V_load"})
        )
        merged = merged.merge(v_load_agg, on=[site_column, time_column], how="left")
    else:
        merged["V_load"] = np.nan

    pv_side = interval_table[interval_table[type_column] == pv_type]
    if len(pv_side) and voltage_column in pv_side.columns:
        v_pv_agg = (
            pv_side.groupby([site_column, time_column])[voltage_column]
            .mean()
            .reset_index()
            .rename(columns={voltage_column: "V_pv"})
        )
        merged = merged.merge(v_pv_agg, on=[site_column, time_column], how="left")
    else:
        merged["V_pv"] = np.nan

    merged["P_load_kw"] = merged["load_w"] / 1000.0
    merged["P_pv_kw"] = merged["pv_generation_w"] / 1000.0
    merged["P_kw"] = merged["gross_load_w"] / 1000.0
    merged["Q_load_kvar"] = merged["load_var"] / 1000.0
    merged["Q_pv_kvar"] = merged["pv_reactive_generation_var"] / 1000.0
    merged["Q_kvar"] = merged["gross_reactive_load_var"] / 1000.0
    merged["year"], merged["month"] = _year_month(merged[time_column])

    if site_capacity is not None and len(site_capacity):
        merged = merged.merge(
            site_capacity[[site_column, "S_99", "ac_capacity_kw"]],
            on=site_column, how="left",
        )
    else:
        merged["S_99"] = np.nan
        merged["ac_capacity_kw"] = np.nan

    normalization_col = "S_99" if normalization_basis == "s_99" else "ac_capacity_kw"
    denominator = merged[normalization_col].where(merged[normalization_col] > 0)
    merged["P_kw_norm"] = merged["P_pv_kw"] / denominator
    merged["Q_kvar_norm"] = merged["Q_pv_kvar"] / denominator
    merged["normalization_basis"] = normalization_basis

    return merged[columns].sort_values([site_column, time_column]).reset_index(drop=True)
