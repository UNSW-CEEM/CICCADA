def build_ami_raw_phaseseparate(
    interval_table: pd.DataFrame, circuit_polarity: pd.DataFrame, *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    device_column: str = "device_id",
    type_column: str = "circuit_type",
    time_column: str = "t_stamp",
    power_column: str = "power",
    reactive_power_column: str = "reactive_power",
    voltage_column: str = "voltage",
    polarity_column: str = "circuit_polarity",
    load_type: str = "ac_load_net",
    pv_type: str = "pv_site_net",
) -> pd.DataFrame:
    """
    One row per (site_id, device_id, circuit_id, t_stamp), for EVERY kept
    `load_type`/`pv_type` circuit -- load AND PV circuits both get their
    own row, tagged by `circuit_type`. This is deliberately a thin,
    circuit-preserving view: `P_kw_signed`/`Q_kvar_signed` are that
    circuit's OWN reading, polarity-corrected (see module docstring for
    why this deliberately differs from `ami_meter.P_kw`/`Q_kvar`, which
    are raw/uncorrected). `V` is the raw per-circuit voltage (not
    polarity-corrected -- voltage has no sign-convention ambiguity).

    On a `load_type` row, `P_kw_signed` is `ac_load_net`'s own reading --
    already net-of-PV as landed (see `build_ami_raw`'s docstring), NOT a
    PV-independent gross load figure. On a `pv_type` row, `P_kw_signed` is
    that PV circuit's own generation reading, undivided. This table does
    NOT allocate or split PV across load phases, and does NOT compute a
    per-phase gross-load reconciliation -- earlier versions did both, but
    that baked a debatable heuristic (`equal_split_across_load_phases`,
    for the common case where PV/load circuit counts don't match) into a
    numeric column that looked like measured ground truth. If you need a
    PV-independent per-phase gross load, compute it explicitly yourself
    from these raw load/PV rows, making your own matching choice for
    mismatched-count sites; for the SITE-level equivalent (which uses a
    different, validated method, not this per-phase heuristic), use
    `ami_raw.P_kw`/`Q_kvar` directly.

    `n_phases_at_site` (from the LOAD side's circuit count) and
    `pv_allocation_method` (`direct_matched_circuit` when kept PV circuit
    count equals kept load circuit count, `equal_split_across_load_phases`
    when it doesn't, `no_pv_present` when there's no surviving PV circuit
    at all) are kept as lightweight, per-site descriptive tags -- decided
    once per site (same rule as before) and copied onto EVERY row for that
    site, load or PV alike, since they describe the site's circuit
    topology, not a load-specific derived quantity. They carry no
    allocated Watts with them any more.
    """
    columns = [
        site_column, device_column, circuit_column, type_column, time_column,
        "year", "month", "V", "P_kw_signed", "Q_kvar_signed",
        "n_phases_at_site", "pv_allocation_method",
    ]
    if interval_table is None or not len(interval_table):
        return pd.DataFrame(columns=columns)

    frame = interval_table.merge(circuit_polarity, on=circuit_column, how="left")
    frame["power_signed"] = frame[power_column] * frame[polarity_column]
    if reactive_power_column in frame.columns:
        frame["reactive_power_signed"] = frame[reactive_power_column] * frame[polarity_column]
    else:
        frame["reactive_power_signed"] = np.nan

    load_side = frame[frame[type_column] == load_type].copy()
    if not len(load_side):
        return pd.DataFrame(columns=columns)
    pv_side = frame[frame[type_column] == pv_type].copy()

    load_circuits_by_site = load_side.groupby(site_column)[circuit_column].unique()
    pv_circuits_by_site = (
        pv_side.groupby(site_column)[circuit_column].unique() if len(pv_side) else pd.Series(dtype=object)
    )

    site_plan_rows = []
    for site_id, load_ids in load_circuits_by_site.items():
        n_phases = len(sorted(load_ids))
        pv_ids_sorted = (
            sorted(pv_circuits_by_site.loc[site_id])
            if site_id in getattr(pv_circuits_by_site, "index", []) else []
        )
        if not pv_ids_sorted:
            method = "no_pv_present"
        elif len(pv_ids_sorted) == n_phases:
            method = "direct_matched_circuit"
        else:
            method = "equal_split_across_load_phases"
        site_plan_rows.append({
            site_column: site_id, "n_phases_at_site": n_phases, "pv_allocation_method": method,
        })
    site_plan = pd.DataFrame(site_plan_rows)

    rows = pd.concat([load_side, pv_side], ignore_index=True) if len(pv_side) else load_side.copy()
    rows["P_kw_signed"] = rows["power_signed"] / 1000.0
    rows["Q_kvar_signed"] = rows["reactive_power_signed"] / 1000.0
    if voltage_column in rows.columns:
        rows["V"] = rows[voltage_column]
    else:
        rows["V"] = np.nan
    rows = rows.merge(site_plan, on=site_column, how="left")

    rows["year"], rows["month"] = _year_month(rows[time_column])

    return (
        rows[columns]
        .sort_values([site_column, circuit_column, time_column])
        .reset_index(drop=True)
    )
