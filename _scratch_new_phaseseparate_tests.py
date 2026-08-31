# build_ami_raw_phaseseparate
# --------------------------------------------------------------------------- #

def test_phaseseparate_direct_matches_when_pv_count_equals_phase_count():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = pd.DataFrame([
        {"site_id": 1, "device_id": 100, "circuit_id": 10, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 500.0, "reactive_power": 40.0, "voltage": 241.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 20, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 300.0, "reactive_power": 20.0, "voltage": 242.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 11, "circuit_type": "pv_site_net",
         "t_stamp": t, "power": 900.0, "reactive_power": 10.0, "voltage": 243.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 21, "circuit_type": "pv_site_net",
         "t_stamp": t, "power": 700.0, "reactive_power": 5.0, "voltage": 244.0},
    ])
    circuit_polarity = pd.DataFrame({
        "circuit_id": [10, 20, 11, 21], "circuit_polarity": [1, 1, -1, -1],
    })
    result = Build.build_ami_raw_phaseseparate(interval_table, circuit_polarity)

    # every kept circuit -- load AND PV -- gets its own row now
    assert len(result) == 4
    assert set(result.circuit_id) == {10, 20, 11, 21}
    assert set(result.circuit_type) == {"ac_load_net", "pv_site_net"}
    # site-level tags are copied onto every row, load or PV
    assert (result.pv_allocation_method == "direct_matched_circuit").all()
    assert set(result.n_phases_at_site) == {2}

    row10 = result.set_index("circuit_id").loc[10]
    assert row10.P_kw_signed == pytest.approx(0.5)
    assert row10.Q_kvar_signed == pytest.approx(0.04)
    assert row10.V == 241.0

    row11 = result.set_index("circuit_id").loc[11]   # PV circuit's own undivided reading
    assert row11.P_kw_signed == pytest.approx(-0.9)
    assert row11.Q_kvar_signed == pytest.approx(-0.01)
    assert row11.circuit_type == "pv_site_net"
    assert row11.pv_allocation_method == "direct_matched_circuit"
    assert row11.n_phases_at_site == 2


def test_phaseseparate_equal_splits_when_counts_dont_match():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = pd.DataFrame([
        {"site_id": 1, "device_id": 100, "circuit_id": 10, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 500.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 20, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 300.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 30, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 100.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 11, "circuit_type": "pv_site_net",
         "t_stamp": t, "power": 3000.0},
    ])
    circuit_polarity = pd.DataFrame({
        "circuit_id": [10, 20, 30, 11], "circuit_polarity": [1, 1, 1, -1],
    })
    result = Build.build_ami_raw_phaseseparate(interval_table, circuit_polarity)

    assert len(result) == 4   # 3 load rows + 1 PV row, no split/allocation math
    assert (result.pv_allocation_method == "equal_split_across_load_phases").all()
    assert set(result.n_phases_at_site) == {3}

    pv_row = result.set_index("circuit_id").loc[11]
    assert pv_row.circuit_type == "pv_site_net"
    assert pv_row.P_kw_signed == pytest.approx(-3.0)   # undivided -- no per-phase split stored


def test_phaseseparate_no_pv_present_gives_no_pv_rows():
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = _interval_table([(1, 100, 10, "ac_load_net", t, 500.0)])
    circuit_polarity = pd.DataFrame({"circuit_id": [10], "circuit_polarity": [1]})
    result = Build.build_ami_raw_phaseseparate(interval_table, circuit_polarity)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.pv_allocation_method == "no_pv_present"
    assert row.circuit_type == "ac_load_net"
    assert row.P_kw_signed == pytest.approx(0.5)


def test_phaseseparate_load_and_pv_rows_reconcile_with_ami_raw_p_kw():
    # sum(P_kw_signed) across a site's load rows PLUS its PV rows should
    # equal ami_raw.P_kw/Q_kvar for that site/timestamp -- computed
    # explicitly here (this table no longer stores that reconciliation as
    # a column, see the function's docstring).
    t = pd.Timestamp("2025-06-01 12:00", tz="UTC")
    interval_table = pd.DataFrame([
        {"site_id": 1, "device_id": 100, "circuit_id": 10, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 500.0, "reactive_power": 30.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 20, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": -300.0, "reactive_power": 15.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 30, "circuit_type": "ac_load_net",
         "t_stamp": t, "power": 100.0, "reactive_power": 5.0},
        {"site_id": 1, "device_id": 100, "circuit_id": 11, "circuit_type": "pv_site_net",
         "t_stamp": t, "power": 1500.0, "reactive_power": 25.0},
    ])
    circuit_polarity = pd.DataFrame({
        "circuit_id": [10, 20, 30, 11], "circuit_polarity": [1, 1, 1, -1],
    })
    phase_split = Build.build_ami_raw_phaseseparate(interval_table, circuit_polarity)
    ami_raw = Build.build_ami_raw(interval_table, circuit_polarity)

    assert phase_split.P_kw_signed.sum() == pytest.approx(ami_raw.iloc[0].P_kw)
    assert phase_split.Q_kvar_signed.sum() == pytest.approx(ami_raw.iloc[0].Q_kvar)


def test_phaseseparate_empty_interval_table():
    result = Build.build_ami_raw_phaseseparate(pd.DataFrame(), pd.DataFrame())
    assert len(result) == 0
    assert "pv_allocation_method" in result.columns
    assert "circuit_type" in result.columns
