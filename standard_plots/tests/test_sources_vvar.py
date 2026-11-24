"""
standard_plots data sources use the shared Volt-VAr rules: the DuckDB scans
(category_counts, find_examples) must give the same counts as
bms_sa_review/shared/vvar_metrics on the same intervals.

Runs on a small invented local store written to a temp folder (no real data).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "standard_plots"), str(REPO / "bms_sa_review"), str(REPO)]

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from shared.vvar_metrics import (  # noqa: E402
    conformance_metrics,
    score_intervals,
    zone_counts,
)


def _fake_store(tmp_path, tol_case_sites=6, days=4, seed=7):
    """Raw PV-circuit rows for a few invented 1P and 3P sites, Jan 2025."""
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    t_aest = pd.date_range("2025-01-01", periods=288 * days, freq="5min")
    for k in range(tol_case_sites):
        site = 9000 + k
        n_pv = 3 if k % 3 == 0 else 1
        S = 5.0 * n_pv
        hour = t_aest.hour + t_aest.minute / 60
        sun = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
        v = 236 + 10 * sun + rng.normal(0, 3, len(t_aest)) - 20 * (k == 5) * (hour < 8)
        p_tot = S * sun * rng.uniform(0.6, 1.0)
        gain = [0.0, -1.0, 1.0, -0.5, -1.2, 0.3][k]  # response strength / sign
        q_tot = gain * np.clip((v - 240) / 18, 0, 1) * 0.6 * S + rng.normal(
            0, 0.03 * S, len(v)
        )
        for c in range(n_pv):
            vc = v + rng.normal(0, 0.5, len(v))
            pc, qc = p_tot / n_pv, q_tot / n_pv
            df = pd.DataFrame(
                {
                    "site_id": site,
                    "circuit_id": site * 10 + c,
                    "circuit_type": "pv_site_net",
                    "t_stamp": t_aest - pd.Timedelta(hours=10),
                    "V": vc,
                    "P_kw_signed": pc,
                    "Q_kvar_signed": qc,
                }
            )
            df.loc[rng.random(len(df)) < 0.01, "Q_kvar_signed"] = np.nan  # gaps
            rows.append(df)
        meta.append(
            {
                "site_id": site,
                "ac_capacity_kw": S,
                "state": "NSW",
                "dnsp_name": "Network A",
                "manufacturer": "Maker A",
                "pv_install_date": pd.Timestamp("2023-01-01"),
            }
        )
    raw = pd.concat(rows, ignore_index=True)
    store = tmp_path / "ami_store"
    for m, part in raw.groupby(raw.t_stamp.dt.strftime("%Y-%m")):
        d = store / "ami_raw_phaseseparate" / f"dt_month={m}"
        d.mkdir(parents=True)
        part.to_parquet(d / "part0.parquet", index=False)
    pd.DataFrame(meta).to_parquet(store / "ami_site_metadata.parquet", index=False)
    return store, raw, pd.DataFrame(meta)


def _python_counts(raw, meta, tol):
    """Same site aggregation as the scans, scored with shared.vvar_metrics."""
    r = raw.copy()
    r["t_aest"] = r.t_stamp + pd.Timedelta(hours=10)
    r = r[(r.t_aest >= "2025-01-01") & (r.t_aest < "2025-02-01")]
    r["ok"] = r.V.gt(0) & r.V.lt(300) & r.P_kw_signed.notna() & r.Q_kvar_signed.notna()
    n_pv = r.groupby("site_id").circuit_id.nunique().rename("n_pv")
    g = (
        r[r.ok]
        .groupby(["site_id", "t_aest"])
        .agg(
            V=("V", "mean"),
            P_kW=("P_kw_signed", "sum"),
            Q_kvar=("Q_kvar_signed", "sum"),
            n_ok=("V", "size"),
        )
        .reset_index()
        .merge(n_pv, on="site_id")
        .merge(
            meta[["site_id", "ac_capacity_kw"]].rename(
                columns={"ac_capacity_kw": "s_rated"}
            ),
            on="site_id",
        )
    )
    g = g[g.n_ok == g.n_pv]  # complete timestamps only
    g["day"] = g.t_aest.dt.date
    return zone_counts(score_intervals(g, tol=tol), by=["site_id"])


@pytest.mark.parametrize("tol", [0.0, 0.04])
def test_category_counts_match_shared_python(tmp_path, monkeypatch, tol):
    store, raw, meta = _fake_store(tmp_path)
    monkeypatch.setenv("CICCADA_AMI_STORE_DIR", str(store))
    from lib import sources

    sql = sources.category_counts("2025-01", tol=tol, temp_dir=str(tmp_path / "tmp"))
    py = _python_counts(raw, meta, tol)
    sql = sql.set_index("site_id").sort_index()
    py = py.set_index("site_id").sort_index()
    cols = [
        c
        for c in py.columns
        if c.startswith("n_") and c in sql.columns and not c.startswith("n_missing")
    ]
    cols = [c for c in cols if c not in ("n_rows",)]
    assert "n_A_adverse" in cols and "n_legacy_fail" in cols and "n_days_A" in cols
    pd.testing.assert_frame_equal(
        sql[cols].astype("int64"), py[cols].astype("int64"), check_names=False
    )
    # metrics run directly on the scan output
    m = conformance_metrics(sql.reset_index())
    assert {
        "nonconformance_frequency_verdict",
        "remote_conformance_rate_verdict",
        "legacy_conformance_rate_verdict",
        "voltage_exposure",
    } <= set(m.columns)


def test_find_examples_uses_zone_rules(tmp_path, monkeypatch):
    store, raw, meta = _fake_store(tmp_path)
    monkeypatch.setenv("CICCADA_AMI_STORE_DIR", str(store))
    from lib import sources

    days = sources.find_examples(
        "2025-01", all_days=True, temp_dir=str(tmp_path / "tmp")
    )
    py = _python_counts(raw, meta, 0.04)
    tot = days.groupby("site_id")[["vv_n", "vv_fail_A", "vv_fail_C"]].sum().sort_index()
    m = conformance_metrics(py).set_index("site_id").sort_index()
    assert (tot.vv_n.values == m.n_A.values).all()
    assert (tot.vv_fail_A.values == m.fail_A.values).all()
    assert (tot.vv_fail_C.values == m.fail_C.values).all()


def test_synthetic_counts_have_the_shared_columns():
    from lib import sources

    d, _ = sources.load_synthetic_category_counts()
    m = conformance_metrics(d)
    assert (
        m["remote_conformance_rate_verdict"]
        .isin(["conformant", "non-conformant", "not assessable"])
        .all()
    )
