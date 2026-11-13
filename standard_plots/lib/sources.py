"""
Data sources for the standard plots. Every loader returns the same thing:

    df    tidy frame for one AEST day: t_aest, phase, V, P_kw, Q_kvar (+ P_pot_kw if known)
    info  dict with rating_kva, label, date, source, and `site` (metadata, see SITE_FIELDS)

so any plot function works with any source.

    load_synthetic(example)        CSV in standard_plots/ (no access needed)
    load_local(site_id, date)      local Solar Analytics store (%LOCALAPPDATA%/ciccada/ami_store)
    load_aws(site_id, date)        Athena, solar_analytics_iceberg (needs `aws sso login --profile ciccada`)
    list_days(site_id, month, source)   rank a month's days for plotting (local or aws)

  Several days of one site (family B), same tidy frame:
    load_synthetic_month(example)  CSV in standard_plots/
    load_month(site_id, month, source)   'local' or 'aws'

  Many sites at once (family B fleet view):
    load_synthetic_fleet()         CSV in standard_plots/
    qv_profiles(month)             per-site binned median Q vs V from the local store

Sign conventions: P generation positive, Q generator convention (+ supplying,
- absorbing). Both real-data loaders apply circuit_polarity, as
data_query/lib/site_selection.pull_site_telemetry does. PV circuits are named
L1, L2, L3 in circuit_id order.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]           # standard_plots/
_REPO = ROOT.parent
for _p in (_REPO, _REPO / "bms_sa_review"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SYNTHETIC_CSV = ROOT / "synthetic_site_day.csv"
SYNTHETIC_MONTH_CSV = ROOT / "synthetic_site_month.csv"
SYNTHETIC_FLEET_CSV = ROOT / "synthetic_fleet_qv.csv"

#: Site-level metadata carried into info["site"] (Solar Analytics meta_up23c fields;
#: the local store's ami_site_metadata.parquet has the same names).
SITE_FIELDS = ("state", "postcode", "dnsp_name", "manufacturer", "model", "inverter_count",
               "ac_capacity_kw", "dc_capacity_kw", "export_limit_kw", "pv_install_date",
               "monitoring_start", "flex_export_detected")
#: Per-PV-circuit fields, summarised into info["site"] (unique values joined).
CIRCUIT_FIELDS = ("device_type", "voltage_class", "circuit_polarity")
COLUMNS = ["t_aest", "phase", "V", "P_kw", "Q_kvar", "P_pot_kw"]


def _day_bounds_utc(date):
    """AEST calendar day -> [start, end) in UTC (fixed +10 h, no DST)."""
    day = pd.Timestamp(date).normalize()
    return day - pd.Timedelta(hours=10), day + pd.Timedelta(hours=14)


def _label_phases(df: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(df["circuit_id"].unique())
    df["phase"] = df["circuit_id"].map({c: f"L{i + 1}" for i, c in enumerate(ids)})
    return df


# =============================================================================
# 1. Synthetic
# =============================================================================
def load_synthetic(example: str = "1P"):
    """'1P' (5 kVA single-phase) or '3P' (15 kVA three-phase) from the CSV, building it if absent."""
    if not SYNTHETIC_CSV.exists():
        make_synthetic()
    d = pd.read_csv(SYNTHETIC_CSV, parse_dates=["t_aest"])
    d = d[d["example"] == example]
    if d.empty:
        raise ValueError(f"No synthetic example {example!r}; use '1P' or '3P'")
    info = {"rating_kva": float(d["rating_kva"].iloc[0]), "label": f"Synthetic site ({example})",
            "date": str(d["t_aest"].dt.date.iloc[0]), "source": "synthetic (invented data)",
            "site": {"manufacturer": "Synthetic", "model": example, "ac_capacity_kw": float(d["rating_kva"].iloc[0]),
                     "n_pv_circuits": d["phase"].nunique()}}
    return d[COLUMNS].reset_index(drop=True), info


def make_synthetic(path: Path = SYNTHETIC_CSV, seed: int = 4777) -> Path:
    """
    Write two invented site-days that exercise every feature of family A:
    low-power morning and evening (not assessable), voltage above 253 V at midday
    (Volt-Watt active), a partial Volt-VAr response, short spells above the
    Volt-Watt ceiling, and (3P) one phase that does not respond. Not real data.
    """
    from shared.as4777_curves import vvar_required_q, vw_max_p

    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-21 00:00", periods=288, freq="5min")
    h = t.hour + t.minute / 60
    shape = np.clip(np.sin(np.pi * (h - 6.0) / 13.0), 0, None) ** 1.3          # 06:00-19:00
    cloud = 1 - 0.35 * ((h > 15.2) & (h < 15.8))

    def site(rating, v_offsets, p_share, q_gain, example):
        n = len(v_offsets)
        v_site = 234.0 + 22.0 * shape                                          # site voltage shape
        pot_total = 0.97 * rating * shape * cloud
        breach = 1 + 0.09 * ((h > 12.4) & (h < 12.9))                          # brief Volt-Watt breach
        ceil_total = np.array([vw_max_p(x, rating) for x in v_site])
        p_total = np.minimum(pot_total, ceil_total * breach)
        rows = []
        for k, (dv, share, gain) in enumerate(zip(v_offsets, p_share, q_gain)):
            v = v_site + dv + rng.normal(0, 0.5, len(t))
            p = np.clip(p_total * share + rng.normal(0, 0.006 * rating / n, len(t)) * (shape > 0), 0, None)
            q_req = np.array([vvar_required_q(x, rating / n) for x in v])
            q = gain * q_req + rng.normal(0, 0.008 * rating / n, len(t))
            rows.append(pd.DataFrame({"example": example, "rating_kva": rating, "t_aest": t,
                                      "phase": f"L{k + 1}", "V": v.round(2), "P_kw": p.round(4),
                                      "Q_kvar": q.round(4), "P_pot_kw": pot_total.round(4)}))
        return pd.concat(rows)

    df = pd.concat([
        site(5.0, [0.0], [1.0], [0.75], "1P"),
        site(15.0, [0.0, 2.0, -1.5], [0.36, 0.33, 0.31], [0.9, 0.8, 0.05], "3P"),
    ], ignore_index=True)
    df.to_csv(path, index=False)
    return path


def _synthetic_info(d: pd.DataFrame, example: str, date_label: str) -> dict:
    rating = float(d["rating_kva"].iloc[0])
    return {"rating_kva": rating, "label": f"Synthetic site ({example})", "date": date_label,
            "source": "synthetic (invented data)",
            "site": {"manufacturer": "Synthetic", "model": example, "ac_capacity_kw": rating,
                     "n_pv_circuits": d["phase"].nunique()}}


def load_synthetic_month(example: str = "1P"):
    """14 invented days for '1P' or '3P' (same sites as load_synthetic), building the CSV if absent."""
    if not SYNTHETIC_MONTH_CSV.exists():
        make_synthetic_month()
    d = pd.read_csv(SYNTHETIC_MONTH_CSV, parse_dates=["t_aest"])
    d = d[d["example"] == example]
    if d.empty:
        raise ValueError(f"No synthetic example {example!r}; use '1P' or '3P'")
    return d[COLUMNS].reset_index(drop=True), _synthetic_info(d, example, "2025-01 (14 days)")


def make_synthetic_month(path: Path = SYNTHETIC_MONTH_CSV, seed: int = 4778, n_days: int = 14) -> Path:
    """
    Invented fortnight for the two synthetic sites: each day has its own voltage
    level, cloud cover and noise, so the Q-V cloud spreads over 225-260 V.
    1P responds at 75% of the curve; in 3P, L3 does not respond. Not real data.
    """
    from shared.as4777_curves import vvar_required_q

    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        t = pd.date_range(pd.Timestamp("2025-01-06") + pd.Timedelta(days=day), periods=288, freq="5min")
        h = t.hour + t.minute / 60
        shape = np.clip(np.sin(np.pi * (h - 6.0) / 13.0), 0, None) ** 1.3
        cloud = np.clip(1 - rng.uniform(0, 0.5) * (rng.random(len(t)) < 0.3), 0.3, 1)
        v_base, v_rise = rng.uniform(228, 238), rng.uniform(10, 24)
        for example, rating, offs, shares, gains in [
            ("1P", 5.0, [0.0], [1.0], [0.75]),
            ("3P", 15.0, [0.0, 2.0, -1.5], [0.36, 0.33, 0.31], [0.9, 0.8, 0.05]),
        ]:
            n = len(offs)
            p_total = 0.95 * rating * shape * cloud
            for k, (dv, share, gain) in enumerate(zip(offs, shares, gains)):
                v = v_base + dv + v_rise * shape * cloud + rng.normal(0, 0.8, len(t))
                p = np.clip(p_total * share + rng.normal(0, 0.006 * rating / n, len(t)) * (shape > 0), 0, None)
                q = gain * np.array([vvar_required_q(x, rating / n) for x in v]) + rng.normal(0, 0.012 * rating / n, len(t))
                rows.append(pd.DataFrame({"example": example, "rating_kva": rating, "t_aest": t,
                                          "phase": f"L{k + 1}", "V": v.round(2), "P_kw": p.round(4),
                                          "Q_kvar": q.round(4), "P_pot_kw": np.nan}))
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    return path


def load_synthetic_fleet():
    """Invented per-site Q-V profiles (the qv_profiles format) for 80 sites, building the CSV if absent."""
    if not SYNTHETIC_FLEET_CSV.exists():
        make_synthetic_fleet()
    prof = pd.read_csv(SYNTHETIC_FLEET_CSV)
    info = {"label": "Synthetic fleet", "date": "2025-01", "source": "synthetic (invented data)",
            "n_sites": prof.site_id.nunique()}
    return prof, info


def make_synthetic_fleet(path: Path = SYNTHETIC_FLEET_CSV, seed: int = 4779, n_sites: int = 80) -> Path:
    """
    Invented fleet: each site responds at a gain of the Volt-VAr curve (about half
    near 1, a quarter flat, the rest shallow or steep), shifted by a small offset,
    with its own voltage range. Labelled 'conformant' when 0.8 <= gain <= 1.2. Not real data.
    """
    from shared.as4777_curves import vvar_required_q

    rng = np.random.default_rng(seed)
    gains = np.concatenate([rng.normal(1.0, 0.12, n_sites // 2), rng.normal(0.02, 0.05, n_sites // 4),
                            rng.uniform(0.3, 0.7, n_sites // 8), rng.uniform(1.3, 1.6, n_sites - n_sites // 2
                                                                            - n_sites // 4 - n_sites // 8)])
    rows = []
    for i, g in enumerate(gains):
        lo, hi = rng.uniform(228, 238), rng.uniform(250, 260)
        off = rng.normal(0, 1.5)
        for v in np.arange(np.floor(lo), np.ceil(hi)) + 0.5:
            rows.append({"site_id": 9000 + i, "cohort": "1P", "v_bin": v,
                         "q_med": round(g * vvar_required_q(v, 100) + off + rng.normal(0, 1.0), 2),
                         "n": int(rng.integers(5, 60)), "conformant": bool(0.8 <= g <= 1.2)})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# =============================================================================
# 2. Local Solar Analytics store
# =============================================================================
def local_store_dir() -> Path:
    """Same resolution as synthetic_ami_creation/config/ami_config.py (override: CICCADA_AMI_STORE_DIR)."""
    override = os.environ.get("CICCADA_AMI_STORE_DIR")
    if override:
        return Path(os.path.expandvars(override)).expanduser()
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else Path.home() / ".local" / "share") / "ciccada" / "ami_store"


def _local_range(site_id: int, t0, t1):
    """PV-circuit rows (UTC [t0, t1)) and site metadata from ami_raw_phaseseparate."""
    import duckdb

    store = local_store_dir()
    months = pd.period_range(t0, t1 - pd.Timedelta(seconds=1), freq="M").strftime("%Y-%m")
    dirs = [store / "ami_raw_phaseseparate" / f"dt_month={m}" for m in months]
    dirs = [d for d in dirs if d.exists()]
    if not dirs:
        raise FileNotFoundError(f"No ami_raw_phaseseparate partitions for {list(months)} under {store}")
    files = ", ".join(f"'{d.as_posix()}/*.parquet'" for d in dirs)
    con = duckdb.connect()
    try:
        df = con.execute(f"""
            SELECT circuit_id, t_stamp, V, P_kw_signed AS P_kw, Q_kvar_signed AS Q_kvar
            FROM read_parquet([{files}])
            WHERE site_id = {int(site_id)} AND circuit_type = 'pv_site_net'
              AND t_stamp >= TIMESTAMP '{t0}' AND t_stamp < TIMESTAMP '{t1}'
        """).df()
        site = con.execute(f"""
            SELECT * FROM read_parquet('{(store / "ami_site_metadata.parquet").as_posix()}')
            WHERE site_id = {int(site_id)} LIMIT 1
        """).df()
        circ = con.execute(f"""
            SELECT circuit_id, {", ".join(CIRCUIT_FIELDS)}
            FROM read_parquet('{(store / "ami_circuit_metadata.parquet").as_posix()}')
            WHERE site_id = {int(site_id)} AND circuit_type = 'pv_site_net'
        """).df()
    finally:
        con.close()
    return df, _site_meta(site, circ)


def load_local(site_id: int, date: str):
    """
    PV circuits of one site for one AEST day from the local store
    (ami_raw_phaseseparate, circuit_type = 'pv_site_net'; P/Q polarity-corrected at build).
    Covers only the sites and months landed in the synthetic-AMI extract.
    """
    df, meta = _local_range(site_id, *_day_bounds_utc(date))
    if df.empty:
        raise ValueError(f"No PV rows for site {site_id} on {date} in the local store")
    return _finish(df, meta, site_id, date, "local store")


# =============================================================================
# 3. AWS (Athena)
# =============================================================================
def _aws_range(site_id: int, t0, t1):
    """
    PV-circuit rows (UTC [t0, t1)) and metadata from solar_analytics_iceberg.
    Partition predicates cover every UTC month the range touches, so an AEST day
    or month is never clipped at a month boundary.
    """
    from shared.aws_config import aq
    from shared.ciccada_config import SAI

    months = pd.period_range(t0, t1 - pd.Timedelta(seconds=1), freq="M")
    part_sql = " OR ".join(f"(year = {m.year} AND month = {m.month})" for m in months)
    meta = aq(f"""
        SELECT DISTINCT circuit_id, {", ".join(SITE_FIELDS + CIRCUIT_FIELDS)}
        FROM meta_up23c
        WHERE is_pv = True AND site_id = {int(site_id)}
    """, database=SAI)
    if meta.empty:
        raise ValueError(f"No PV circuits for site {site_id}")
    ids = ", ".join(str(int(c)) for c in meta["circuit_id"].unique())
    df = aq(f"""
        SELECT circuit_id, t_stamp, voltage AS V, power, energy_reactive
        FROM ts
        WHERE is_pv = True AND ({part_sql}) AND circuit_id IN ({ids})
          AND t_stamp >= TIMESTAMP '{t0}' AND t_stamp < TIMESTAMP '{t1}'
    """, database=SAI)
    if not df.empty:
        pol = meta.drop_duplicates("circuit_id").set_index("circuit_id")["circuit_polarity"]
        k = df["circuit_id"].map(pol).astype(float)              # polarity correction, as pull_site_telemetry
        df["P_kw"] = df["power"] / 1000 * k
        df["Q_kvar"] = df["energy_reactive"] / 1000 * 12 * k    # 5-min energy -> power, as pull_site_telemetry
    return df, _site_meta(meta, meta)


def load_aws(site_id: int, date: str):
    """PV circuits of one site for one AEST day from solar_analytics_iceberg.ts."""
    df, meta = _aws_range(site_id, *_day_bounds_utc(date))
    if df.empty:
        raise ValueError(f"No telemetry for site {site_id} on {date}")
    return _finish(df, meta, site_id, date, "AWS")


# =============================================================================
# 4. Picking a day
# =============================================================================
def list_days(site_id: int, month: str, source: str = "local", top: int = 10) -> pd.DataFrame:
    """
    Rank the AEST days of `month` ('YYYY-MM') for a site-day plot: most intervals
    above 240 V (Volt-VAr absorbing region) first, then most PV energy.
    Voltage is the mean over the site's PV circuits; energy is their sum.
    """
    m0 = pd.Timestamp(f"{month}-01")
    t0, t1 = m0 - pd.Timedelta(hours=10), m0 + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=10)
    fetch = {"local": _local_range, "aws": _aws_range}[source]
    df, site = fetch(site_id, t0, t1)
    if df.empty:
        raise ValueError(f"No PV rows for site {site_id} in {month} ({source})")
    t = pd.to_datetime(df["t_stamp"])
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    df["t_aest"] = t + pd.Timedelta(hours=10)
    n_circ = df["circuit_id"].nunique()
    rating = float(site["ac_capacity_kw"])
    ts = df.groupby("t_aest").agg(V=("V", "mean"), P_kw=("P_kw", "sum"), n=("circuit_id", "nunique"))
    ts = ts[ts.n == n_circ]                                   # timestamps with every PV circuit
    ts["date"] = ts.index.date
    step_h = 5 / 60
    out = ts.groupby("date").agg(
        intervals=("V", "size"),
        v_max=("V", "max"),
        above_240=("V", lambda v: int((v > 240).sum())),
        above_253=("V", lambda v: int((v > 253).sum())),
        pv_kwh=("P_kw", lambda p: float(p.clip(lower=0).sum() * step_h)),
        p_max_pct=("P_kw", lambda p: float(p.max() / rating * 100)),
    ).round(1)
    out = out.sort_values(["above_240", "pv_kwh"], ascending=False).reset_index()
    out["date"] = out["date"].astype(str)
    print(f"Site {site_id}  ·  {month}  ·  {n_circ} PV circuit(s)  ·  rating {rating:g} kW  ·  {len(out)} days")
    return out.head(top)


def _site_meta(site: pd.DataFrame, circ: pd.DataFrame) -> dict:
    """One dict of site metadata (SITE_FIELDS) plus a summary of the PV circuits (CIRCUIT_FIELDS)."""
    if site.empty:
        raise ValueError("No site metadata found")
    row = site.iloc[0]
    out = {f: (None if pd.isna(row[f]) else row[f]) for f in SITE_FIELDS if f in site.columns}
    ids = sorted(circ["circuit_id"].dropna().astype(int).unique()) if "circuit_id" in circ else []
    out["n_pv_circuits"] = len(ids)
    out["pv_circuit_ids"] = ids
    for f in CIRCUIT_FIELDS:
        if f in circ:
            vals = sorted({str(v) for v in circ[f].dropna()})
            out[f] = ", ".join(vals) if vals else None
    if "circuit_polarity" in circ and len(ids):
        out["_polarity"] = circ.drop_duplicates("circuit_id").set_index("circuit_id")["circuit_polarity"]
    return out


def _month_scan(store: Path, month: str):
    """Parquet list and UTC bounds covering one AEST month (the store is partitioned by UTC month)."""
    t0, t1 = _month_bounds_utc(month)
    months = pd.period_range(t0, t1 - pd.Timedelta(seconds=1), freq="M").strftime("%Y-%m")
    dirs = [store / "ami_raw_phaseseparate" / f"dt_month={m}" for m in months]
    dirs = [d for d in dirs if d.exists()]
    if not dirs:
        raise FileNotFoundError(f"No ami_raw_phaseseparate partitions for {list(months)} under {store}")
    return "[" + ", ".join(f"'{d.as_posix()}/*.parquet'" for d in dirs) + "]", t0, t1


def _month_bounds_utc(month: str):
    """AEST calendar month 'YYYY-MM' -> [start, end) in UTC."""
    m0 = pd.Timestamp(f"{month}-01")
    return m0 - pd.Timedelta(hours=10), m0 + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=10)


def load_month(site_id: int, month: str, source: str = "local"):
    """Every PV-circuit interval of one site for one AEST month ('YYYY-MM'), as the tidy frame."""
    fetch = {"local": _local_range, "aws": _aws_range}[source]
    df, site = fetch(site_id, *_month_bounds_utc(month))
    if df.empty:
        raise ValueError(f"No PV rows for site {site_id} in {month} ({source})")
    return _finish(df, site, site_id, month, "local store" if source == "local" else "AWS")


def _finish(df, site, site_id, date, source):
    t = pd.to_datetime(df["t_stamp"])
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    df["t_aest"] = t + pd.Timedelta(hours=10)
    df = _label_phases(df)
    df["P_pot_kw"] = np.nan
    site = {k: v for k, v in site.items() if not k.startswith("_")}
    info = {"rating_kva": float(site["ac_capacity_kw"]),
            "date": date if len(str(date)) == 7 else str(pd.Timestamp(date).date()),
            "label": f"Site {site_id}", "source": f"{source} (Solar Analytics PV circuits)",
            "rating_basis": "ac_capacity_kw (proxy for rated apparent power)", "site": site}
    return df[COLUMNS].sort_values(["phase", "t_aest"]).reset_index(drop=True), info


# =============================================================================
# 5. Finding example site-days (local store)
# =============================================================================
def find_examples(month: str, n: int = 3, *, vv_min: int = 12, vw_min: int = 6,
                  conform_max: float = 0.10, fail_min: float = 0.50,
                  memory_limit: str = "6GB", threads: int = 4, v_floor: float | None = 238.0,
                  where: str = "", temp_dir: str | None = None, all_days: bool = False) -> pd.DataFrame:
    """
    Scan one month of the local store and return example site-days for every
    combination of cohort (1P = one PV circuit, 3P = three PV circuits) and
    Volt-VAr / Volt-Watt outcome, using the SAME rules the family-A plots draw:

      Volt-VAr   assessable (|P_total| >= 20% rating) intervals with V_avg > 240 V;
                 non-conforming = Q_total outside the Figure-2.1-capped band.
      Volt-Watt  intervals with V_avg > 253 V and P_total >= 20% rating;
                 non-conforming = P_total above the ceiling + 4%.

    A mode is "conforms" if <= `conform_max` of those intervals are
    non-conforming, "fails" if >= `fail_min`, and needs at least `vv_min` /
    `vw_min` such intervals (1 h / 30 min at 5-min data) to be judged at all.
    Point-wise, one day: an illustration, not a site verdict.
    v_floor (V): circuit readings below it are dropped before aggregating, which
    makes the scan several times faster. Only intervals with V_avg > 240 V are
    judged, so this loses at most a few 3-phase intervals where one phase sits
    below the floor. None = no prefilter.
    where: extra SQL predicate on the raw rows, e.g. "site_id % 10 = 0" to scan a
    tenth of the sites (for a quick look or to split a long scan).
    all_days=True returns every scanned site-day instead of the picks.
    """
    import duckdb
    from shared.ciccada_config import AS4777
    from shared.as4777_curves import (q_conformance_floor_absorbing_sql, vvar_required_q_sql, vw_max_p_sql)

    store = local_store_dir()
    files, t0, t1 = _month_scan(store, month)
    tol, pmin = AS4777["TOL_FRAC"], AS4777["QCAP"]["P_MIN"]
    meta = (store / "ami_site_metadata.parquet").as_posix()

    sql = f"""
    WITH pv AS (
        SELECT site_id, circuit_id, t_stamp + INTERVAL 10 HOUR AS t_aest, V,
               P_kw_signed AS P, Q_kvar_signed AS Q
        FROM read_parquet({files})
        WHERE circuit_type = 'pv_site_net' AND V BETWEEN {v_floor or 150} AND 300 AND t_stamp >= TIMESTAMP '{t0}' AND t_stamp < TIMESTAMP '{t1}' {"AND " + where if where else ""}
    ),
    ncirc AS (
        SELECT site_id, count(DISTINCT circuit_id) AS n_pv
        FROM read_parquet({files})
        WHERE circuit_type = 'pv_site_net' AND t_stamp >= TIMESTAMP '{t0}' AND t_stamp < TIMESTAMP '{t1}' {"AND " + where if where else ""} GROUP BY 1
    ),
    ts AS (
        SELECT site_id, t_aest, avg(V) AS V, sum(P) AS P, sum(Q) AS Q, count(*) AS n
        FROM pv GROUP BY 1, 2
    ),
    j AS (
        SELECT ts.*, c.n_pv, m.ac_capacity_kw AS S
        FROM ts JOIN ncirc c USING (site_id)
        JOIN (SELECT site_id, any_value(ac_capacity_kw) AS ac_capacity_kw FROM read_parquet('{meta}') GROUP BY 1) m
          USING (site_id)
        WHERE ts.n = c.n_pv AND c.n_pv IN (1, 3) AND m.ac_capacity_kw > 0
    ),
    a AS (
        SELECT *, {vvar_required_q_sql('V', 'S')} AS q_req,
                  {q_conformance_floor_absorbing_sql('P', 'S')} AS q_cap,
                  {vw_max_p_sql('V', 'S')} AS p_max
        FROM j
    ),
    b AS (
        SELECT *,
            CASE WHEN q_req + {tol} * S < 0 THEN greatest(q_req + {tol} * S, q_cap + {tol} * S)
                 ELSE q_req + {tol} * S END AS q_hi,
            CASE WHEN q_req - {tol} * S > 0 THEN least(q_req - {tol} * S, -q_cap - {tol} * S)
                 ELSE q_req - {tol} * S END AS q_lo,
            abs(P) >= {pmin} * S AS assessable
        FROM a
    )
    SELECT site_id, CAST(t_aest AS DATE) AS date, any_value(n_pv) AS n_pv, any_value(S) AS rating_kw,
           round(max(V), 1) AS v_max,
           count(*) FILTER (WHERE assessable AND V > 240) AS vv_n,
           count(*) FILTER (WHERE assessable AND V > 240 AND (Q > q_hi OR Q < q_lo)) AS vv_nc,
           count(*) FILTER (WHERE V > 253 AND P >= {pmin} * S) AS vw_n,
           count(*) FILTER (WHERE V > 253 AND P >= {pmin} * S AND P > p_max + {tol} * S) AS vw_nc
    FROM b GROUP BY 1, 2
    """
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET threads = {threads}")
        con.execute("SET preserve_insertion_order = false")
        tmp = Path(temp_dir) if temp_dir else store.parent / "duckdb_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
        days = con.execute(sql).df()
    finally:
        con.close()

    def status(nc, total, need):
        share = nc / total.where(total > 0)
        return np.select([total < need, share <= conform_max, share >= fail_min],
                         ["not exposed", "conforms", "fails"], "mixed")

    days["cohort"] = np.where(days.n_pv == 3, "3P", "1P")
    days["volt_var"] = status(days.vv_nc, days.vv_n, vv_min)
    days["volt_watt"] = status(days.vw_nc, days.vw_n, vw_min)
    days["vv_nc_share"] = (days.vv_nc / days.vv_n.where(days.vv_n > 0)).round(2)
    days["vw_nc_share"] = (days.vw_nc / days.vw_n.where(days.vw_n > 0)).round(2)
    days["date"] = days["date"].astype(str)
    if all_days:
        return days

    picks = []
    for cohort in ("1P", "3P"):
        for vv in ("conforms", "fails"):
            for vw in ("conforms", "fails"):
                sel = days[(days.cohort == cohort) & (days.volt_var == vv) & (days.volt_watt == vw)]
                sel = sel.sort_values(["vw_n", "vv_n"], ascending=False).drop_duplicates("site_id").head(n)
                picks.append(sel)
    out = pd.concat(picks, ignore_index=True)
    cols = ["cohort", "volt_var", "volt_watt", "site_id", "date", "rating_kw", "v_max",
            "vv_n", "vv_nc_share", "vw_n", "vw_nc_share"]
    print(f"{month}: {days.site_id.nunique():,} sites, {len(days):,} full site-days scanned")
    return out[cols]



# =============================================================================
# 6. Fleet Q-V profiles (local store)
# =============================================================================
def qv_profiles(month: str, *, cohort: str | None = None, site_ids=None, bin_v: float = 1.0,
                v_range: tuple[float, float] = (200.0, 270.0), min_n: int = 5, p_floor_kw: float = 0.2,
                memory_limit: str = "6GB", threads: int = 4, temp_dir: str | None = None) -> pd.DataFrame:
    """
    Per-site Q-V profile for one month: median Q (site total, % of ac_capacity_kw) in
    `bin_v`-wide bins of the site voltage (mean over PV circuits), assessable intervals
    only (|P_total| >= 20% of rating, the Figure 2.1 gate). Bins with fewer than `min_n`
    intervals are dropped. cohort: '1P' (one PV circuit), '3P' (three) or None (both).

    p_floor_kw drops circuit readings below that output before aggregating (speed);
    it cannot remove an assessable interval unless one phase of a 3-phase site is
    below it while the total is above 20% of rating.

    Returns site_id, cohort, v_bin (centre), q_med, q25, q75, n.
    """
    import duckdb
    from shared.ciccada_config import AS4777

    store = local_store_dir()
    files, t0, t1 = _month_scan(store, month)
    pmin = AS4777["QCAP"]["P_MIN"]
    meta = (store / "ami_site_metadata.parquet").as_posix()
    n_pv = {"1P": "(1)", "3P": "(3)", None: "(1, 3)"}[cohort]
    sites = f"AND site_id IN ({', '.join(str(int(x)) for x in site_ids)})" if site_ids is not None else ""

    sql = f"""
    WITH pv AS (
        SELECT site_id, circuit_id, t_stamp, V, P_kw_signed AS P, Q_kvar_signed AS Q
        FROM read_parquet({files})
        WHERE circuit_type = 'pv_site_net' AND V BETWEEN 150 AND 300 AND P_kw_signed >= {p_floor_kw} AND t_stamp >= TIMESTAMP '{t0}' AND t_stamp < TIMESTAMP '{t1}' {sites}
    ),
    ncirc AS (
        SELECT site_id, count(DISTINCT circuit_id) AS n_pv
        FROM read_parquet({files})
        WHERE circuit_type = 'pv_site_net' AND t_stamp >= TIMESTAMP '{t0}' AND t_stamp < TIMESTAMP '{t1}' {sites} GROUP BY 1
    ),
    ts AS (SELECT site_id, t_stamp, avg(V) AS V, sum(P) AS P, sum(Q) AS Q, count(*) AS n FROM pv GROUP BY 1, 2),
    j AS (
        SELECT ts.site_id, c.n_pv, ts.V, 100 * ts.Q / m.S AS q_pct
        FROM ts JOIN ncirc c USING (site_id)
        JOIN (SELECT site_id, any_value(ac_capacity_kw) AS S FROM read_parquet('{meta}') GROUP BY 1) m USING (site_id)
        WHERE ts.n = c.n_pv AND c.n_pv IN {n_pv} AND m.S > 0 AND abs(ts.P) >= {pmin} * m.S
          AND ts.V >= {v_range[0]} AND ts.V < {v_range[1]}
    )
    SELECT site_id, any_value(n_pv) AS n_pv, floor(V / {bin_v}) * {bin_v} + {bin_v / 2} AS v_bin,
           median(q_pct) AS q_med, quantile_cont(q_pct, 0.25) AS q25, quantile_cont(q_pct, 0.75) AS q75,
           count(*) AS n
    FROM j GROUP BY site_id, v_bin
    HAVING count(*) >= {min_n}
    ORDER BY site_id, v_bin
    """
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET threads = {threads}")
        con.execute("SET preserve_insertion_order = false")
        tmp = Path(temp_dir) if temp_dir else store.parent / "duckdb_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
        prof = con.execute(sql).df()
    finally:
        con.close()
    prof.insert(1, "cohort", np.where(prof.pop("n_pv") == 3, "3P", "1P"))
    print(f"{month}: {prof.site_id.nunique():,} sites, {len(prof):,} site-bins")
    return prof.round({"q_med": 2, "q25": 2, "q75": 2})


def slope_verdicts(month: str | None = None, columns=None) -> pd.DataFrame:
    """
    Site-month Volt-VAr verdicts from the slope method (notebook 13,
    vvar_phase/fleet_conformance/fleet_site_month.parquet): site_id, year, month, cohort
    (1L1P / 3L1P / 3L3P), slope / passed (net meter, blind), slope_truth / pass_truth
    (PV circuits, reference; NA where the reference could not be fitted), oem and dnsp
    (anonymised codes), install_date, install_year, install_era, rating_kw.
    `month` 'YYYY-MM' filters to one month; `columns` picks columns (default: all).
    """
    f = local_store_dir() / "vvar_phase" / "fleet_conformance" / "fleet_site_month.parquet"
    d = pd.read_parquet(f, columns=columns)
    if month:
        y, m = map(int, month.split("-"))
        d = d[(d.year == y) & (d.month == m)]
    return d.reset_index(drop=True)


# =============================================================================
# 7. Category counts per site (family C)
# =============================================================================
#: Interval categories, in plotting order (worst -> best -> not judged).
VV_CATEGORIES = ("adverse", "inactive", "significant_shortfall", "near_conformant", "conformant",
                 "major_surplus", "not_exposed", "not_assessable")
VW_CATEGORIES = ("nonconformant", "conformant", "not_exposed")
SYNTHETIC_CATEGORY_CSV = ROOT / "synthetic_category_counts.csv"


def category_counts(month: str, *, cohort: str | None = None, site_ids=None, memory_limit: str = "6GB",
                    threads: int = 4, temp_dir: str | None = None, with_metadata: bool = True,
                    _intervals: bool = False) -> pd.DataFrame:
    """
    One row per site for one month: how many 5-min intervals fall in each Volt-VAr and
    Volt-Watt category, using exactly the rules of families A and B (site total P and Q,
    mean phase voltage, rating = ac_capacity_kw):

      Volt-VAr  not_assessable   |P| < 20 % of rating (Figure 2.1 gate)
                not_exposed      assessable, inside the band, 220 <= V <= 240 (deadband)
                conformant       assessable, inside the Figure-2.1-capped band, outside the deadband
                adverse ... major_surplus   assessable, outside the band, by Q_impact (Stage 2 thresholds)
      Volt-Watt not_exposed      V <= 253
                conformant       V > 253 and P <= ceiling (curve + 4 %)
                nonconformant    V > 253 and P > ceiling

    Columns: site_id, cohort, vv_<category>..., vw_<category>..., plus site metadata
    (state, dnsp_name, manufacturer, pv_install_date, ac_capacity_kw) if with_metadata.
    Reads every interval of the month (no prefilter): about a minute per month.
    """
    import duckdb
    from shared.ciccada_config import AS4777
    from shared.as4777_curves import (q_conformance_floor_absorbing_sql, vvar_required_q_sql, vw_max_p_sql)

    store = local_store_dir()
    files, t0, t1 = _month_scan(store, month)
    tol, pmin = AS4777["TOL_FRAC"], AS4777["QCAP"]["P_MIN"]
    vv, vw, qi = AS4777["VVAR"], AS4777["VW"], AS4777["QIMP"]
    meta = (store / "ami_site_metadata.parquet").as_posix()
    n_pv = {"1P": "(1)", "3P": "(3)", None: "(1, 3)"}[cohort]
    sites = f"AND site_id IN ({', '.join(str(int(x)) for x in site_ids)})" if site_ids is not None else ""

    vv_case = f"""
        CASE WHEN NOT assessable THEN 'not_assessable'
             WHEN inside AND V >= {vv['V2']} AND V <= {vv['V3']} THEN 'not_exposed'
             WHEN inside THEN 'conformant'
             WHEN q_impact < {qi['thr1']} THEN 'adverse'
             WHEN q_impact <= {qi['thr2']} THEN 'inactive'
             WHEN q_impact < {qi['thr3']} THEN 'significant_shortfall'
             WHEN q_impact <= {qi['thr4']} THEN 'near_conformant'
             ELSE 'major_surplus' END"""
    vw_case = f"""
        CASE WHEN V <= {vw['V1']} THEN 'not_exposed'
             WHEN P > p_max + {tol} * S THEN 'nonconformant'
             ELSE 'conformant' END"""
    cnt = ",\n           ".join(
        [f"count(*) FILTER (WHERE vv_cat = '{c}') AS vv_{c}" for c in VV_CATEGORIES]
        + [f"count(*) FILTER (WHERE vw_cat = '{c}') AS vw_{c}" for c in VW_CATEGORIES])

    sql = f"""
    WITH pv AS (
        SELECT site_id, circuit_id, t_stamp, V, P_kw_signed AS P, Q_kvar_signed AS Q
        FROM read_parquet({files})
        WHERE circuit_type = 'pv_site_net' AND V BETWEEN 150 AND 300 AND t_stamp >= TIMESTAMP '{t0}' AND t_stamp < TIMESTAMP '{t1}' {sites}
    ),
    ncirc AS (SELECT site_id, count(DISTINCT circuit_id) AS n_pv FROM pv GROUP BY 1),
    ts AS (SELECT site_id, t_stamp, avg(V) AS V, sum(P) AS P, sum(Q) AS Q, count(*) AS n FROM pv GROUP BY 1, 2),
    j AS (
        SELECT ts.*, c.n_pv, m.S
        FROM ts JOIN ncirc c USING (site_id)
        JOIN (SELECT site_id, any_value(ac_capacity_kw) AS S FROM read_parquet('{meta}') GROUP BY 1) m USING (site_id)
        WHERE ts.n = c.n_pv AND c.n_pv IN {n_pv} AND m.S > 0
    ),
    a AS (
        SELECT *, {vvar_required_q_sql('V', 'S')} AS q_req,
                  {q_conformance_floor_absorbing_sql('P', 'S')} AS q_cap,
                  {vw_max_p_sql('V', 'S')} AS p_max
        FROM j
    ),
    b AS (
        SELECT *,
            CASE WHEN q_req + {tol} * S < 0 THEN greatest(q_req + {tol} * S, q_cap + {tol} * S)
                 ELSE q_req + {tol} * S END AS q_hi,
            CASE WHEN q_req - {tol} * S > 0 THEN least(q_req - {tol} * S, -q_cap - {tol} * S)
                 ELSE q_req - {tol} * S END AS q_lo,
            abs(P) >= {pmin} * S AS assessable
        FROM a
    ),
    c AS (
        SELECT *, Q >= q_lo AND Q <= q_hi AS inside,
            CASE WHEN abs(Q - q_hi) <= abs(Q - q_lo) THEN q_hi ELSE q_lo END AS edge
        FROM b
    ),
    e AS (
        SELECT *,
            (CASE WHEN abs(q_hi + q_lo) <= 1e-9 THEN 1 ELSE sign(edge) * sign(Q) END)
                * abs(Q) / (abs(edge) + 1e-9) AS q_impact
        FROM c
    ),
    f AS (SELECT site_id, t_stamp, V, P, Q, q_lo, q_hi, n_pv, {vv_case} AS vv_cat, {vw_case} AS vw_cat FROM e)
    SELECT site_id, any_value(n_pv) AS n_pv,
           {cnt}
    FROM f GROUP BY site_id
    """
    if _intervals:                      # debugging: one row per interval instead of counts
        sql = sql[:sql.rindex("SELECT site_id, any_value")] + "SELECT * FROM f"
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET threads = {threads}")
        con.execute("SET preserve_insertion_order = false")
        tmp = Path(temp_dir) if temp_dir else store.parent / "duckdb_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
        out = con.execute(sql).df()
        if _intervals:
            return out
        if with_metadata:
            m = con.execute(f"""SELECT site_id, state, dnsp_name, manufacturer, pv_install_date, ac_capacity_kw
                                FROM read_parquet('{meta}')""").df().drop_duplicates("site_id")
            out = out.merge(m, on="site_id", how="left")
    finally:
        con.close()
    out.insert(1, "cohort", np.where(out.pop("n_pv") == 3, "3P", "1P"))
    out["month"] = month
    print(f"{month}: {len(out):,} sites")
    return out


def load_synthetic_category_counts():
    """Invented category_counts table (400 sites), building the CSV if absent."""
    if not SYNTHETIC_CATEGORY_CSV.exists():
        make_synthetic_category_counts()
    d = pd.read_csv(SYNTHETIC_CATEGORY_CSV, parse_dates=["pv_install_date"])
    return d, {"label": "Synthetic fleet", "date": "2025-01", "source": "synthetic (invented data)"}


def make_synthetic_category_counts(path: Path = SYNTHETIC_CATEGORY_CSV, seed: int = 4780, n_sites: int = 400) -> Path:
    """
    Invented per-site counts in the category_counts format. Sites installed after the
    standard became mandatory conform more often; three networks and six manufacturers
    differ a little. Not real data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sites):
        year = int(rng.integers(2016, 2026))
        after = year >= 2022
        cohort = "3P" if rng.random() < 0.12 else "1P"
        dnsp = rng.choice(["Network A", "Network B", "Network C"], p=[0.5, 0.3, 0.2])
        oem = rng.choice([f"Maker {k}" for k in "ABCDEF"], p=[0.3, 0.25, 0.15, 0.12, 0.1, 0.08])
        good = (0.65 if after else 0.25) + {"Network A": 0, "Network B": 0.05, "Network C": -0.05}[dnsp]
        n = 8640
        night = rng.uniform(0.5, 0.6)
        judged = rng.uniform(0.05, 0.25)
        if rng.random() < good:
            vv_p = [0.0, 0.005, 0.01, 0.04, 0.92, 0.025]
        else:
            vv_p = rng.dirichlet([0.3, 6, 3, 1, 1, 0.3])
        vv_judged = rng.multinomial(int(n * judged), vv_p)
        deadband = int(n * (1 - night - judged))
        vw_exposed = int(n * rng.uniform(0, 0.05))
        vw_nc = rng.binomial(vw_exposed, 0.05 if rng.random() < good else 0.6)
        row = {"site_id": 8000 + i, "cohort": cohort}
        row.update({f"vv_{c}": int(v) for c, v in zip(VV_CATEGORIES[:6], vv_judged)})
        row.update({"vv_not_exposed": deadband, "vv_not_assessable": n - deadband - int(vv_judged.sum()),
                    "vw_nonconformant": int(vw_nc), "vw_conformant": int(vw_exposed - vw_nc),
                    "vw_not_exposed": n - vw_exposed, "state": "NSW", "dnsp_name": dnsp, "manufacturer": oem,
                    "pv_install_date": pd.Timestamp(year, int(rng.integers(1, 13)), 1),
                    "ac_capacity_kw": 15.0 if cohort == "3P" else 5.0, "month": "2025-01"})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path



def slope_sites() -> pd.DataFrame:
    """
    Site-level Volt-VAr verdicts from notebook 13 (fleet_site.parquet): conformant = the
    net-meter slope passes in every assessable month (conformant_sens: in >= 50 % of
    months); n_months, share_pass, cohort, oem, dnsp, install_year, install_era, rating_kw.
    """
    return pd.read_parquet(local_store_dir() / "vvar_phase" / "fleet_conformance" / "fleet_site.parquet")


SYNTHETIC_SLOPE_CSV = ROOT / "synthetic_slope_verdicts.csv"


def load_synthetic_slope_verdicts():
    """Invented site-month slope verdicts in the slope_verdicts format, building the CSV if absent."""
    if not SYNTHETIC_SLOPE_CSV.exists():
        make_synthetic_slope_verdicts()
    d = pd.read_csv(SYNTHETIC_SLOPE_CSV, parse_dates=["install_date"])
    return d, {"label": "Synthetic fleet", "date": "2024-01 to 2025-12", "source": "synthetic (invented data)"}


def make_synthetic_slope_verdicts(path: Path = SYNTHETIC_SLOPE_CSV, seed: int = 4781, n_sites: int = 600) -> Path:
    """
    Invented site-months (24 months) in the slope_verdicts format. Each site has a fixed
    chance of passing that rises with install year, plus a summer bump; the net-meter
    verdict agrees with the reference ~93 % of the time. Not real data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sites):
        year = int(rng.integers(2014, 2026))
        inst = pd.Timestamp(year, int(rng.integers(1, 13)), 1)
        cohort = rng.choice(["1L1P", "3L1P", "3L3P"], p=[0.85, 0.08, 0.07])
        oem = rng.choice([f"M{k}" for k in (10, 13, 15, 32, 39)], p=[0.15, 0.3, 0.25, 0.1, 0.2])
        dnsp = rng.choice(["DNSP_5", "DNSP_8", "DNSP_13", "DNSP_15"], p=[0.15, 0.25, 0.4, 0.2])
        p_site = 0.05 + (0.45 if inst >= pd.Timestamp("2021-12-18") else 0.0) + {"M13": 0.05, "M39": -0.03}.get(oem, 0)
        for y in (2024, 2025):
            for m in range(1, 13):
                if pd.Timestamp(y, m, 1) < inst or rng.random() < 0.25:
                    continue
                p = np.clip(p_site + (0.05 if m in (12, 1, 2) else 0), 0, 1)
                truth = bool(rng.random() < p)
                if truth:
                    s_truth = float(np.clip(rng.normal(-3.3, 0.3), -3.99, -2.68))
                else:
                    r = rng.random()
                    s_truth = float(rng.normal(0, 0.3) if r < 0.6 else
                                    (rng.uniform(-2.6, -0.5) if r < 0.9 else rng.uniform(-6.5, -4.1)))
                s_blind = s_truth + rng.normal(0, 0.35)
                blind = bool(-4.0 <= s_blind <= -2.667)
                rows.append({"site_id": 7000 + i, "year": y, "month": m, "cohort": cohort, "passed": blind,
                             "pass_truth": truth, "slope": round(s_blind, 3), "slope_truth": round(s_truth, 3),
                             "oem": oem, "dnsp": dnsp, "install_date": inst,
                             "install_year": year,
                             "install_era": "after 2020 standard mandatory" if inst >= pd.Timestamp("2021-12-18") else "before",
                             "rating_kw": 5.0})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path



# =============================================================================
# 8. Curtailed energy per site (family F)
# =============================================================================
SYNTHETIC_CURTAILMENT_CSV = ROOT / "synthetic_curtailment.csv"


def oem_store_dir() -> Path:
    """Local SolarEdge (OEM) analysis store, next to the AMI store: %LOCALAPPDATA%/ciccada/oem_analysis_store."""
    return local_store_dir().parent / "oem_analysis_store"


def se_curtailment(year: int | None = 2025) -> pd.DataFrame:
    """
    Volt-Watt curtailment per site from the local SolarEdge store (oem_analysis D11:
    se_conformance_voltwattghi, GHI counterfactual; generation from se_conformance_voltwatt).
    One row per site: site_id (= site_alias), cohort ('1P' / '3P'), state,
    voltwatt_kwh, generation_kwh, rating_kva (s_99 proxy), days_exposed.
    Energies use the 5-min interval (kW sums x 5/60). Volt-VAr-induced curtailment
    (Method B) needs se_uncurtailedpv, which is not in the store, so it is not included.
    """
    store = oem_store_dir()
    ghi = pd.read_parquet(store / "se_conformance_voltwattghi.parquet")
    vw = pd.read_parquet(store / "se_conformance_voltwatt.parquet",
                         columns=["site_alias", "is_three_phase", "state", "day_aest", "rating_kva", "P_kW_sum"])
    if year is not None:
        ghi = ghi[ghi.day_aest.dt.year == year]
        vw = vw[vw.day_aest.dt.year == year]
    h = 5 / 60
    gen = vw.groupby("site_alias").agg(generation_kwh=("P_kW_sum", "sum"), rating_kva=("rating_kva", "first"),
                                       is_three_phase=("is_three_phase", "first"), state=("state", "first"))
    gen["generation_kwh"] *= h
    cur = ghi.groupby("site_alias").agg(voltwatt_kwh=("curtailment_sum", "sum"),
                                        days_exposed=("exposed_count", lambda x: int((x > 0).sum())))
    cur["voltwatt_kwh"] *= h
    out = gen.join(cur, how="left").fillna({"voltwatt_kwh": 0.0, "days_exposed": 0}).reset_index()
    out = out.rename(columns={"site_alias": "site_id"})
    out.insert(1, "cohort", np.where(out.pop("is_three_phase"), "3P", "1P"))
    return out


def sa_curtailment_aws(year: int = 2025) -> pd.DataFrame:
    """
    Volt-Watt curtailment per site from the Solar Analytics Stage 2 table on Athena
    (conformance_voltwattghi_v2_flex_included, GHI counterfactual): site_id,
    voltwatt_kwh, exposed_kwh (generation on the days in that table). Untested.
    """
    from shared.aws_config import aq
    from shared.ciccada_config import SAI, TABLES

    return aq(f"""
        SELECT site_id,
               sum(curtailment_voltwattghi_sum) * 5.0 / 60 AS voltwatt_kwh,
               sum(P_kW_sum) * 5.0 / 60 AS exposed_kwh
        FROM {TABLES["conformance_voltwattghi"]}
        WHERE year = {int(year)}
        GROUP BY site_id
    """, database=SAI)


def load_synthetic_curtailment():
    """Invented per-site curtailment (two mechanisms) in the se_curtailment format, building the CSV if absent."""
    if not SYNTHETIC_CURTAILMENT_CSV.exists():
        make_synthetic_curtailment()
    return pd.read_csv(SYNTHETIC_CURTAILMENT_CSV), {"label": "Synthetic fleet", "date": "2025",
                                                    "source": "synthetic (invented data)"}


def make_synthetic_curtailment(path: Path = SYNTHETIC_CURTAILMENT_CSV, seed: int = 4782, n_sites: int = 800) -> Path:
    """
    Invented per-site curtailed energy: about 40 % of sites affected by Volt-Watt and 60 %
    by Volt-VAr-induced curtailment, both heavy-tailed (lognormal), more in high-voltage
    networks. Not real data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sites):
        state = rng.choice(["New South Wales", "Queensland", "South Australia"], p=[0.45, 0.35, 0.2])
        hv = {"New South Wales": 1.0, "Queensland": 1.6, "South Australia": 1.3}[state]
        rating = float(rng.choice([5.0, 6.0, 8.0, 10.0]))
        gen = rating * rng.uniform(1100, 1600)
        vw = float(rng.lognormal(2.0, 1.4) * hv) if rng.random() < 0.4 * hv / 1.3 else 0.0
        vv = float(rng.lognormal(1.5, 1.2) * hv) if rng.random() < 0.6 else 0.0
        rows.append({"site_id": f"SYN{i:04d}", "cohort": "3P" if rng.random() < 0.15 else "1P", "state": state,
                     "rating_kva": rating, "generation_kwh": round(gen, 1), "voltwatt_kwh": round(min(vw, 0.2 * gen), 2),
                     "voltvar_kwh": round(min(vv, 0.2 * gen), 2)})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
