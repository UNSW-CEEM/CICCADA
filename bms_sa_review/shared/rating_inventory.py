"""
Rating-basis inventory (phase 6). Diagnostic only: changes no result.
=====================================================================

Every AS/NZS 4777.2 quantity is scaled by S_rated, the inverter's rated
(maximum) apparent power in kVA: the required Q, the +/-4% tolerance band, the
20% power threshold between zone A (response required) and zone C (low power,
voltage exposed), and the Figure 2.1 minimum. The pipelines use
`ac_capacity_kw` from the site metadata as the stand-in. This module asks how
good a stand-in it is:

    load_site_metadata    one row per site: manufacturer, model, ac_capacity_kw,
                          s_99, inverter_count (local store, free; or Athena,
                          a metadata-only query, megabytes)
    field_inventory       what the metadata can and cannot tell (model coverage,
                          unidentified models, one value per model?)
    model_table           per model: sites, ac_capacity_kw value(s), observed
                          apparent power s_99 and its ratio to ac_capacity_kw
    load_datasheets       PRIVATE datasheet ratings (outside the repository:
                          they name manufacturers): nominal kW, max kVA, source
    compare_with_datasheets / rating_gap_summary
                          ac_capacity_kw vs datasheet max kVA per model and the
                          number of sites in each case
    candidate_rating      per-site datasheet max kVA where known, else
                          ac_capacity_kw. NOT used by any pipeline.

Nothing here writes to Athena or to the site metadata table.

Reading s_99: it is the 99th percentile of the apparent power a site actually
delivered. It is capped by the PV array, the weather and export limits, so a
ratio below 1 does not prove the rating is right; a ratio clearly above 1 does
show the inverter delivers more than ac_capacity_kw.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

#: Datasheet file columns.
DATASHEET_COLUMNS = ("manufacturer", "model", "nominal_kw", "max_kva", "source_url", "note")
#: ac_capacity_kw within this relative distance of the datasheet max kVA counts as a match.
MATCH_TOL = 0.02
#: Manufacturer labels that mean "model unknown" (matched case-insensitively, substring).
UNIDENTIFIED = ("generic",)


# ---------------------------------------------------------------------------
# Paths (private data lives outside the repository)
# ---------------------------------------------------------------------------
def _ciccada_base() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else Path.home() / ".local" / "share") / "ciccada"


def local_metadata_path() -> Path:
    """Local Solar Analytics site metadata (override: CICCADA_AMI_STORE_DIR)."""
    store = os.environ.get("CICCADA_AMI_STORE_DIR")
    root = Path(os.path.expandvars(store)).expanduser() if store else _ciccada_base() / "ami_store"
    return root / "ami_site_metadata.parquet"


def datasheet_path() -> Path:
    """Private datasheet ratings CSV (override: CICCADA_RATINGS_CSV)."""
    p = os.environ.get("CICCADA_RATINGS_CSV")
    return Path(os.path.expandvars(p)).expanduser() if p else (
        _ciccada_base() / "ratings" / "inverter_datasheet_ratings.csv")


# ---------------------------------------------------------------------------
# Site metadata
# ---------------------------------------------------------------------------
META_COLUMNS = ("site_id", "manufacturer", "model", "ac_capacity_kw", "dc_capacity_kw",
                "s_99", "inverter_count", "flex_export_detected")


def athena_metadata_sql(metadata_table="meta_up23c") -> str:
    """One row per PV site. Metadata only (no telemetry): megabytes scanned."""
    return f"""
        SELECT site_id,
               max(manufacturer)        AS manufacturer,
               max(model)               AS model,
               max(ac_capacity_kw)      AS ac_capacity_kw,
               max(dc_capacity_kw)      AS dc_capacity_kw,
               max(s_99)                AS s_99,
               max(inverter_count)      AS inverter_count,
               bool_or(coalesce(flex_export_detected, False)) AS flex_export_detected,
               count(DISTINCT model)    AS n_model_values,
               count(DISTINCT ac_capacity_kw) AS n_ac_capacity_values
        FROM {metadata_table}
        WHERE is_pv = True
        GROUP BY site_id
    """


def load_site_metadata(source="local", aq=None, database=None, path=None) -> pd.DataFrame:
    """
    source='local': the local store's ami_site_metadata.parquet (free; the sites
    in the local extract). source='athena': athena_metadata_sql via `aq`
    (all PV sites; metadata only).
    """
    if source == "local":
        p = Path(path) if path else local_metadata_path()
        d = pd.read_parquet(p)
        cols = [c for c in META_COLUMNS if c in d.columns]
        d = d[cols].drop_duplicates("site_id")
    elif source == "athena":
        if aq is None:
            raise ValueError("source='athena' needs aq")
        d = aq(athena_metadata_sql(), database=database) if database else aq(athena_metadata_sql())
    else:
        raise ValueError("source must be 'local' or 'athena'")
    d = d.copy()
    d["model_identified"] = ~d["manufacturer"].fillna("").str.lower().str.contains(
        "|".join(UNIDENTIFIED)) & d["model"].notna()
    d["s99_over_ac"] = d["s_99"] / d["ac_capacity_kw"].where(d["ac_capacity_kw"] > 0)
    return d


# ---------------------------------------------------------------------------
# What the metadata can tell
# ---------------------------------------------------------------------------
def field_inventory(meta: pd.DataFrame, top=(10, 20, 50)) -> pd.DataFrame:
    """One column of facts about the rating fields."""
    n = len(meta)
    ident = meta[meta.model_identified]
    per_model = ident.groupby(["manufacturer", "model"]).size().sort_values(ascending=False)
    ac_values = ident.groupby(["manufacturer", "model"])["ac_capacity_kw"].nunique()
    rows = {
        "sites": n,
        "sites with ac_capacity_kw > 0": int((meta.ac_capacity_kw > 0).sum()),
        "sites with an identified model": len(ident),
        "sites with an unidentified ('generic') model": n - len(ident),
        "distinct identified models": len(per_model),
        "identified models with more than one ac_capacity_kw value": int((ac_values > 1).sum()),
        "sites with more than one inverter": int((meta.get("inverter_count", pd.Series(1, index=meta.index)) > 1).sum()),
        "sites with s_99 above ac_capacity_kw by > 3%": int((meta.s99_over_ac > 1.03).sum()),
        "median s_99 / ac_capacity_kw": round(float(meta.s99_over_ac.median()), 3),
    }
    for k in top:
        rows[f"sites covered by the top {k} identified models"] = int(per_model.head(k).sum())
    out = pd.Series(rows, name="value").to_frame()
    out["% of sites"] = (100 * out["value"] / n).round(1)
    out.loc[["sites", "distinct identified models", "identified models with more than one "
             "ac_capacity_kw value", "median s_99 / ac_capacity_kw"], "% of sites"] = np.nan
    return out


def model_table(meta: pd.DataFrame, top=30, min_sites=1) -> pd.DataFrame:
    """Per identified model: sites, ac_capacity_kw, observed apparent power."""
    d = meta[meta.model_identified & (meta.ac_capacity_kw > 0)]
    g = d.groupby(["manufacturer", "model"])
    out = pd.DataFrame({
        "n_sites": g.size(),
        "ac_capacity_kw": g["ac_capacity_kw"].median(),
        "n_ac_capacity_values": g["ac_capacity_kw"].nunique(),
        "s_99_median_kva": g["s_99"].median().round(2),
        "s99_over_ac_median": g["s99_over_ac"].median().round(3),
        "s99_over_ac_p90": g["s99_over_ac"].quantile(0.9).round(3),
        "share_sites_s99_above_ac_3pct": g["s99_over_ac"].apply(lambda x: (x > 1.03).mean()).round(3),
    }).reset_index()
    out = out[out.n_sites >= min_sites].sort_values("n_sites", ascending=False)
    total = len(meta)
    out["cum_share_of_all_sites_pct"] = (100 * out.n_sites.cumsum() / total).round(1)
    return out.head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Datasheets (private file)
# ---------------------------------------------------------------------------
def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").str.lower().str.replace(r"[\s_]+", "", regex=True)


def _model_key(manufacturer: pd.Series, model: pd.Series) -> pd.Series:
    """Lower-case, no spaces, manufacturer prefix removed ('Acme X 5.0' = 'X 5.0')."""
    mf, md = _norm(manufacturer), _norm(model)
    return pd.Series([m[len(f):] if f and m.startswith(f) else m for f, m in zip(mf, md)],
                     index=model.index)


def load_datasheets(path=None) -> pd.DataFrame:
    p = Path(path) if path else datasheet_path()
    if not p.exists():
        raise FileNotFoundError(
            f"No datasheet ratings file at {p}. It is kept outside the repository "
            "(it names manufacturers); set CICCADA_RATINGS_CSV or create it.")
    d = pd.read_csv(p)
    missing = set(DATASHEET_COLUMNS) - set(d.columns)
    if missing:
        raise ValueError(f"datasheet file lacks columns {sorted(missing)}")
    d["key"] = _model_key(d.manufacturer, d.model)
    if d.duplicated(["key"]).any():
        raise ValueError("datasheet file has duplicate models")
    return d


def compare_with_datasheets(models: pd.DataFrame, datasheets: pd.DataFrame,
                            match_tol=MATCH_TOL) -> pd.DataFrame:
    """
    models: model_table (or any frame with manufacturer, model, n_sites,
    ac_capacity_kw). Adds the datasheet values and a `case`:
      'matches max kVA'                 ac_capacity_kw = datasheet max kVA (within match_tol)
      'below max kVA (kW, not kVA)'     ac_capacity_kw is the active-power rating and
                                        the inverter may deliver more apparent power
      'above max kVA'                   ac_capacity_kw higher than the datasheet allows
      'no datasheet'                    model not (yet) in the private file
    gap_pct = 100 * (max kVA / ac_capacity_kw - 1): how much S_rated is understated.
    """
    m = models.copy()
    m["key"] = _model_key(m.manufacturer, m.model)
    ds = datasheets[["key", "nominal_kw", "max_kva", "source_url", "note"]]
    out = m.merge(ds, on="key", how="left").drop(columns="key")
    rel = out["max_kva"] / out["ac_capacity_kw"] - 1
    out["gap_pct"] = (100 * rel).round(1)
    out["case"] = np.select(
        [out["max_kva"].isna(), rel.abs() <= match_tol, rel > match_tol],
        ["no datasheet", "matches max kVA", "below max kVA (kW, not kVA)"],
        "above max kVA",
    )
    return out


def rating_gap_summary(comparison: pd.DataFrame, n_sites_total=None) -> pd.DataFrame:
    """Sites and models per case, with the median gap."""
    g = comparison.groupby("case")
    out = pd.DataFrame({
        "models": g.size(),
        "sites": g["n_sites"].sum(),
        "median_gap_pct": g["gap_pct"].median(),
        "max_gap_pct": g["gap_pct"].max(),
    }).sort_values("sites", ascending=False)
    if n_sites_total:
        out["% of all sites"] = (100 * out["sites"] / n_sites_total).round(1)
    return out


def candidate_rating(meta: pd.DataFrame, datasheets: pd.DataFrame) -> pd.DataFrame:
    """
    Per site: datasheet max kVA where the model is known, else ac_capacity_kw,
    and where the value came from. For a FUTURE opt-in rating basis only; no
    pipeline reads it.
    """
    d = meta[["site_id", "manufacturer", "model", "ac_capacity_kw"]].copy()
    d["key"] = _model_key(d.manufacturer, d.model)
    d = d.merge(datasheets[["key", "max_kva"]], on="key", how="left").drop(columns="key")
    d["candidate_s_rated_kva"] = d["max_kva"].fillna(d["ac_capacity_kw"])
    d["candidate_source"] = np.where(d["max_kva"].notna(), "datasheet max kVA", "ac_capacity_kw")
    return d


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_gap(comparison: pd.DataFrame, title="ac_capacity_kw vs datasheet max kVA"):
    """Bar per model with a datasheet: gap_pct (how much S_rated is understated)."""
    import matplotlib.pyplot as plt

    d = comparison.dropna(subset=["max_kva"]).sort_values("gap_pct")
    labels = d["manufacturer"].astype(str) + " " + d["model"].astype(str) + \
        "  (" + d["n_sites"].astype(int).astype(str) + " sites)"
    fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(d))), dpi=120)
    colours = np.where(d["case"] == "below max kVA (kW, not kVA)", "#C0504D", "#7F7F7F")
    ax.barh(labels, d["gap_pct"], color=colours)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="datasheet max kVA vs ac_capacity_kw (%)", title=title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig
