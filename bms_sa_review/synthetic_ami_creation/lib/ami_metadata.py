"""
Phase 6 -- two dimension tables filling the gap the Phase 5 deliverable
trio left open: neither `ami_raw`, `ami_meter`, nor `ami_raw_phaseseparate`
carries site or circuit metadata (location, capacity, manufacturer, DNSP,
lifetime voltage/power-factor percentiles, ...) anywhere queryable
alongside them in the local Parquet store.
==============================================================================

Two tables, split by GRAIN, not by "input vs ground truth" (both tables are
useful with all three Phase 5 tables -- see each function's own docstring):

`ami_site_metadata` -- one row per site_id. Genuine site-level attributes
(location, DNSP, capacities, manufacturer/model, install dates) pulled
straight from `meta_up23c`, plus derived per-site facts that are cheap to
compute once here and expensive to recompute from the multi-hundred-million
-row Phase 5 tables every time: `n_load_phases`/`n_pv_phases` (from kept
circuit counts), `s_99` (aggregated via `max` across the site's circuits --
the SAME convention `ami_build.build_ami_raw`'s own `site_capacity_lookup`
already uses for PV normalization, reused here for consistency), and
`first_seen`/`last_seen` (aggregated `min`/`max` of each surviving circuit's
own `min_time`/`max_time` metadata fields -- cheaper and more principled
than scanning the whole landed store, since these are already computed by
the source system over each circuit's full reporting history). Also carries
one presence flag per Phase 5 table (`in_ami_raw`, `in_ami_meter`,
`in_ami_raw_phaseseparate`), since -- as established during this project --
those three tables do NOT share an identical site_id set.

`ami_circuit_metadata` -- one row per (site_id, device_id, circuit_id).
Genuinely circuit-level attributes that cannot be folded into
`ami_site_metadata` without an aggregation decision: `circuit_type`,
`is_pv`, `device_type` (the field the `apply_power_correction` device-model
correction is keyed on), `circuit_polarity`, `voltage_class`, `m_id`
(carried through for completeness but NOT reliable for phase-grouping --
see `ami_resolution`'s own docstrings), `min_time`/`max_time`, the lifetime
voltage percentiles (`v_95`/`v_05`/`v_99`/`v_01`) and power-factor
percentiles (`avg_pf`/`std_pf`/`pf_99`/`pf_01`), and the raw per-circuit
`s_99` (distinct from `ami_site_metadata`'s max-aggregated site-level
value). Also carries `n_long`/`n_lat`/`distance_km` as untouched pass
-through columns -- their meaning was not established during this project
and they are not used or interpreted here; keeping them raw was a
deliberate choice over guessing at where they belong.

Both tables are scoped to whatever is ACTUALLY landed in the local Parquet
store right now (the union of site_ids / circuit_ids across the three
Phase 5 tables), not to a specific historical notebook run's in-memory
resolution objects -- see `determine_landed_scope` -- so they stay correct
across reruns and store extensions (e.g. widening `TRIAL_MONTHS`) without
needing to replay Phase 4/5's resolution notebook.

Every Athena-touching function here is pure given the `aq_fn` callable it's
handed, mirroring `ami_extract`'s own convention -- no `ami_athena` import
happens in this module; a notebook wires in `ami_athena.aq` for a real run,
tests wire in a fake. `meta_up23c` is small and unpartitioned (~424k rows,
~17MB) and is not in `ami_config.BIG_TABLES`, so `Athena.aq`'s partition
guard does not require a partition predicate for the queries built here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import ami_extract as Extract

__all__ = [
    "SITE_LEVEL_COLUMNS",
    "CIRCUIT_LEVEL_COLUMNS",
    "build_site_level_query",
    "build_circuit_level_query",
    "pull_site_level_meta",
    "pull_circuit_level_meta",
    "determine_landed_scope",
    "determine_landed_circuit_scope",
    "build_ami_site_metadata",
    "build_ami_circuit_metadata",
    "write_metadata_table",
]


#: Genuine site-level attributes -- one true value per site_id, safe to
#: pull as-is (after de-duplicating on site_id; `meta_up23c` itself is
#: circuit-grain, so a naive SELECT repeats these once per circuit).
SITE_LEVEL_COLUMNS = (
    "site_id", "state", "postcode", "longitude", "latitude", "dnsp_name",
    "dc_capacity_kw", "ac_capacity_kw", "ac_capacity_kw_json",
    "export_limit_kw", "monitoring_start", "inverter_count",
    "pv_install_date", "manufacturer", "model", "flex_export_detected",
)

#: Genuinely circuit-level attributes -- vary per (device_id, circuit_id)
#: within a site. `n_long`/`n_lat`/`distance_km` are carried through
#: unexamined (see module docstring).
CIRCUIT_LEVEL_COLUMNS = (
    "site_id", "device_id", "circuit_id", "circuit_type", "is_pv",
    "device_type", "circuit_polarity", "voltage_class", "m_id",
    "min_time", "max_time", "v_95", "v_05", "v_99", "v_01",
    "avg_pf", "std_pf", "pf_99", "pf_01", "s_99",
    "n_long", "n_lat", "distance_km",
)


def build_site_level_query(site_ids) -> str:
    """`SELECT <SITE_LEVEL_COLUMNS> FROM meta_up23c WHERE site_id IN (...)`. Pure."""
    ids = ",".join(str(int(s)) for s in sorted({int(s) for s in site_ids}))
    cols = ", ".join(SITE_LEVEL_COLUMNS)
    return f"SELECT {cols} FROM meta_up23c WHERE site_id IN ({ids})"


def build_circuit_level_query(circuit_ids) -> str:
    """`SELECT <CIRCUIT_LEVEL_COLUMNS> FROM meta_up23c WHERE circuit_id IN (...)`. Pure."""
    ids = ",".join(str(int(c)) for c in sorted({int(c) for c in circuit_ids}))
    cols = ", ".join(CIRCUIT_LEVEL_COLUMNS)
    return f"SELECT {cols} FROM meta_up23c WHERE circuit_id IN ({ids})"


def pull_site_level_meta(
    aq_fn, site_ids, *, chunk_size: int = 800, database=None,
) -> pd.DataFrame:
    """
    Chunked `meta_up23c` pull for `SITE_LEVEL_COLUMNS`, one row per site_id
    after de-duplication (guards against any of these columns turning out
    to actually vary per circuit at a site -- takes the first occurrence,
    same convention notebook 05's `site_level_meta` already used).
    `chunk_size`/chunking reuses `ami_extract.chunk_circuit_ids` -- despite
    its name, it is a generic sorted-int chunker with no circuit-specific
    behaviour.
    """
    chunks = Extract.chunk_circuit_ids(site_ids, chunk_size=chunk_size)
    if not chunks:
        return pd.DataFrame(columns=list(SITE_LEVEL_COLUMNS))
    frames = [
        aq_fn(
            build_site_level_query(chunk), database=database,
            label=f"site metadata chunk ({len(chunk)} sites)",
        )
        for chunk in chunks
    ]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="site_id").reset_index(drop=True)


def pull_circuit_level_meta(
    aq_fn, circuit_ids, *, chunk_size: int = 800, database=None,
) -> pd.DataFrame:
    """Chunked `meta_up23c` pull for `CIRCUIT_LEVEL_COLUMNS`, one row per circuit_id."""
    chunks = Extract.chunk_circuit_ids(circuit_ids, chunk_size=chunk_size)
    if not chunks:
        return pd.DataFrame(columns=list(CIRCUIT_LEVEL_COLUMNS))
    frames = [
        aq_fn(
            build_circuit_level_query(chunk), database=database,
            label=f"circuit metadata chunk ({len(chunk)} circuits)",
        )
        for chunk in chunks
    ]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="circuit_id").reset_index(drop=True)


def determine_landed_scope(con, table_paths: dict[str, str]) -> dict[str, set]:
    """
    Per-table set of distinct `site_id`s actually landed in the local
    Parquet store right now, via DuckDB against real files -- NOT derived
    from any notebook's in-memory resolution objects, so this stays correct
    across reruns/store extensions.

    `con` : an open `duckdb.connect()` connection.
    `table_paths` : logical table name -> a `read_parquet`-ready glob path,
        e.g. `{"ami_raw": f"{(store_dir / 'ami_raw').as_posix()}/dt_month=*/*.parquet"}`.

    Returns `{table_name: {site_id, ...}}`, one entry per key in
    `table_paths` (empty set for a table with no landed files).
    """
    result: dict[str, set] = {}
    for name, glob_path in table_paths.items():
        try:
            frame = con.sql(
                f"SELECT DISTINCT site_id FROM read_parquet('{glob_path}', hive_partitioning=1)"
            ).df()
        except Exception:
            frame = pd.DataFrame(columns=["site_id"])
        result[name] = set(frame["site_id"].tolist())
    return result


def determine_landed_circuit_scope(con, table_paths: dict[str, str]) -> set:
    """
    Union of distinct `circuit_id`s actually landed across `table_paths`'
    tables right now, via DuckDB against real files. Same idea as
    `determine_landed_scope`, but returns one combined set (there is no
    per-table circuit presence tracked here, unlike the per-table site
    presence flags `build_ami_site_metadata` records).
    """
    ids: set = set()
    for glob_path in table_paths.values():
        try:
            frame = con.sql(
                f"SELECT DISTINCT circuit_id FROM read_parquet('{glob_path}', hive_partitioning=1)"
            ).df()
        except Exception:
            continue
        ids |= set(frame["circuit_id"].tolist())
    return ids


def build_ami_site_metadata(
    site_meta: pd.DataFrame,
    circuit_meta: pd.DataFrame,
    presence: dict[str, set] | None = None,
    *,
    site_column: str = "site_id",
    circuit_column: str = "circuit_id",
    type_column: str = "circuit_type",
    load_type: str = "ac_load_net",
    pv_type: str = "pv_site_net",
) -> pd.DataFrame:
    """
    One row per site_id: `site_meta`'s columns (already de-duplicated by
    `pull_site_level_meta`, but de-duplicated again here defensively) plus
    derived columns computed from `circuit_meta` -- `n_load_phases`/
    `n_pv_phases` (kept circuit counts by type), `s_99` (max across the
    site's circuits, matching `ami_build`'s own PV-normalization
    convention), `first_seen`/`last_seen` (min/max of each circuit's own
    `min_time`/`max_time`) -- and one boolean presence column per key in
    `presence` (`in_<table_name>`), since the three Phase 5 tables do not
    share an identical site_id set (see project history: `ami_raw` is the
    strictest, `ami_raw_phaseseparate` the most permissive).

    A site present in `site_meta` but absent from `circuit_meta` still gets
    a row (derived columns come back null/zero, not dropped) -- this
    function does not assume every site_meta row has matching circuit
    detail, since the two are pulled from meta_up23c independently.
    """
    if site_meta is None or not len(site_meta):
        return pd.DataFrame(columns=[
            *SITE_LEVEL_COLUMNS, "n_load_phases", "n_pv_phases", "s_99",
            "first_seen", "last_seen",
        ])

    frame = site_meta.drop_duplicates(subset=site_column).reset_index(drop=True).copy()

    if circuit_meta is not None and len(circuit_meta):
        typed = circuit_meta[circuit_meta[type_column].isin([load_type, pv_type])]
        if len(typed):
            phase_counts = (
                typed.groupby([site_column, type_column])[circuit_column]
                .nunique()
                .unstack(fill_value=0)
                .rename(columns={load_type: "n_load_phases", pv_type: "n_pv_phases"})
                .reset_index()
            )
        else:
            phase_counts = pd.DataFrame(columns=[site_column, "n_load_phases", "n_pv_phases"])
        frame = frame.merge(phase_counts, on=site_column, how="left")
        for col in ("n_load_phases", "n_pv_phases"):
            if col not in frame.columns:
                frame[col] = 0
        frame[["n_load_phases", "n_pv_phases"]] = (
            frame[["n_load_phases", "n_pv_phases"]].fillna(0).astype(int)
        )

        if "s_99" in circuit_meta.columns:
            s99_site = circuit_meta.groupby(site_column)["s_99"].max().rename("s_99")
            frame = frame.merge(s99_site, on=site_column, how="left")
        else:
            frame["s_99"] = pd.NA

        if "min_time" in circuit_meta.columns:
            first_seen = circuit_meta.groupby(site_column)["min_time"].min().rename("first_seen")
            frame = frame.merge(first_seen, on=site_column, how="left")
        else:
            frame["first_seen"] = pd.NaT
        if "max_time" in circuit_meta.columns:
            last_seen = circuit_meta.groupby(site_column)["max_time"].max().rename("last_seen")
            frame = frame.merge(last_seen, on=site_column, how="left")
        else:
            frame["last_seen"] = pd.NaT
    else:
        frame["n_load_phases"] = 0
        frame["n_pv_phases"] = 0
        frame["s_99"] = pd.NA
        frame["first_seen"] = pd.NaT
        frame["last_seen"] = pd.NaT

    for table_name, ids in (presence or {}).items():
        frame[f"in_{table_name}"] = frame[site_column].isin(ids)

    return frame.reset_index(drop=True)


def build_ami_circuit_metadata(circuit_meta: pd.DataFrame) -> pd.DataFrame:
    """
    One row per circuit_id: `CIRCUIT_LEVEL_COLUMNS`, selected and ordered,
    de-duplicated on circuit_id. Otherwise a pass-through -- no derivation,
    unlike `build_ami_site_metadata` -- this table exists to make the raw
    circuit-level metadata queryable alongside `ami_meter`/
    `ami_raw_phaseseparate` (both circuit-grain), not to aggregate it.
    """
    columns = list(CIRCUIT_LEVEL_COLUMNS)
    if circuit_meta is None or not len(circuit_meta):
        return pd.DataFrame(columns=columns)
    present = [c for c in columns if c in circuit_meta.columns]
    return (
        circuit_meta[present]
        .drop_duplicates(subset="circuit_id")
        .sort_values(["site_id", "circuit_id"])
        .reset_index(drop=True)
    )


def write_metadata_table(frame: pd.DataFrame, path, *, compression: str = "zstd") -> Path:
    """
    Write a single (non-partitioned) Parquet file -- both metadata tables
    are one-row-per-site or one-row-per-circuit, small enough (thousands to
    tens of thousands of rows) that Hive-partitioning by month, like the
    three Phase 5 tables, would be pointless and just fragment the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, compression=compression, index=False)
    return path
