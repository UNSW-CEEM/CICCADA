"""
build_all_uncurtailedpv.py  —  Stage 1, step 4 of 4
===================================================

Clean rewrite of Hossein's `Write_All_uncartailedPV.ipynb`.

WHAT IT DOES
------------
Applies the fitted GHI model to EVERY eligible interval to produce the
counterfactual "what this site would have generated without curtailment":

        P_norm_est = P_norm_cs * (a + b * GHI/GHI_cs)   [floored at actual]
        uncurtailed_P = P_norm_est * S_99

This `all_uncurtailedpv{SUFFIX}` table is what your notebook 03 (Method B and
the eligible/all denominators) joins to. It is the single most important
Stage-1 output for the curtailment paper.

FIXES BAKED IN
--------------
  * R6 — caps uncurtailed_P at ac_capacity_kw (nameplate) as a sanity bound.
         Hossein's model produced impossible outliers (e.g. 591 kW on a 17 kW
         site). The counterfactual cannot exceed the inverter's rated capacity.
         The number of intervals hit by the cap is reported by validate() so you
         can quote it in the paper.

  * Keeps Hossein's conservative floor: the estimate is never below the actual
    measured power (you can't be curtailed to MORE than your potential).

READS
-----
`structured_data{SUFFIX}` + `pv_ghi_norm_model{SUFFIX}`, restricted to sites in
the local `mape<50_sites.csv` quality gate (MAPE < 50%).

SAFE-BY-DEFAULT
---------------
Writes to `all_uncurtailedpv{SUFFIX}` (default "_v2").
"""

import pandas as pd
from build_structured_data import TABLE_SUFFIX

SD    = f"structured_data{TABLE_SUFFIX}"
MODEL = f"pv_ghi_norm_model{TABLE_SUFFIX}"
TARGET = f"all_uncurtailedpv{TABLE_SUFFIX}"

TIME_BIN_MIN = 5


def create_table(aq, database):
    aq(f"DROP TABLE IF EXISTS {TARGET}", database=database)
    aq(f"""
        CREATE TABLE {TARGET} (
            site_id       BIGINT,
            t_stamp       TIMESTAMP,
            year          INT,
            month         INT,
            uncurtailed_P DOUBLE,
            P_kw          DOUBLE,
            GHI           DOUBLE,
            n_train       BIGINT,
            capped        BOOLEAN,     -- R6: TRUE where the nameplate cap bit
            year_p        INT,
            month_p       INT
        )
        WITH (
            format = 'PARQUET',
            partitioning = ARRAY['year_p', 'month_p']
        )
    """, database=database)
    return f"Created empty {TARGET}"


def _acceptable_sites_csv(csv_path):
    """Read the MAPE<50% site list into a comma-separated SQL fragment."""
    ids = pd.read_csv(csv_path)["site_id"].tolist()
    return ", ".join(map(str, ids))


def run_year(aq, database, year, mape_csv_path, n_parts=3, parts=None):
    """
    Apply the model for one year. `mape_csv_path` points at your local
    mape<50_sites.csv quality-gate file.

    FIX R6 (nameplate cap) is applied in the final SELECT.
    """
    if parts is None:
        parts = list(range(n_parts))

    acceptable = _acceptable_sites_csv(mape_csv_path)
    results = []
    for part in parts:
        part_filter = f"site_id % {n_parts} = {part}"
        aq(f"""
            INSERT INTO {TARGET}
            WITH eligible AS (
                SELECT
                    sd.site_id, sd.actual_day, sd.t_stamp,
                    CAST(date_trunc('minute', sd.t_stamp + interval '10' hour)
                         - interval '1' minute * (minute(sd.t_stamp + interval '10' hour) % {TIME_BIN_MIN})
                         AS TIME) AS tod_bin,
                    sd.GHI / sd.GHI_cs AS x,
                    sd.P_kw_norm, sd.P_kw_norm_cs, sd.S_99,
                    m.ac_capacity_kw
                FROM {SD} sd
                JOIN (SELECT DISTINCT site_id, ac_capacity_kw FROM meta_up23c) m
                  ON sd.site_id = m.site_id
                WHERE sd.P_kw_norm_cs > 0.2 AND sd.GHI > 50 AND sd.P_kw_norm > 0.05
                  AND sd.P_kw_norm <= sd.P_kw_norm_cs
                  AND sd.year = {year} AND {part_filter}
                  AND sd.site_id IN ({acceptable})
            ),
            applied AS (
                SELECT
                    e.site_id, e.t_stamp, e.x AS GHI,
                    e.P_kw_norm * e.S_99 AS P_kw,
                    e.ac_capacity_kw,
                    CASE WHEN e.P_kw_norm_cs * (mo.a + mo.b * e.x) >= e.P_kw_norm
                         THEN e.P_kw_norm_cs * (mo.a + mo.b * e.x)
                         ELSE e.P_kw_norm END * e.S_99 AS uncurtailed_P_raw,
                    mo.n AS n_train
                FROM eligible e
                JOIN {MODEL} mo ON e.site_id = mo.site_id AND e.tod_bin = mo.tod_bin
            )
            SELECT
                site_id, t_stamp,
                year(t_stamp)  AS year,
                month(t_stamp) AS month,
                -- FIX R6: cap the counterfactual at nameplate
                least(uncurtailed_P_raw, ac_capacity_kw) AS uncurtailed_P,
                P_kw, GHI, n_train,
                (uncurtailed_P_raw > ac_capacity_kw)     AS capped,
                year(t_stamp)  AS year_p,
                month(t_stamp) AS month_p
            FROM applied
            WHERE uncurtailed_P_raw IS NOT NULL
        """, database=database)
        results.append(f"applied year={year} part={part}/{n_parts}")
        print(results[-1])
    return results


def validate(aq, database):
    # Sanity: uncurtailed_P should never be below measured P_kw
    below = aq(f"SELECT count(*) AS n FROM {TARGET} WHERE uncurtailed_P < P_kw",
               database=database)
    # How often did the R6 cap bite?
    cap = aq(f"""
        SELECT count(*) AS n_rows,
               sum(CASE WHEN capped THEN 1 ELSE 0 END) AS n_capped,
               round(100.0 * sum(CASE WHEN capped THEN 1 ELSE 0 END) / count(*), 3) AS pct_capped
        FROM {TARGET}
    """, database=database)
    print(f"Rows with uncurtailed_P < P_kw (should be 0): {int(below['n'].iloc[0])}")
    print("R6 nameplate cap impact:")
    print(cap.to_string(index=False))
    return below, cap
