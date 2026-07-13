"""
Data calc-write pipeline: Stage 1. Step 3b of 4
================================================

Computes per-site MAPE on the held-out validation days, then saves a CSV of
site_ids that pass the quality gate (MAPE < threshold).

This CSV is consumed by step 4 (build_all_uncurtailedpv) to restrict the
counterfactual to sites where the model is trustworthy.

READS
-----
`structured_data{SUFFIX}`, `pv_ghi_norm_model{SUFFIX}`, `split_days{SUFFIX}`

WRITES
------
A local CSV file (not an Athena table). Default: `mape_under50_sites.csv`
in the current working directory.
"""

from build_structured_data import TABLE_SUFFIX

SD    = f"structured_data{TABLE_SUFFIX}"
MODEL = f"pv_ghi_norm_model{TABLE_SUFFIX}"
SPLIT = f"split_days{TABLE_SUFFIX}"

DEFAULT_MAPE_THRESHOLD = 50  # percent
DEFAULT_CSV_PATH = "mape_under50_sites.csv"


def compute_mape(aq, database):
    """
    Run the model on validation days and return a DataFrame with columns:
    site_id, n_val_intervals, mape_pct.
    """
    mape_df = aq(f"""
        WITH val_data AS (
            SELECT
                sd.site_id, sd.t_stamp,
                sd.P_kw_norm,
                sd.P_kw_norm_cs,
                sd.GHI / sd.GHI_cs AS x,
                CAST(date_trunc('minute', sd.t_stamp + interval '10' hour)
                     - interval '1' minute * (minute(sd.t_stamp + interval '10' hour) % 5)
                     AS TIME) AS tod_bin
            FROM {SD} sd
            JOIN {SPLIT} sp ON sd.site_id = sp.site_id AND sd.actual_day = sp.actual_day
            WHERE sp.day_type = 'val'
              AND sd.P_kw_norm_cs > 0.2 AND sd.GHI > 50 AND sd.P_kw_norm > 0.05
              AND sd.P_kw_norm <= sd.P_kw_norm_cs
              AND sd.V <= 253 AND (sd.P_kw_norm >= 1 OR sd.S_norm < 1.001)
        ),
        predictions AS (
            SELECT
                v.site_id,
                v.P_kw_norm AS actual,
                v.P_kw_norm_cs * (m.a + m.b * v.x) AS predicted
            FROM val_data v
            JOIN {MODEL} m ON v.site_id = m.site_id
                AND CAST(v.tod_bin AS VARCHAR) = m.tod_bin
            WHERE m.n >= 5
        )
        SELECT
            site_id,
            count(*) AS n_val_intervals,
            round(100.0 * avg(abs(predicted - actual) / NULLIF(actual, 0)), 2) AS mape_pct
        FROM predictions
        GROUP BY site_id
    """, database=database)
    return mape_df


def save_quality_gate(mape_df, threshold=DEFAULT_MAPE_THRESHOLD, csv_path=DEFAULT_CSV_PATH):
    """
    Filter to sites with MAPE < threshold and save to CSV.
    Returns the filtered DataFrame.
    """
    good_sites = mape_df[mape_df['mape_pct'] < threshold][['site_id']].copy()
    good_sites.to_csv(csv_path, index=False)
    return good_sites


def run(aq, database, threshold=DEFAULT_MAPE_THRESHOLD, csv_path=DEFAULT_CSV_PATH):
    """
    Full pipeline: compute MAPE, print summary, save CSV.
    Returns (mape_df, good_sites).
    """
    mape_df = compute_mape(aq, database)

    print(f"Total sites with validation data: {len(mape_df):,}")
    print(f"MAPE distribution:")
    print(mape_df['mape_pct'].describe().round(1))

    good_sites = save_quality_gate(mape_df, threshold=threshold, csv_path=csv_path)

    print(f"\nSites with MAPE < {threshold}%: {len(good_sites):,} "
          f"({100 * len(good_sites) / len(mape_df):.1f}% of total)")
    print(f"Saved to {csv_path}")

    return mape_df, good_sites
