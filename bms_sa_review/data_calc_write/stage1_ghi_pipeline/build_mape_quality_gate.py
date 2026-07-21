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
DEFAULT_METRICS_CSV_PATH = "mape_validation_metrics.csv"
MODEL_MIN_P_NORM = 0.05
MAPE_MIN_ACTUAL_NORM = 0.20
# at least XX validation intervals:
DEFAULT_MIN_VAL_INTERVALS = 30
# at least YY validation days:
DEFAULT_MIN_VAL_DAYS = 3


def compute_mape(
    aq, database, *, sd=SD, model=MODEL, split=SPLIT,
    min_actual_norm=MAPE_MIN_ACTUAL_NORM,
):
    """
    Run the model on validation days and return a DataFrame with columns:
    site_id, n_val_intervals, mape_pct.
    """
    mape_df = aq(f"""
        WITH val_data AS (
            SELECT
                sd.site_id, sd.actual_day, sd.t_stamp,
                sd.P_kw_norm,
                sd.P_kw_norm_cs,
                sd.GHI / sd.GHI_cs AS x,
                CAST(date_trunc('minute', sd.t_stamp + interval '10' hour)
                     - interval '1' minute * (minute(sd.t_stamp + interval '10' hour) % 5)
                     AS TIME) AS tod_bin
            FROM {sd} sd
            JOIN {split} sp ON sd.site_id = sp.site_id AND sd.actual_day = sp.actual_day
            WHERE sp.day_type = 'val'
              AND sd.P_kw_norm_cs > 0.2 AND sd.GHI > 50
              AND sd.P_kw_norm > {MODEL_MIN_P_NORM}
              AND sd.P_kw_norm <= sd.P_kw_norm_cs
              AND sd.V <= 253 AND (sd.P_kw_norm >= 1 OR sd.S_norm < 1.001)
        ),
        predictions AS (
            SELECT
                v.site_id,
                v.actual_day,
                v.P_kw_norm AS actual,
                v.P_kw_norm_cs * (m.a + m.b * v.x) AS predicted
            FROM val_data v
            JOIN {model} m ON v.site_id = m.site_id
                AND CAST(v.tod_bin AS VARCHAR) = m.tod_bin
            WHERE m.n >= 5
              AND abs(v.P_kw_norm) > {min_actual_norm}
        )
        SELECT
            site_id,
            count(*) AS n_val_intervals,
            count(DISTINCT actual_day) AS n_val_days,
            round(100.0 * avg(abs(predicted - actual) / NULLIF(abs(actual), 0)), 2)
                AS mape_pct,
            round(100.0 * sum(abs(predicted - actual)) / NULLIF(sum(abs(actual)), 0), 2)
                AS wape_pct,
            round(avg(abs(predicted - actual)), 4) AS mae_norm,
            round(sqrt(avg(power(predicted - actual, 2))), 4) AS rmse_norm,
            round(avg(predicted - actual), 4) AS bias_norm
        FROM predictions
        GROUP BY site_id
    """, database=database)
    return mape_df


def save_quality_gate(
    mape_df, threshold=DEFAULT_MAPE_THRESHOLD, csv_path=DEFAULT_CSV_PATH,
    min_val_intervals=DEFAULT_MIN_VAL_INTERVALS,
    min_val_days=DEFAULT_MIN_VAL_DAYS,
):
    """
    Filter to sites with MAPE < threshold and save to CSV.
    Returns the filtered DataFrame.
    """
    mask = (
        (mape_df["mape_pct"] < threshold)
        & (mape_df["n_val_intervals"] >= min_val_intervals)
        & (mape_df["n_val_days"] >= min_val_days)
    )
    good_sites = mape_df.loc[mask, ["site_id"]].copy()
    good_sites.to_csv(csv_path, index=False)
    return good_sites


def run(
    aq, database, threshold=DEFAULT_MAPE_THRESHOLD,
    csv_path=DEFAULT_CSV_PATH,
    metrics_csv_path=DEFAULT_METRICS_CSV_PATH,
    min_actual_norm=MAPE_MIN_ACTUAL_NORM,
    min_val_intervals=DEFAULT_MIN_VAL_INTERVALS,
    min_val_days=DEFAULT_MIN_VAL_DAYS,
    sd=SD, model=MODEL, split=SPLIT,
):
    """
    Full pipeline: compute MAPE, print summary, save CSV.
    Returns (mape_df, good_sites).
    """
    mape_df = compute_mape(
        aq, database, sd=sd, model=model, split=split,
        min_actual_norm=min_actual_norm,
    )
    mape_df.to_csv(metrics_csv_path, index=False)

    print(f"Total sites with validation data: {len(mape_df):,}")
    print(f"MAPE distribution:")
    print(mape_df['mape_pct'].describe().round(1))

    good_sites = save_quality_gate(
        mape_df,
        threshold=threshold,
        csv_path=csv_path,
        min_val_intervals=min_val_intervals,
        min_val_days=min_val_days,
    )

    accepted_pct = 100 * len(good_sites) / len(mape_df) if len(mape_df) else 0.0
    print(f"\nSites with MAPE < {threshold}%: {len(good_sites):,} "
          f"({accepted_pct:.1f}% of total)")
    print(f"Saved to {csv_path}")
    print(f"Full validation metrics saved to {metrics_csv_path}")
    print(
        f"Metric population: abs(actual P_norm) > {min_actual_norm}; "
        f"minimum {min_val_intervals} intervals across {min_val_days} days"
    )

    return mape_df, good_sites
