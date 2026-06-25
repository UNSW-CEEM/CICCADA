"""Simple data checks for SAPN2022 5-minute curtailment metrics."""

import polars as pl


KEY_COLUMNS = ["site_id", "local_tstamp"]


def as_lazy(df):
    if isinstance(df, pl.LazyFrame):
        return df
    return df.lazy()


def assert_unique_site_timestamp_keys(df, label, sample_size=10):
    """Raise if site_id/local_tstamp is not unique for the provided frame."""
    duplicates = (
        as_lazy(df)
        .group_by(KEY_COLUMNS)
        .len()
        .filter(pl.col("len") > 1)
        .sort(KEY_COLUMNS)
        .collect()
    )

    if duplicates.is_empty():
        return

    sample = duplicates.head(sample_size).to_dicts()
    raise ValueError(
        f"{label} contains {duplicates.height} duplicate site_id/local_tstamp key rows. "
        f"Sample: {sample}"
    )


def find_unmatched_rows(left_df, right_df):
    """Return site/timestamp keys in left_df that do not exist in right_df."""
    return (
        as_lazy(left_df)
        .select(KEY_COLUMNS)
        .join(
            as_lazy(right_df).select(KEY_COLUMNS),
            on=KEY_COLUMNS,
            how="anti",
        )
        .sort(KEY_COLUMNS)
        .collect()
    )


def assert_all_rows_have_match(left_df, right_df, left_label, right_label, sample_size=10):
    """Raise if any site/timestamp key in left_df is missing from right_df."""
    unmatched = find_unmatched_rows(left_df, right_df)
    if unmatched.is_empty():
        return

    sample = unmatched.head(sample_size).to_dicts()
    raise ValueError(
        f"Found {unmatched.height} {left_label} rows without a matching {right_label} row. "
        f"Sample: {sample}"
    )
