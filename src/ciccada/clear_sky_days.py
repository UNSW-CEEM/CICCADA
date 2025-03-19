import pandas as pd


def detect_clear_sky_day(ghi_df: pd.DataFrame, min_max_ghi: float) -> bool:
    """Check whether a certain day is a clear sky day or not.

    It will judge that it is a clear sky day if satisfying two criteria:
    1. The average change in ghi is small (less than 5 W/m2).
    2. The maximum ghi value is higher than a certain threshold (min_max_ghi).

    This algorithim is based on: https://github.com/UNSW-CEEM/Solar-Curtailment/blob/163d31545bcc7bdf049b59b470fa15636c867fed/src/solarcurtailment/clear_sky_day.py#L204C1-L236C42

    Examples:

    >>> ghi_data = pd.DataFrame({
    ... 'ghi_mean': [501.0, 502.0, 503.0]
    ... })

    >>> detect_clear_sky_day(ghi_data, min_max_ghi=500.0)
    True

    Args:
        ghi_df (df) : ghi data sorted in time sequential order with a column
            `mean_ghi` specifying the ghi for the interval in W/m2.
        min_max_ghi (int) : the minimum value of maximum ghi. If the maximum ghi is
            lower than this value it means there must be cloud.

    Returns:
        (bool) : bool value if the day is clear sky day or not.
    """
    df_daytime = ghi_df.loc[ghi_df["ghi_mean"] > 0]
    collective_change = df_daytime["ghi_mean"].diff().abs().sum()
    average_change = collective_change / len(df_daytime.index)
    return average_change < 26 and max(ghi_df.ghi_mean) > min_max_ghi, average_change


def old_detect_clear_sky_day(ghi_df, min_max_ghi):
    """Check whether a certain day is a clear sky day or not.

    Args:
        ghi_df (df) : ghi data
        min_max_ghi (int) : the minimum value of maximum ghi. If the maximum ghi is lower than
                            this value, means there must be cloud.

    Returns:
        (bool) : bool value if the day is clear sky day or not.

    It will judge that it is a clear sky day if satisfying two criterias:
    1. There is no sudden change in ghi (means cloud)
    2. The maximum ghi value is higher than a certain threshold (min_max_ghi).
    """

    df_daytime = ghi_df.loc[ghi_df["ghi_mean"] > 0]

    collective_change = 0
    ghi_list = df_daytime.ghi_mean.tolist()

    for i in range(len(ghi_list) - 1):
        collective_change += abs(ghi_list[i + 1] - ghi_list[i])

    if len(df_daytime.index) == 0:
        return False, 0

    average_delta_y = collective_change / len(df_daytime.index)

    if average_delta_y < 26 and max(ghi_df.ghi_mean) > min_max_ghi:
        return True, average_delta_y
    else:
        return False, average_delta_y
