"""Plot single-site, single-day EVM nonconformance traces."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
import sapn2022_metrics_5m_data_checks as data_checks
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from path_config import require_local_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALL_UNCURTAILED = (
    PROJECT_ROOT / "outputs" / "prediction" / "all_uncurtailedPV_5m.parquet"
)
DEFAULT_CURTAILMENT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "curtailed_estimates_5m"
    / "curtailment_sapn2022_5m.parquet"
)
# The tier-bucket CSV lives outside this repo, so the local SAPN root is
# defined in the ignored `local_paths.py` file instead of being committed here.
SAPN_ROOT = require_local_path(
    "SAPN_ROOT",
    "root folder containing `updated results/phase b info for curtailment/tier based/`.",
)
DEFAULT_ELIGIBLE_BUCKETS5M = Path(
    SAPN_ROOT
    / "updated results"
    / "phase b info for curtailment"
    / "tier based"
    / "tier_based_5min_buckets.csv"
)

DEFAULT_YEAR = 2022
DEFAULT_MONTH = 11
ADELAIDE_TZ = "Australia/Adelaide"
LOCAL_TZINFO = ZoneInfo(ADELAIDE_TZ)
WINDOW_START = time(6, 0, 0)
WINDOW_END = time(18, 0, 0)

FONT_FAMILY = "Times New Roman"
# Match Grafana's default classic palette for the four displayed series.
GRAFANA_CLASSIC_COLORS = (
    "#7EB26D",
    "#EAB839",
    "#6ED0E0",
    "#EF843C",
)
P_KW_COLOR = GRAFANA_CLASSIC_COLORS[0]
UNCURTAILED_COLOR = GRAFANA_CLASSIC_COLORS[1]
NONCONFORMANCE_COLOR = GRAFANA_CLASSIC_COLORS[2]
VOLTAGE_COLOR = GRAFANA_CLASSIC_COLORS[3]
GRID_COLOR = "#E5E7EB"
SPINE_COLOR = "#D1D5DB"
TEXT_COLOR = "#374151"

PLOT_COLUMNS = [
    "site_id",
    "local_tstamp",
    "P_kw",
    "uncurtailed_P",
    "los_or_ov1_flag",
    "voltage_10m_avg",
    "nonconformance_EVM",
]


def _coerce_local_date(local_date: int | str | date) -> date:
    """Normalise supported day inputs to an Adelaide local calendar date.

    Accepted inputs:
    - `14` -> interpreted as 2022-11-14 for this SAPN November 2022 dataset
    - `"2022-11-14"` -> parsed directly
    - `date` or `datetime` -> converted to a date
    """
    if isinstance(local_date, datetime):
        return local_date.date()

    if isinstance(local_date, date):
        return local_date

    if isinstance(local_date, int):
        return date(DEFAULT_YEAR, DEFAULT_MONTH, local_date)

    try:
        return datetime.strptime(local_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"local_date must be an int day-of-month, YYYY-MM-DD string, or date object, got {local_date!r}"
        ) from exc


def _window_filter(timestamp_col: str, target_date: date) -> pl.Expr:
    """Return the fixed 06:00-18:00 Adelaide local-day filter expression."""
    return (
        (pl.col(timestamp_col).dt.date() == pl.lit(target_date))
        & (pl.col(timestamp_col).dt.time() >= WINDOW_START)
        & (pl.col(timestamp_col).dt.time() <= WINDOW_END)
    )


def _read_uncurtailed_rows(
    all_uncurtailed_path: Path,
    site_id: int,
    target_date: date,
) -> pl.LazyFrame:
    """Load optional uncurtailed parquet rows for one site and one local day window."""
    return (
        pl.scan_parquet(all_uncurtailed_path)
        .select(["site_id", "local_tstamp", "uncurtailed_P"])
        .with_columns(
            [
                pl.col("site_id").cast(pl.Int64),
                pl.col("local_tstamp").cast(pl.Datetime(time_zone=ADELAIDE_TZ)),
                pl.col("uncurtailed_P").cast(pl.Float64),
            ]
        )
        .filter(pl.col("site_id") == site_id)
        .filter(_window_filter("local_tstamp", target_date))
        .sort("local_tstamp")
    )


def _read_bucket_rows(
    eligible_buckets5m_path: Path,
    site_id: int,
    target_date: date,
) -> pl.LazyFrame:
    """Load full-timeline tier-based 5-minute bucket rows for one site and one local day window."""
    return (
        pl.scan_csv(eligible_buckets5m_path)
        .with_columns(
            [
                pl.col("site_id").cast(pl.Int64),
                pl.col("los_or_ov1_flag").cast(pl.Int8),
                pl.col("site_power_kw_avg").cast(pl.Float64),
                pl.col("v10m_avg_avg").cast(pl.Float64),
                pl.col("bucket_5min_local")
                .cast(pl.Utf8)
                .str.strptime(
                    pl.Datetime(time_zone=ADELAIDE_TZ),
                    "%Y-%m-%d %H:%M:%S%z",
                    strict=False,
                )
                .alias("local_tstamp"),
            ]
        )
        .filter(pl.col("local_tstamp").is_not_null())
        .select(
            [
                "site_id",
                "local_tstamp",
                "site_power_kw_avg",
                "los_or_ov1_flag",
                "v10m_avg_avg",
            ]
        )
        .unique(subset=["site_id", "local_tstamp"], keep="first")
        .filter(pl.col("site_id") == site_id)
        .filter(_window_filter("local_tstamp", target_date))
        .sort("local_tstamp")
    )


def prepare_site_day_evm_data(
    site_id: int,
    local_date: int | str | date,
    all_uncurtailed_path: Path = DEFAULT_ALL_UNCURTAILED,
    eligible_buckets5m_path: Path = DEFAULT_ELIGIBLE_BUCKETS5M,
) -> pl.DataFrame:
    """Build the joined dataframe used by the EVM day plot.

    This function:
    - reads full-timeline `P_kw`, `los_or_ov1_flag`, and `v10m_avg_avg`
      from `tier_based_5min_buckets`
    - reads optional `uncurtailed_P` from `all_uncurtailedPV_5m`
    - keeps only the requested site and local day within 06:00-18:00
    - left-joins `uncurtailed_P` onto the full bucket timeline
    - derives `voltage_10m_avg` and `nonconformance_EVM`
    """
    target_date = _coerce_local_date(local_date)
    all_uncurtailed_path = Path(all_uncurtailed_path)
    eligible_buckets5m_path = Path(eligible_buckets5m_path)

    bucket_rows = _read_bucket_rows(
        eligible_buckets5m_path,
        site_id,
        target_date,
    ).collect()
    uncurtailed = _read_uncurtailed_rows(
        all_uncurtailed_path,
        site_id,
        target_date,
    ).collect()

    if bucket_rows.is_empty():
        raise ValueError(
            f"No bucket plot rows found for site_id={site_id} on {target_date:%Y-%m-%d} "
            f"between {WINDOW_START:%H:%M} and {WINDOW_END:%H:%M} local time."
        )

    # Reuse the shared 5-minute key checks so the plot fails loudly if an input
    # unexpectedly contains duplicate site/timestamp rows.
    data_checks.assert_unique_site_timestamp_keys(
        bucket_rows,
        "tier_based_5min_buckets plot rows",
    )
    if not uncurtailed.is_empty():
        data_checks.assert_unique_site_timestamp_keys(
            uncurtailed,
            "all_uncurtailedPV_5m plot rows",
        )

    return (
        bucket_rows.join(
            uncurtailed,
            on=["site_id", "local_tstamp"],
            how="left",
        )
        .with_columns(
            [
                pl.col("site_power_kw_avg").alias("P_kw"),
                pl.col("v10m_avg_avg").alias("voltage_10m_avg"),
                pl.when(pl.col("los_or_ov1_flag") == 1)
                .then(pl.col("site_power_kw_avg"))
                .otherwise(pl.lit(0.0))
                .alias("nonconformance_EVM"),
            ]
        )
        .select(PLOT_COLUMNS)
        .sort("local_tstamp")
    )


def site_with_days(site_id: int, curtailment_summary: pl.DataFrame) -> list[int]:
    """Return sorted unique day values for one site from the curtailment summary."""
    return (
        curtailment_summary.filter(pl.col("site_id") == site_id)
        .get_column("day")
        .unique()
        .sort()
        .to_list()
    )


def plot_site_day_evm(
    site_id: int,
    local_date: int | str | date,
    all_uncurtailed_path: Path = DEFAULT_ALL_UNCURTAILED,
    eligible_buckets5m_path: Path = DEFAULT_ELIGIBLE_BUCKETS5M,
    save_plot: bool = False,
    save_path: Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Create a single-site, single-day EVM plot and optionally save it.

    The returned figure and axes let you inspect or tweak the plot after calling
    the function. If `save_plot` is True, the PNG is written either to the
    provided `save_path` or to the default `outputs/plots` location.
    """
    target_date = _coerce_local_date(local_date)
    if save_path is None:
        save_path = (
            PROJECT_ROOT
            / "outputs"
            / "plots"
            / f"site_{site_id}_{target_date.day}_evm_day_plot.png"
        )

    plot_df = prepare_site_day_evm_data(
        site_id=site_id,
        local_date=target_date,
        all_uncurtailed_path=Path(all_uncurtailed_path),
        eligible_buckets5m_path=Path(eligible_buckets5m_path),
    )
    plot_pd = plot_df.to_pandas()

    x_start = datetime.combine(target_date, WINDOW_START, tzinfo=LOCAL_TZINFO)
    x_end = datetime.combine(target_date, WINDOW_END, tzinfo=LOCAL_TZINFO)

    with plt.rc_context({"font.family": FONT_FAMILY}):
        fig, ax_left = plt.subplots(figsize=(12.5, 4.0), dpi=dpi)
        ax_right = ax_left.twinx()

        fig.patch.set_facecolor("white")
        ax_left.set_facecolor("white")

        ax_left.plot(
            plot_pd["local_tstamp"],
            plot_pd["uncurtailed_P"],
            color=UNCURTAILED_COLOR,
            linewidth=2.0,
            label="uncurtailed_P",
            zorder=2,
        )
        ax_left.plot(
            plot_pd["local_tstamp"],
            plot_pd["nonconformance_EVM"],
            color=NONCONFORMANCE_COLOR,
            linewidth=2.2,
            label="nonconformance_EVM",
            alpha=0.85,
            zorder=3,
        )
        # Plot actual power last so it remains visible where traces overlap.
        ax_left.plot(
            plot_pd["local_tstamp"],
            plot_pd["P_kw"],
            color=P_KW_COLOR,
            linewidth=2.8,
            label="P_kw",
            zorder=4,
        )
        ax_right.plot(
            plot_pd["local_tstamp"],
            plot_pd["voltage_10m_avg"],
            color=VOLTAGE_COLOR,
            linewidth=2.0,
            label="voltage_10m_avg",
            zorder=1,
        )

        ax_left.set_xlim(x_start, x_end)
        ax_left.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=LOCAL_TZINFO))
        ax_left.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M", tz=LOCAL_TZINFO)
        )

        ax_left.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
        ax_left.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6)
        ax_right.grid(False)

        for axis in (ax_left, ax_right):
            axis.tick_params(colors=TEXT_COLOR, labelsize=9)
            axis.spines["top"].set_visible(False)
            axis.spines["bottom"].set_color(SPINE_COLOR)
            axis.spines["left"].set_color(SPINE_COLOR)
            axis.spines["right"].set_color(SPINE_COLOR)

        ax_left.set_xlabel("Local time", color=TEXT_COLOR, fontsize=10)
        ax_left.set_ylabel("Power (kW)", color=TEXT_COLOR, fontsize=10)
        ax_right.set_ylabel("Voltage (V)", color=TEXT_COLOR, fontsize=10)

        ax_left.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.1f} kW")
        )
        ax_right.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.1f} V")
        )

        handles_left, labels_left = ax_left.get_legend_handles_labels()
        handles_right, labels_right = ax_right.get_legend_handles_labels()
        left_lookup = dict(zip(labels_left, handles_left))
        ax_left.legend(
            [
                left_lookup["P_kw"],
                left_lookup["uncurtailed_P"],
                left_lookup["nonconformance_EVM"],
                handles_right[0],
            ],
            ["P_kw", "uncurtailed_P", "nonconformance_EVM", "voltage_10m_avg"],
            loc="upper left",
            bbox_to_anchor=(0.0, -0.18),
            ncol=4,
            frameon=False,
            fontsize=9,
            handlelength=1.6,
            columnspacing=1.0,
        )

        fig.subplots_adjust(bottom=0.24, left=0.08, right=0.92, top=0.97)

        if save_plot:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, (ax_left, ax_right)


def main():
    # plot for each site-day listed in curtailment_sapn2022_5m."""
    curtailment_summary = (
        pl.read_parquet(DEFAULT_CURTAILMENT_SUMMARY)
        .select(["site_id", "day"])
        .unique()
        .sort(["site_id", "day"])
    )
    site_ids = curtailment_summary["site_id"].unique().sort().to_list()

    for site_id in site_ids:
        days = site_with_days(site_id, curtailment_summary)
        for day in days:
            fig, _ = plot_site_day_evm(
                site_id=site_id,
                local_date=day,
                save_plot=True,
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
