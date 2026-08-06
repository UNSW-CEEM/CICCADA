"""Create the cleaned Solar Analytics parquet required by conformance."""

from solar_analytics_workflow.preprocessing import build_cleaned_site_data
from solar_analytics_workflow.solar_paths import CLEANED_DATA_PATH


def main():
    print(
        "Building deduplicated Solar Analytics data in 128 circuit buckets...\n"
        f"Output: {CLEANED_DATA_PATH}",
        flush=True,
    )
    cleaned_path = build_cleaned_site_data(
        deduplicate=True,
        num_buckets=128,
    )
    print(f"Saved cleaned Solar Analytics data to {cleaned_path}")


if __name__ == "__main__":
    main()
