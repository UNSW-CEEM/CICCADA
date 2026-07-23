"""Create the cleaned Solar Analytics parquet required by conformance."""

from solar_analytics_workflow.preprocessing import build_cleaned_site_data


def main():
    cleaned_path = build_cleaned_site_data()
    print(f"Saved cleaned Solar Analytics data to {cleaned_path}")


if __name__ == "__main__":
    main()
