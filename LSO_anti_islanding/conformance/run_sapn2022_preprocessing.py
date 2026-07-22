"""Create the cleaned SAPN parquet required by conformance."""

from sapn2022_workflow.preprocessing import build_cleaned_site_data


def main():
    cleaned_path = build_cleaned_site_data()
    print(f"Saved cleaned site data to {cleaned_path}")


if __name__ == "__main__":
    main()
