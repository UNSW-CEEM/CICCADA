"""Create the cleaned SAPN parquet required by conformance."""

from sapn2022_workflow.preprocessing import build_cleaned_site_data
from sapn2022_workflow.sapn_paths import CLEANED_SITE_DATA_PATH


def main():
    print(
        "Building deduplicated SAPN site data in 128 circuit buckets "
        f"(the 4 GB source may take several minutes)...\n"
        f"Output: {CLEANED_SITE_DATA_PATH}",
        flush=True,
    )
    cleaned_path = build_cleaned_site_data(
        deduplicate=True,
        num_buckets=128,
    )
    print(f"Saved cleaned site data to {cleaned_path}")


if __name__ == "__main__":
    main()
