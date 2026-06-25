from funcs import (
    CLEANED_SITE_DATA_PATH,
    CIRCUIT_DETAILS_PATH,
    RAW_SITE_DATA_PATH,
    buildCleanedSiteData,
)


def main():
    cleaned_path = buildCleanedSiteData(
        rawLocalPath=RAW_SITE_DATA_PATH,
        circuitDetailsPath=CIRCUIT_DETAILS_PATH,
        cleanedLocalPath=CLEANED_SITE_DATA_PATH,
    )
    print(f"Saved cleaned site data to {cleaned_path}")


if __name__ == "__main__":
    main()
