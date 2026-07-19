"""prepare_customers_data.py - clean and prepare customers data.

Cleans customers_data.csv, including the custom LoyaltyPointsCount and
PreferredContactMethod columns added in the D3.1 assignment.

Author: Abdelhafidh Mahouel
Date: 2026-07

Process:
    - Load raw customer data.
    - Normalize Region.
    - Clean LoyaltyPointsCount (numeric).
    - Clean PreferredContactMethod (category).
    - Remove duplicate rows.
    - Save prepared data to data/prepared/.

Data Source:
- data/raw/customers_data.csv

Output:
- data/prepared/customers_data_prepared.csv

Terminal command to run this file from the root project folder:

uv run python -m analytics_project.data_prep.prepare_customers_data
"""

from pathlib import Path
from typing import Final

import pandas as pd

from bizintel.utils_data import (
    check_quality,
    inspect_basic,
    load_data,
    summarize_numeric,
)
from bizintel.utils_logger import LOG, log_header

DATA_RAW: Final[Path] = Path("data/raw")
DATA_PREPARED: Final[Path] = Path("data/prepared")

CUSTOMERS_FILE: Final[Path] = DATA_RAW / "customers_data.csv"
CUSTOMERS_PREPARED: Final[Path] = DATA_PREPARED / "customers_data_prepared.csv"

# Valid contact methods, used to catch any unexpected values after normalization.
VALID_CONTACT_METHODS: Final[set[str]] = {"Email", "Phone", "Sms", "Mail"}


def prepare_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the customers DataFrame.

    WHY: Inconsistent region and contact method values, along with
    invalid loyalty point counts, will cause problems in the
    warehouse and in reporting.

    Args:
        df: Raw customers DataFrame.

    Returns:
        Cleaned customers DataFrame.
    """
    LOG.info("Preparing customers data")

    df = df.copy()

    LOG.info("Customers Prep 1. Normalize Region values")
    df["Region"] = df["Region"].str.strip().str.title()
    regions_sorted: list[str] = sorted(df["Region"].dropna().unique().tolist())
    LOG.info(f"  Regions after normalization: {regions_sorted}")

    LOG.info("Customers Prep 2. Clean LoyaltyPointsCount (numeric)")
    df["LoyaltyPointsCount"] = pd.to_numeric(df["LoyaltyPointsCount"], errors="coerce")

    is_invalid_points = df["LoyaltyPointsCount"].isna() | (df["LoyaltyPointsCount"] < 0)
    invalid_points_count: int = int(is_invalid_points.sum())
    if invalid_points_count > 0:
        LOG.warning(
            f"  {invalid_points_count} row(s) with missing or negative "
            "LoyaltyPointsCount - will be removed"
        )
    df = df.loc[~is_invalid_points].copy()

    LOG.info("Customers Prep 3. Clean PreferredContactMethod (category)")
    df["PreferredContactMethod"] = (
        df["PreferredContactMethod"].astype(str).str.strip().str.title()
    )

    is_missing_contact = df["PreferredContactMethod"].isna() | (
        df["PreferredContactMethod"].isin(["", "Nan"])
    )
    missing_contact_count: int = int(is_missing_contact.sum())
    if missing_contact_count > 0:
        LOG.warning(
            f"  {missing_contact_count} row(s) with missing "
            "PreferredContactMethod - will be removed"
        )
    df = df.loc[~is_missing_contact].copy()

    unexpected: set[str] = (
        set(df["PreferredContactMethod"].unique()) - VALID_CONTACT_METHODS
    )
    if unexpected:
        LOG.warning(f"  Unexpected PreferredContactMethod values found: {unexpected}")

    LOG.info("Customers Prep 4. Remove duplicate rows")
    before: int = df.shape[0]
    df = df.drop_duplicates()
    after: int = df.shape[0]
    LOG.info(f"  Rows before: {before}")
    LOG.info(f"  Rows after: {after}")
    LOG.info(f"  Removed {before - after} duplicate row(s)")
    LOG.info(f"  Customers prepared: {df.shape[0]} rows")
    return df


def save_prepared(df: pd.DataFrame, filepath: Path, name: str) -> None:
    """Save a prepared DataFrame to CSV.

    Args:
        df: Prepared DataFrame to save.
        filepath: Path to the output CSV file.
        name: A short name for logging.

    Returns:
        None
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    LOG.info(f"Saved {name}")
    LOG.info(f"  Rows: {df.shape[0]}")
    LOG.info(f"  Path: {filepath}")


def main() -> None:
    """Main function to run the customers data preparation logic."""
    log_header(LOG, "BI")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    LOG.info("Task 1. LOAD. Call a function to load the dataset......")
    df_customers = load_data(CUSTOMERS_FILE, "customers")

    LOG.info("Task 2. INSPECT. Call a function to inspect the dataset...")
    inspect_basic(df_customers, "customers")

    LOG.info("Task 3. CHECK QUALITY BEFORE........")
    check_quality(df_customers, "customers")

    LOG.info("Task 4. SUMMARIZE BEFORE..........")
    summarize_numeric(df_customers, "customers")

    LOG.info("Task 5. PREPARE DATASET.........")
    df_customers_prepared = prepare_customers(df_customers)

    LOG.info("Task 6. CHECK QUALITY AFTER PREPARATION........")
    check_quality(df_customers_prepared, "customers prepared")

    LOG.info("Task 7. SUMMARIZE AFTER PREPARATION........")
    summarize_numeric(df_customers_prepared, "customers prepared")

    LOG.info("Task 8. SAVE PREPARED DATASET........")
    save_prepared(df_customers_prepared, CUSTOMERS_PREPARED, "customers")

    LOG.info("Workflow complete")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


if __name__ == "__main__":
    main()
