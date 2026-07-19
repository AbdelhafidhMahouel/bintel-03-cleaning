"""prepare_products_data.py - clean and prepare products data.

Cleans products_data.csv, including the custom StockQuantityUnits and
StoreSection columns added in the D3.1 assignment.

Author: Abdelhafidh Mahouel
Date: 2026-07

Process:
    - Load raw product data.
    - Convert UnitPrice to numeric.
    - Clean StockQuantityUnits (numeric, remove missing/negative/unrealistic values).
    - Clean StoreSection (category).
    - Remove duplicate rows.
    - Save prepared data to data/prepared/.

Data Source:
- data/raw/products_data.csv

Output:
- data/prepared/products_data_prepared.csv

Terminal command to run this file from the root project folder:

uv run python -m analytics_project.data_prep.prepare_products_data
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

PRODUCTS_FILE: Final[Path] = DATA_RAW / "products_data.csv"
PRODUCTS_PREPARED: Final[Path] = DATA_PREPARED / "products_data_prepared.csv"

# Business rule: a single store location realistically holds between
# 0 and 2000 units of any one product. Anything above that is treated
# as a data entry error rather than a real inventory count.
MAX_REALISTIC_STOCK: Final[int] = 2000

VALID_STORE_SECTIONS: Final[set[str]] = {
    "Front-Store",
    "Back-Store",
    "Warehouse",
    "Online-Only",
    "Endcap",
}


def prepare_products(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the products DataFrame.

    WHY: Even clean-looking data should be verified. We confirm types,
    check for unexpected values, and validate that stock counts fall
    within a realistic business range.

    Args:
        df: Raw products DataFrame.

    Returns:
        Cleaned products DataFrame.
    """
    LOG.info("Preparing products data")

    df = df.copy()

    LOG.info("Products Prep 1. Convert UnitPrice to numeric")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    bad_prices: int = int(df["UnitPrice"].isna().sum())
    LOG.info(f"  Non-numeric prices replaced with NaN: {bad_prices}")

    LOG.info("Products Prep 2. Clean StockQuantityUnits (numeric)")
    df["StockQuantityUnits"] = pd.to_numeric(df["StockQuantityUnits"], errors="coerce")

    is_invalid_stock = (
        df["StockQuantityUnits"].isna()
        | (df["StockQuantityUnits"] < 0)
        | (df["StockQuantityUnits"] > MAX_REALISTIC_STOCK)
    )
    invalid_stock_count: int = int(is_invalid_stock.sum())
    if invalid_stock_count > 0:
        LOG.warning(
            f"  {invalid_stock_count} row(s) with missing, negative, or "
            f"unrealistic (> {MAX_REALISTIC_STOCK}) StockQuantityUnits - "
            "will be removed"
        )
    df = df.loc[~is_invalid_stock].copy()

    LOG.info("Products Prep 3. Clean StoreSection (category)")
    df["StoreSection"] = df["StoreSection"].astype(str).str.strip().str.title()

    is_missing_section = df["StoreSection"].isna() | (
        df["StoreSection"].isin(["", "Nan"])
    )
    missing_section_count: int = int(is_missing_section.sum())
    if missing_section_count > 0:
        LOG.warning(
            f"  {missing_section_count} row(s) with missing StoreSection - "
            "will be removed"
        )
    df = df.loc[~is_missing_section].copy()

    unexpected: set[str] = set(df["StoreSection"].unique()) - VALID_STORE_SECTIONS
    if unexpected:
        LOG.warning(f"  Unexpected StoreSection values found: {unexpected}")

    LOG.info("Products Prep 4. Remove duplicate rows")
    before: int = df.shape[0]
    df = df.drop_duplicates()
    after: int = df.shape[0]
    LOG.info(f"  Rows before: {before}")
    LOG.info(f"  Rows after: {after}")
    LOG.info(f"  Removed {before - after} duplicate row(s)")
    LOG.info(f"  Products prepared: {df.shape[0]} rows")
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
    """Main function to run the products data preparation logic."""
    log_header(LOG, "BI")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    LOG.info("Task 1. LOAD. Call a function to load the dataset......")
    df_products = load_data(PRODUCTS_FILE, "products")

    LOG.info("Task 2. INSPECT. Call a function to inspect the dataset...")
    inspect_basic(df_products, "products")

    LOG.info("Task 3. CHECK QUALITY BEFORE........")
    check_quality(df_products, "products")

    LOG.info("Task 4. SUMMARIZE BEFORE..........")
    summarize_numeric(df_products, "products")

    LOG.info("Task 5. PREPARE DATASET.........")
    df_products_prepared = prepare_products(df_products)

    LOG.info("Task 6. CHECK QUALITY AFTER PREPARATION........")
    check_quality(df_products_prepared, "products prepared")

    LOG.info("Task 7. SUMMARIZE AFTER PREPARATION........")
    summarize_numeric(df_products_prepared, "products prepared")

    LOG.info("Task 8. SAVE PREPARED DATASET........")
    save_prepared(df_products_prepared, PRODUCTS_PREPARED, "products")

    LOG.info("Workflow complete")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


if __name__ == "__main__":
    main()
