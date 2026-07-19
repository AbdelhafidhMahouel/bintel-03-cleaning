"""prepare_sales_data.py - clean and prepare sales data.

Cleans sales_data.csv, including the custom DiscountPercent and
PaymentType columns added in the D3.1 assignment.

Author: Abdelhafidh Mahouel
Date: 2026-07

Process:
    - Load raw sales data.
    - Convert SaleDate to datetime and SaleAmount to numeric.
    - Remove rows with missing date or amount.
    - Check CustomerID and ProductID foreign key integrity against
      the raw customers and products files.
    - Clean DiscountPercent (numeric) and PaymentType (category).
    - Remove duplicate rows.
    - Save prepared data to data/prepared/.

Data Source:
- data/raw/sales_data.csv
- data/raw/customers_data.csv (for CustomerID validation only)
- data/raw/products_data.csv (for ProductID validation only)

Output:
- data/prepared/sales_data_prepared.csv

Terminal command to run this file from the root project folder:

uv run python -m analytics_project.data_prep.prepare_sales_data

OBS: This module loads the raw customers and products files directly
(rather than depending on the other two prepare_*.py scripts having
already run) so it can be run independently, in any order.
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

SALES_FILE: Final[Path] = DATA_RAW / "sales_data.csv"
CUSTOMERS_FILE: Final[Path] = DATA_RAW / "customers_data.csv"
PRODUCTS_FILE: Final[Path] = DATA_RAW / "products_data.csv"

SALES_PREPARED: Final[Path] = DATA_PREPARED / "sales_data_prepared.csv"

# Business rule: a discount between 0% and 50% is realistic for this
# business. Anything outside that range is treated as a data entry error.
MIN_DISCOUNT_PERCENT: Final[float] = 0.0
MAX_DISCOUNT_PERCENT: Final[float] = 50.0

# PaymentType has camelCase canonical spellings (e.g. "CreditCard"), so
# str.title() alone would break them (turning "creditcard" into
# "Creditcard" instead of "CreditCard"). We normalize explicitly instead.
PAYMENT_TYPE_CANONICAL: Final[dict[str, str]] = {
    "creditcard": "CreditCard",
    "debitcard": "DebitCard",
    "cash": "Cash",
    "paypal": "PayPal",
    "giftcard": "GiftCard",
}


def prepare_sales(
    df: pd.DataFrame,
    valid_customer_ids: set[int],
    valid_product_ids: set[int],
) -> pd.DataFrame:
    """Clean and prepare the sales DataFrame.

    WHY: Sales data links customers and products. Invalid foreign keys
    mean orphaned records in the warehouse that cannot be joined to
    any customer or product. The custom DiscountPercent and
    PaymentType columns also need their own validation.

    Args:
        df: Raw sales DataFrame.
        valid_customer_ids: Set of valid CustomerIDs from customers table.
        valid_product_ids: Set of valid ProductIDs from products table.

    Returns:
        Cleaned sales DataFrame.
    """
    LOG.info("Preparing sales data")

    df = df.copy()

    LOG.info("Sales Prep 1. Convert SaleDate to datetime")
    df["SaleDate"] = pd.to_datetime(df["SaleDate"], errors="coerce")
    bad_dates: int = int(df["SaleDate"].isna().sum())
    if bad_dates > 0:
        LOG.warning(f"  {bad_dates} row(s) with invalid SaleDate - will be removed")

    LOG.info("Sales Prep 2. Convert SaleAmount to numeric")
    df["SaleAmount"] = pd.to_numeric(df["SaleAmount"], errors="coerce")
    bad_amounts: int = int(df["SaleAmount"].isna().sum())
    if bad_amounts > 0:
        LOG.warning(f"  {bad_amounts} row(s) with invalid SaleAmount - will be removed")

    LOG.info("Sales Prep 3. Remove rows with missing date or amount")
    before: int = df.shape[0]
    df = df.dropna(subset=["SaleDate", "SaleAmount"])
    after: int = df.shape[0]
    LOG.info(f"  Rows before: {before}")
    LOG.info(f"  Rows after: {after}")
    LOG.info(f"  Removed {before - after} row(s) with missing date or amount")

    LOG.info("Sales Prep 4. Check CustomerID foreign key integrity")
    invalid_customers = ~df["CustomerID"].isin(valid_customer_ids)
    invalid_customer_count: int = int(invalid_customers.sum())
    if invalid_customer_count > 0:
        LOG.warning(
            f"  {invalid_customer_count} row(s) with CustomerID "
            "not found in customers table - will be removed"
        )
        df = df[~invalid_customers]

    LOG.info("Sales Prep 5. Check ProductID foreign key integrity")
    invalid_products = ~df["ProductID"].isin(valid_product_ids)
    invalid_product_count: int = int(invalid_products.sum())
    if invalid_product_count > 0:
        LOG.warning(
            f"  {invalid_product_count} row(s) with ProductID "
            "not found in products table - will be removed"
        )
        df = df[~invalid_products]

    LOG.info("Sales Prep 6. Clean DiscountPercent (numeric)")
    df["DiscountPercent"] = pd.to_numeric(df["DiscountPercent"], errors="coerce")

    is_invalid_discount = (
        df["DiscountPercent"].isna()
        | (df["DiscountPercent"] < MIN_DISCOUNT_PERCENT)
        | (df["DiscountPercent"] > MAX_DISCOUNT_PERCENT)
    )
    invalid_discount_count: int = int(is_invalid_discount.sum())
    if invalid_discount_count > 0:
        LOG.warning(
            f"  {invalid_discount_count} row(s) with missing or "
            f"unrealistic (outside {MIN_DISCOUNT_PERCENT}-{MAX_DISCOUNT_PERCENT}%) "
            "DiscountPercent - will be removed"
        )
    df = df.loc[~is_invalid_discount].copy()

    LOG.info("Sales Prep 7. Clean PaymentType (category)")
    normalized_key = df["PaymentType"].astype(str).str.strip().str.lower()
    df["PaymentType"] = normalized_key.map(PAYMENT_TYPE_CANONICAL)

    is_missing_payment = df["PaymentType"].isna()
    missing_payment_count: int = int(is_missing_payment.sum())
    if missing_payment_count > 0:
        LOG.warning(
            f"  {missing_payment_count} row(s) with missing or unrecognized "
            "PaymentType - will be removed"
        )
    df = df.loc[~is_missing_payment].copy()

    LOG.info("Sales Prep 8. Remove duplicate rows")
    before_count: int = df.shape[0]
    df = df.drop_duplicates()
    after_count: int = df.shape[0]
    LOG.info(f"  Rows before: {before_count}")
    LOG.info(f"  Rows after: {after_count}")
    LOG.info(f"  Removed {before_count - after_count} duplicate row(s)")
    LOG.info(f"  Sales prepared: {df.shape[0]} rows")
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
    """Main function to run the sales data preparation logic."""
    log_header(LOG, "BI")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    LOG.info("Task 1. LOAD. Call a function to load each dataset......")
    df_sales = load_data(SALES_FILE, "sales")
    df_customers = load_data(CUSTOMERS_FILE, "customers")
    df_products = load_data(PRODUCTS_FILE, "products")

    LOG.info("Task 2. INSPECT. Call a function to inspect the sales dataset...")
    inspect_basic(df_sales, "sales")

    LOG.info("Task 3. CHECK QUALITY BEFORE........")
    check_quality(df_sales, "sales")

    LOG.info("Task 4. SUMMARIZE BEFORE..........")
    summarize_numeric(df_sales, "sales")

    LOG.info("Task 5. PREPARE DATASET.........")
    valid_customer_ids: set[int] = set(df_customers["CustomerID"])
    valid_product_ids: set[int] = set(df_products["ProductID"])

    df_sales_prepared = prepare_sales(df_sales, valid_customer_ids, valid_product_ids)

    LOG.info("Task 6. CHECK QUALITY AFTER PREPARATION........")
    check_quality(df_sales_prepared, "sales prepared")

    LOG.info("Task 7. SUMMARIZE AFTER PREPARATION........")
    summarize_numeric(df_sales_prepared, "sales prepared")

    LOG.info("Task 8. SAVE PREPARED DATASET........")
    save_prepared(df_sales_prepared, SALES_PREPARED, "sales")

    LOG.info("Workflow complete")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


if __name__ == "__main__":
    main()
