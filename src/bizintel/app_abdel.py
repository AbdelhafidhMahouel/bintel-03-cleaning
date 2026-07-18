"""app_abdel.py - custom modification of the example.

An extension of the example loading and visualizing raw business data.
Adds a new breakdown by PaymentType, a custom attribute added to this
project's sales data, alongside the original region and category analysis.

Author: Abdelhafidh Mahouel
Date: 2026-07

Process:
    - Load raw CSV data files.
    - Visualize sales by region, product category, and payment type.
    - Log a summary of findings, including data quality notes.

Data Source:
- data/raw/customers_data.csv
- data/raw/products_data.csv
- data/raw/sales_data.csv

Terminal command to run this file from the root project folder:

uv run python -m bizintel.app_abdel

Modification Summary:
    - Added sales_by_payment_type(): aggregates total sales by the
      custom PaymentType column added to sales_data.csv.
    - Added a third chart: Total Sales by Payment Type.
    - Added observability: logs how many sales rows were excluded
      from the payment type analysis due to missing PaymentType values.
"""

# === DECLARE IMPORTS (bring in free code from elsewhere) ===

from pathlib import Path
from typing import Final

from datafun_toolkit.logger import log_path
import matplotlib.pyplot as plt
import pandas as pd

from bizintel.utils_data import (
    load_data,
)
from bizintel.utils_logger import LOG, log_header
from bizintel.utils_viz import plot_bar

# === DECLARE GLOBAL CONSTANTS AND CONFIGURATION ===

# Raw data folder path (relative to the root project folder).
DATA_RAW: Final[Path] = Path("data/raw")

# The three raw data files for the smart sales project.
CUSTOMERS_FILE: Final[Path] = DATA_RAW / "customers_data.csv"
PRODUCTS_FILE: Final[Path] = DATA_RAW / "products_data.csv"
SALES_FILE: Final[Path] = DATA_RAW / "sales_data.csv"


# === Section 2. Define Reusable Functions ===

# === Section 2.1 DEFINE A SALES BY REGION FUNCTION ===


def sales_by_region(
    df_customers: pd.DataFrame,
    df_sales: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate total sales amount by customer region.

    Args:
        df_customers: Customers DataFrame with CustomerID and Region columns.
        df_sales: Sales DataFrame with CustomerID and SaleAmount columns.

    Returns:
        DataFrame with Region and SaleAmount columns, sorted by SaleAmount.
    """
    LOG.info("Aggregating sales by region")

    df_sales = df_sales.copy()
    df_sales["SaleAmount"] = pd.to_numeric(df_sales["SaleAmount"], errors="coerce")

    df_merged: pd.DataFrame = df_sales.merge(
        df_customers[["CustomerID", "Region"]],
        on="CustomerID",
        how="left",
    )

    df_merged["Region"] = df_merged["Region"].str.strip().str.title()

    grouped: pd.Series = pd.Series(df_merged.groupby("Region")["SaleAmount"].sum())

    df_region: pd.DataFrame = grouped.reset_index().sort_values(
        "SaleAmount", ascending=False
    )

    top_region: str = str(df_region.iloc[0]["Region"])
    top_sales: float = float(df_region.iloc[0]["SaleAmount"])

    LOG.info(f"  Top region: {top_region} (${top_sales:,.2f})")
    LOG.info("Returning DataFrame with total sales by region")
    return df_region


# === Section 2.2 DEFINE A SALES BY CATEGORY FUNCTION ===


def sales_by_category(
    df_products: pd.DataFrame,
    df_sales: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate total sales amount by product category.

    WHY: Product category is another key business dimension.
    Understanding which categories drive revenue helps prioritize
    inventory, marketing, and purchasing decisions.

    Args:
        df_products: Products DataFrame with ProductID and Category columns.
        df_sales: Sales DataFrame with ProductID and SaleAmount columns.

    Returns:
        DataFrame with Category and SaleAmount columns, sorted by SaleAmount.
    """
    LOG.info("Aggregating sales by product category")

    df_sales = df_sales.copy()
    df_sales["SaleAmount"] = pd.to_numeric(df_sales["SaleAmount"], errors="coerce")

    df_merged: pd.DataFrame = df_sales.merge(
        df_products[["ProductID", "Category"]],
        on="ProductID",
        how="left",
    )

    grouped: pd.Series = pd.Series(df_merged.groupby("Category")["SaleAmount"].sum())

    df_category: pd.DataFrame = grouped.reset_index().sort_values(
        "SaleAmount", ascending=False
    )

    top_category: str = str(df_category.iloc[0]["Category"])
    top_sales: float = float(df_category.iloc[0]["SaleAmount"])

    LOG.info(f"  Top category: {top_category} (${top_sales:,.2f})")
    return df_category


# === Section 2.3 DEFINE A SALES BY PAYMENT TYPE FUNCTION (NEW) ===


def sales_by_payment_type(
    df_sales: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate total sales amount by payment type.

    WHY: PaymentType is a custom attribute added to this project's
    sales data (see D3.1 Data Collection). Understanding which payment
    methods drive the most revenue can inform decisions about payment
    processing fees, fraud monitoring, or checkout experience priorities.

    OBSERVABILITY: Some sales rows have a missing or blank PaymentType
    value (an intentional data quality issue introduced in the raw data).
    Those rows are excluded from this analysis, and the number excluded
    is logged so the gap is visible rather than silently dropped.

    Args:
        df_sales: Sales DataFrame with PaymentType and SaleAmount columns.

    Returns:
        DataFrame with PaymentType and SaleAmount columns, sorted by
        SaleAmount descending.
    """
    LOG.info("Aggregating sales by payment type")

    df_sales = df_sales.copy()
    df_sales["SaleAmount"] = pd.to_numeric(df_sales["SaleAmount"], errors="coerce")

    total_rows: int = len(df_sales)

    # Identify rows with missing or blank PaymentType before dropping them,
    # so the exclusion is visible in the log rather than a silent data loss.
    is_missing: pd.Series = df_sales["PaymentType"].isna() | (
        df_sales["PaymentType"].astype(str).str.strip() == ""
    )
    missing_count: int = int(is_missing.sum())

    if missing_count > 0:
        pct_missing: float = (missing_count / total_rows) * 100
        LOG.warning(
            f"  Excluding {missing_count} of {total_rows} sales rows "
            f"({pct_missing:.1f}%) with missing PaymentType"
        )

    df_valid: pd.DataFrame = df_sales.loc[~is_missing].copy()

    grouped: pd.Series = pd.Series(df_valid.groupby("PaymentType")["SaleAmount"].sum())

    df_payment: pd.DataFrame = grouped.reset_index().sort_values(
        "SaleAmount", ascending=False
    )

    top_payment: str = str(df_payment.iloc[0]["PaymentType"])
    top_sales: float = float(df_payment.iloc[0]["SaleAmount"])

    LOG.info(f"  Top payment type: {top_payment} (${top_sales:,.2f})")
    return df_payment


# === Section 2.4 DEFINE A SUMMARIZE FUNCTION ===


def summarize(
    df_customers: pd.DataFrame,
    df_products: pd.DataFrame,
    df_sales: pd.DataFrame,
) -> None:
    """Log a brief summary of all three datasets.

    Args:
        df_customers: Customers DataFrame.
        df_products: Products DataFrame.
        df_sales: Sales DataFrame.

    Returns:
        None
    """
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")

    cust_rows: int = df_customers.shape[0]
    cust_cols: int = df_customers.shape[1]

    prod_rows: int = df_products.shape[0]
    prod_cols: int = df_products.shape[1]

    sale_rows: int = df_sales.shape[0]
    sale_cols: int = df_sales.shape[1]

    LOG.info(f"Customers:  {cust_rows} rows, {cust_cols} columns")
    LOG.info(f"Products:   {prod_rows} rows, {prod_cols} columns")
    LOG.info(f"Sales:      {sale_rows} rows, {sale_cols} columns")

    LOG.info("========================")
    LOG.info("ANALYST NOTES:")
    LOG.info("Note any data quality issues.")
    LOG.info("We will clean data later.")
    LOG.info("========================")


# === DEFINE THE MAIN FUNCTION (WHERE THE MAGIC HAPPENS) ===


def main() -> None:
    """Main function to run the extended BI logic.

    This is where the main logic starts when this script is run.
    Extends the original example with a payment type breakdown.
    """

    log_header(LOG, "BI")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    log_path(LOG, "Raw data: ", DATA_RAW)
    log_path(LOG, "Customers:", CUSTOMERS_FILE)
    log_path(LOG, "Products: ", PRODUCTS_FILE)
    log_path(LOG, "Sales:    ", SALES_FILE)

    LOG.info("CALL a function to load each dataset.............")
    df_customers = load_data(CUSTOMERS_FILE, "customers")
    df_products = load_data(PRODUCTS_FILE, "products")
    df_sales = load_data(SALES_FILE, "sales")

    LOG.info("CALL a function to get sales by region........")
    df_region = sales_by_region(df_customers, df_sales)

    LOG.info("CALL a function to plot sales by region........")
    plot_bar(
        df=df_region,
        x="Region",
        y="SaleAmount",
        title="Total Sales by Region",
        xlabel="Region",
        ylabel="Total Sales Amount ($)",
        palette="Blues_d",
    )

    LOG.info("CALL a function to get sales by product category........")
    df_category = sales_by_category(df_products, df_sales)

    LOG.info("CALL a function to plot sales by product category........")
    plot_bar(
        df=df_category,
        x="Category",
        y="SaleAmount",
        title="Total Sales by Product Category",
        xlabel="Category",
        ylabel="Total Sales Amount ($)",
        palette="Greens_d",
    )

    LOG.info("CALL a function to get sales by payment type........")
    df_payment = sales_by_payment_type(df_sales)

    LOG.info("CALL a function to plot sales by payment type........")
    plot_bar(
        df=df_payment,
        x="PaymentType",
        y="SaleAmount",
        title="Total Sales by Payment Type",
        xlabel="Payment Type",
        ylabel="Total Sales Amount ($)",
        palette="Oranges_d",
    )

    LOG.info("CALL a function to summarize the datasets........")
    summarize(df_customers, df_products, df_sales)

    LOG.info("CALL a function to show charts........")
    plt.show()

    LOG.info("Workflow complete")
    LOG.info("CLOSE chart windows to continue.")
    LOG.info("Terminate this process with CTRL+c as needed.")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
