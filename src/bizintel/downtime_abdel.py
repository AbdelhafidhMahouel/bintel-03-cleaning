"""downtime_abdel.py - custom Phase 5 project.

A custom application of the example's load/aggregate/visualize/log
pattern to manufacturing downtime data instead of sales data.

Author: Abdelhafidh Mahouel
Date: 2026-07

Process:
    - Load raw downtime records.
    - Aggregate total downtime by production line and by cause.
    - Visualize both breakdowns.
    - Log a summary, including data quality notes.

Data Source:
- data/raw/downtime_abdel.csv

Terminal command to run this file from the root project folder:

uv run python -m bizintel.downtime_abdel

Background:
    In my work as an engineering team lead in manufacturing, tracking
    why and where production time is lost is a core part of improving
    line performance (OEE). This module applies the same load, clean,
    aggregate, log, and visualize pattern used in app_case.py to a
    production downtime dataset instead of sales data.
"""

# === DECLARE IMPORTS ===

from pathlib import Path
from typing import Final

from datafun_toolkit.logger import log_path
import matplotlib.pyplot as plt
import pandas as pd

from bizintel.utils_data import load_data
from bizintel.utils_logger import LOG, log_header
from bizintel.utils_viz import plot_bar

# === DECLARE GLOBAL CONSTANTS ===

DATA_RAW: Final[Path] = Path("data/raw")
DOWNTIME_FILE: Final[Path] = DATA_RAW / "downtime_abdel.csv"


# === Section 2.1 DEFINE A DOWNTIME BY LINE FUNCTION ===


def downtime_by_line(df_downtime: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total downtime minutes by production line.

    WHY: Identifying which production line loses the most time is the
    first step in prioritizing improvement efforts (OEE, changeover
    reduction, preventive maintenance scheduling).

    Args:
        df_downtime: Downtime DataFrame with ProductionLine and
            DowntimeMinutes columns.

    Returns:
        DataFrame with ProductionLine and DowntimeMinutes columns,
        sorted by DowntimeMinutes descending.
    """
    LOG.info("Aggregating downtime by production line")

    df = df_downtime.copy()
    df["DowntimeMinutes"] = pd.to_numeric(df["DowntimeMinutes"], errors="coerce")

    # Exclude invalid (negative or missing) downtime values rather than
    # silently letting them distort the totals.
    total_rows = len(df)
    is_invalid = df["DowntimeMinutes"].isna() | (df["DowntimeMinutes"] < 0)
    invalid_count = int(is_invalid.sum())
    if invalid_count > 0:
        pct = (invalid_count / total_rows) * 100
        LOG.warning(
            f"  Excluding {invalid_count} of {total_rows} downtime rows "
            f"({pct:.1f}%) with missing or negative DowntimeMinutes"
        )
    df = df.loc[~is_invalid].copy()

    df["ProductionLine"] = df["ProductionLine"].str.strip().str.title()

    grouped = pd.Series(df.groupby("ProductionLine")["DowntimeMinutes"].sum())
    df_line = grouped.reset_index().sort_values("DowntimeMinutes", ascending=False)

    top_line = str(df_line.iloc[0]["ProductionLine"])
    top_minutes = float(df_line.iloc[0]["DowntimeMinutes"])
    LOG.info(f"  Highest downtime line: {top_line} ({top_minutes:,.1f} minutes)")

    return df_line


# === Section 2.2 DEFINE A DOWNTIME BY CAUSE FUNCTION ===


def downtime_by_cause(df_downtime: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total downtime minutes by root cause category.

    WHY: Root cause breakdowns drive corrective action priorities,
    for example whether to invest in preventive maintenance, faster
    changeovers, or supply chain reliability.

    Args:
        df_downtime: Downtime DataFrame with DowntimeCause and
            DowntimeMinutes columns.

    Returns:
        DataFrame with DowntimeCause and DowntimeMinutes columns,
        sorted by DowntimeMinutes descending.
    """
    LOG.info("Aggregating downtime by cause")

    df = df_downtime.copy()
    df["DowntimeMinutes"] = pd.to_numeric(df["DowntimeMinutes"], errors="coerce")

    is_invalid = df["DowntimeMinutes"].isna() | (df["DowntimeMinutes"] < 0)
    df = df.loc[~is_invalid].copy()

    # Exclude rows with a missing cause; log how many were dropped.
    is_missing_cause = df["DowntimeCause"].isna() | (
        df["DowntimeCause"].astype(str).str.strip() == ""
    )
    missing_count = int(is_missing_cause.sum())
    if missing_count > 0:
        LOG.warning(f"  Excluding {missing_count} rows with missing DowntimeCause")
    df = df.loc[~is_missing_cause].copy()

    df["DowntimeCause"] = df["DowntimeCause"].str.strip().str.title()

    grouped = pd.Series(df.groupby("DowntimeCause")["DowntimeMinutes"].sum())
    df_cause = grouped.reset_index().sort_values("DowntimeMinutes", ascending=False)

    top_cause = str(df_cause.iloc[0]["DowntimeCause"])
    top_minutes = float(df_cause.iloc[0]["DowntimeMinutes"])
    LOG.info(f"  Top downtime cause: {top_cause} ({top_minutes:,.1f} minutes)")

    return df_cause


# === Section 2.3 DEFINE A SUMMARIZE FUNCTION ===


def summarize(df_downtime: pd.DataFrame) -> None:
    """Log a brief summary of the downtime dataset.

    Args:
        df_downtime: Downtime DataFrame.

    Returns:
        None
    """
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")

    rows, cols = df_downtime.shape
    LOG.info(f"Downtime records: {rows} rows, {cols} columns")

    LOG.info("========================")
    LOG.info("ANALYST NOTES:")
    LOG.info("Note any data quality issues (missing/negative durations,")
    LOG.info("inconsistent casing in Shift or DowntimeCause).")
    LOG.info("Cleaning is planned in a future project phase.")
    LOG.info("========================")


# === DEFINE THE MAIN FUNCTION ===


def main() -> None:
    """Main function to run the downtime analysis workflow."""

    log_header(LOG, "BI")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    log_path(LOG, "Raw data: ", DATA_RAW)
    log_path(LOG, "Downtime:", DOWNTIME_FILE)

    LOG.info("CALL a function to load the downtime dataset.............")
    df_downtime = load_data(DOWNTIME_FILE, "downtime")

    LOG.info("CALL a function to get downtime by production line........")
    df_line = downtime_by_line(df_downtime)

    LOG.info("CALL a function to plot downtime by production line........")
    plot_bar(
        df=df_line,
        x="ProductionLine",
        y="DowntimeMinutes",
        title="Total Downtime by Production Line",
        xlabel="Production Line",
        ylabel="Total Downtime (minutes)",
        palette="Reds_d",
    )

    LOG.info("CALL a function to get downtime by cause........")
    df_cause = downtime_by_cause(df_downtime)

    LOG.info("CALL a function to plot downtime by cause........")
    plot_bar(
        df=df_cause,
        x="DowntimeCause",
        y="DowntimeMinutes",
        title="Total Downtime by Cause",
        xlabel="Downtime Cause",
        ylabel="Total Downtime (minutes)",
        palette="Purples_d",
    )

    LOG.info("CALL a function to summarize the dataset........")
    summarize(df_downtime)

    LOG.info("CALL a function to show charts........")
    plt.show()

    LOG.info("Workflow complete")
    LOG.info("CLOSE chart windows to continue.")
    LOG.info("Terminate this process with CTRL+c as needed.")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


if __name__ == "__main__":
    main()
