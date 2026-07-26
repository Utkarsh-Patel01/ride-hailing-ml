"""
Loads the raw NYC taxi trip duration dataset from disk into a validated,
correctly-typed pandas DataFrame.

This module is intentionally "dumb": it performs structural validation
(does the file exist, are the expected columns present, are datetimes
parsed) but makes no business-logic decisions about what counts as bad
data. Those decisions belong in src/data/clean_data.py (Phase 4).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.utils.config_loader import load_config, resolve_path

logger = logging.getLogger(__name__)

# The raw Kaggle schema this project was built against. Unlike the values
# in config.yaml, these are not tunable parameters — they're an intrinsic
# fact about the dataset's structure, so they're defined here rather than
# in YAML, where changing them would silently break every downstream
# assumption about what the raw data looks like.
EXPECTED_COLUMNS: List[str] = [
    "id",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
    "trip_duration",
]

# Columns with a small, fixed set of repeated values. Loading these as
# pandas 'category' dtype instead of the default object/string dtype cuts
# memory usage meaningfully on a 1.45M-row dataset, with zero downside
# since we never need arbitrary string operations on these columns.
_CATEGORICAL_COLUMNS: List[str] = ["vendor_id", "store_and_fwd_flag"]


class DataValidationError(Exception):
    """Raised when the raw dataset does not match the expected schema."""


def load_raw_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and structurally validate the raw taxi trip dataset.

    Args:
        csv_path: Optional override for the CSV location. Defaults to the
            path configured under paths.raw_data in config.yaml.

    Returns:
        A DataFrame with all EXPECTED_COLUMNS present, datetime columns
        parsed as datetime64, and low-cardinality columns loaded as
        categoricals.

    Raises:
        DataValidationError: If the file is missing or does not contain
            all expected columns.
    """
    config = load_config()

    resolved_path = csv_path or resolve_path(config["paths"]["raw_data"])

    if not resolved_path.exists():
        raise DataValidationError(
            f"Raw dataset not found at '{resolved_path}'. "
            "Download train.csv from the Kaggle NYC Taxi Trip Duration "
            "competition and place it there, as described in Phase 2."
        )

    logger.info("Loading raw dataset from %s", resolved_path)

    df = pd.read_csv(
        resolved_path,
        dtype={col: "category" for col in _CATEGORICAL_COLUMNS},
    )

    _validate_schema(df, resolved_path)

    datetime_columns = config["data"]["datetime_columns"]
    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col])

    logger.info(
        "Loaded %s rows, %s columns (%.1f MB in memory)",
        f"{len(df):,}",
        len(df.columns),
        df.memory_usage(deep=True).sum() / 1_048_576,
    )

    return df


def _validate_schema(df: pd.DataFrame, source_path: Path) -> None:
    """
    Verify that every column in EXPECTED_COLUMNS is present in the
    loaded DataFrame.

    Args:
        df: The freshly loaded DataFrame.
        source_path: The file path it was loaded from, included in the
            error message so the user knows exactly which file to check.

    Raises:
        DataValidationError: If any expected column is missing.
    """
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"'{source_path}' is missing expected column(s): {missing_columns}. "
            f"Expected schema: {EXPECTED_COLUMNS}. "
            "Confirm you downloaded the correct Kaggle dataset version."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataframe = load_raw_data()
    print(dataframe.head())
    print(dataframe.dtypes)