"""
Applies business-logic cleaning rules to the raw taxi trip dataset,
based on the findings quantified in notebooks/01_eda.py. Each rule is a
small function returning a boolean mask of rows to remove; rules run
sequentially and every step's removal count is captured in a report,
so a cleaning run is auditable rather than a silent chain of filters.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Tuple

import pandas as pd

from src.data.load_data import load_raw_data
from src.utils.config_loader import load_config, resolve_path

logger = logging.getLogger(__name__)

# Above this fraction of rows removed, something is more likely wrong
# with a rule (or the input file) than with the data itself.
_REMOVAL_SANITY_THRESHOLD = 0.05

CRITICAL_COLUMNS: List[str] = [
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
    "trip_duration",
]


def clean_trip_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Apply the full cleaning rule pipeline to a raw trip DataFrame.

    Args:
        df: Raw, schema-validated DataFrame from load_raw_data().

    Returns:
        A tuple of (cleaned DataFrame, report dict). The report contains
        rows_initial, rows_final, rows_removed_total, pct_removed, and
        a per-rule removal count for every step in the pipeline.
    """
    config = load_config()
    report: Dict[str, object] = {"rows_initial": len(df)}

    df = df.copy()

    steps: List[Tuple[str, Callable[[pd.DataFrame, dict], pd.Series]]] = [
        ("duplicate_ids", _flag_duplicate_ids),
        ("duration_mismatch", _flag_duration_mismatch),
        ("duration_out_of_bounds", _flag_duration_out_of_bounds),
        ("pickup_out_of_bounds", _flag_pickup_out_of_bounds),
        ("dropoff_out_of_bounds", _flag_dropoff_out_of_bounds),
        ("invalid_passenger_count", _flag_invalid_passenger_count),
        ("missing_critical_values", _flag_missing_critical_values),
    ]

    for step_name, flag_fn in steps:
        bad_mask = flag_fn(df, config)
        removed = int(bad_mask.sum())
        report[step_name] = removed
        if removed:
            logger.info("Removing %s rows failing '%s'", f"{removed:,}", step_name)
        df = df.loc[~bad_mask].reset_index(drop=True)

    report["rows_final"] = len(df)
    report["rows_removed_total"] = report["rows_initial"] - report["rows_final"]
    report["pct_removed"] = report["rows_removed_total"] / report["rows_initial"]

    if report["pct_removed"] > _REMOVAL_SANITY_THRESHOLD:
        logger.warning(
            "Cleaning removed %.2f%% of rows - unusually high (threshold %.0f%%). "
            "Verify this isn't caused by an overly strict rule before proceeding to Phase 5.",
            report["pct_removed"] * 100,
            _REMOVAL_SANITY_THRESHOLD * 100,
        )

    logger.info(
        "Cleaning complete: %s -> %s rows (%.2f%% removed)",
        f"{report['rows_initial']:,}",
        f"{report['rows_final']:,}",
        report["pct_removed"] * 100,
    )

    return df, report


def _flag_duplicate_ids(df: pd.DataFrame, config: dict) -> pd.Series:
    """Flag rows sharing a trip id with an earlier row (keeps the first)."""
    return df.duplicated(subset="id", keep="first")


def _flag_duration_mismatch(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Flag rows where trip_duration disagrees with
    (dropoff_datetime - pickup_datetime) by more than a 1-second
    rounding tolerance.

    This is a structural invariant, not a business judgment call: the
    dataset defines trip_duration as exactly this difference, so a
    mismatch means the row is corrupted, not merely unusual.
    """
    computed_seconds = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()
    return (computed_seconds - df["trip_duration"]).abs() > 1


def _flag_duration_out_of_bounds(df: pd.DataFrame, config: dict) -> pd.Series:
    """Flag trips shorter than min_trip_duration_seconds or longer than max."""
    min_s = config["data"]["min_trip_duration_seconds"]
    max_s = config["data"]["max_trip_duration_seconds"]
    return ~df["trip_duration"].between(min_s, max_s)


def _flag_pickup_out_of_bounds(df: pd.DataFrame, config: dict) -> pd.Series:
    """Flag trips whose pickup coordinates fall outside the NYC bounding box."""
    lat_lo, lat_hi = config["data"]["nyc_lat_bounds"]
    lon_lo, lon_hi = config["data"]["nyc_lon_bounds"]
    return (
        ~df["pickup_latitude"].between(lat_lo, lat_hi)
        | ~df["pickup_longitude"].between(lon_lo, lon_hi)
    )


def _flag_dropoff_out_of_bounds(df: pd.DataFrame, config: dict) -> pd.Series:
    """Flag trips whose dropoff coordinates fall outside the NYC bounding box."""
    lat_lo, lat_hi = config["data"]["nyc_lat_bounds"]
    lon_lo, lon_hi = config["data"]["nyc_lon_bounds"]
    return (
        ~df["dropoff_latitude"].between(lat_lo, lat_hi)
        | ~df["dropoff_longitude"].between(lon_lo, lon_hi)
    )


def _flag_invalid_passenger_count(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Flag trips with 0 passengers (a taxi cannot dispatch empty - a data
    error) or more than max_passenger_count (implausible for a taxi).
    """
    max_passengers = config["data"]["max_passenger_count"]
    return (df["passenger_count"] == 0) | (df["passenger_count"] > max_passengers)


def _flag_missing_critical_values(df: pd.DataFrame, config: dict) -> pd.Series:
    """Flag rows with a missing value in any column this project cannot proceed without."""
    return df[CRITICAL_COLUMNS].isna().any(axis=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    raw_df = load_raw_data()
    cleaned_df, cleaning_report = clean_trip_data(raw_df)

    interim_dir = resolve_path(config["paths"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / "train_cleaned.parquet"
    cleaned_df.to_parquet(output_path, index=False)

    logger.info("Saved cleaned data to %s", output_path)
    print(json.dumps(cleaning_report, indent=2))