"""
Engineers time-based features from pickup_datetime. Consumed by all
three pipelines (trip duration, demand forecasting, anomaly detection)
via src/features/build_features.py.

This module is intentionally pure: given a DataFrame, it returns a new
DataFrame with additional columns. It performs no file I/O itself, so
it stays trivially testable and reusable from a notebook, a training
script, or the FastAPI inference path alike.
"""

from __future__ import annotations

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

TEMPORAL_FEATURE_COLUMNS: List[str] = [
    "pickup_hour",
    "pickup_weekday",
    "pickup_is_weekend",
    "pickup_is_rush_hour",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "pickup_weekday_sin",
    "pickup_weekday_cos",
]


def cyclic_encode(series: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    """
    Encode a periodic integer series (e.g. hour-of-day, day-of-week) as
    a (sin, cos) pair on the unit circle, so adjacent values at the
    wrap-around boundary (23 -> 0, Sunday -> Monday) stay close in
    feature space instead of appearing maximally distant to a linear
    model.

    Shared by both trip-level temporal features and the hourly demand
    dataset (Phase 7) - extracted here once a second consumer needed
    identical logic, rather than duplicating the sin/cos formula again.

    Args:
        series: Integer series of a periodic quantity.
        period: The full cycle length (24 for hour, 7 for weekday).

    Returns:
        Tuple of (sin_component, cos_component), each a pandas Series
        aligned to the input's index.
    """
    radians = 2 * np.pi * series / period
    return np.sin(radians), np.cos(radians)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features derived from pickup_datetime.

    Args:
        df: DataFrame containing a 'pickup_datetime' column with
            datetime64 dtype (guaranteed by src.data.load_data).

    Returns:
        A copy of df with TEMPORAL_FEATURE_COLUMNS appended. The input
        DataFrame is not mutated.

    Raises:
        KeyError: If 'pickup_datetime' is not present in df.
    """
    if "pickup_datetime" not in df.columns:
        raise KeyError(
            "add_temporal_features requires a 'pickup_datetime' column. "
            "Confirm this DataFrame came from src.data.load_data.load_raw_data()."
        )

    config = load_config()
    df = df.copy()

    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday  # Monday=0 ... Sunday=6
    df["pickup_is_weekend"] = df["pickup_weekday"].isin([5, 6]).astype(int)
    df["pickup_is_rush_hour"] = _compute_rush_hour_flag(df["pickup_hour"], config)

    # Cyclic encoding: maps hour/weekday onto a circle so that adjacent
    # times (23:00 -> 00:00, Sunday -> Monday) stay close in feature
    # space instead of appearing maximally distant to a linear model.
    df["pickup_hour_sin"], df["pickup_hour_cos"] = cyclic_encode(df["pickup_hour"], period=24)
    df["pickup_weekday_sin"], df["pickup_weekday_cos"] = cyclic_encode(df["pickup_weekday"], period=7)

    logger.info("Added %d temporal features", len(TEMPORAL_FEATURE_COLUMNS))
    return df


def _compute_rush_hour_flag(hour_series: pd.Series, config: dict) -> pd.Series:
    """
    Flag pickup hours falling inside either configured rush-hour window.

    Args:
        hour_series: The pickup_hour column (integers 0-23).
        config: Loaded project config, read for the rush-hour windows
            defined under temporal_features in config.yaml.

    Returns:
        A 0/1 integer Series, 1 where the hour falls in either window.
    """
    morning_start, morning_end = config["temporal_features"]["rush_hour_morning"]
    evening_start, evening_end = config["temporal_features"]["rush_hour_evening"]

    is_morning_rush = hour_series.between(morning_start, morning_end)
    is_evening_rush = hour_series.between(evening_start, evening_end)

    return (is_morning_rush | is_evening_rush).astype(int)


if __name__ == "__main__":
    from src.utils.config_loader import resolve_path

    logging.basicConfig(level=logging.INFO)

    config = load_config()
    interim_dir = resolve_path(config["paths"]["interim_dir"])
    df = pd.read_parquet(interim_dir / "train_cleaned.parquet")

    df = add_temporal_features(df)
    print(df[TEMPORAL_FEATURE_COLUMNS].describe())