"""
Transforms trip-level data (with pickup zones assigned in Phase 6) into
a complete hourly zone-level demand dataset: one row per (zone, hour)
with pickup_count as the forecasting target, plus lag and rolling-
window features computed with strict temporal-leakage prevention.

See Phase 7's write-up for why the "complete grid" step and the
shift-before-rolling ordering are both required, not stylistic choices.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from src.features.temporal_features import cyclic_encode
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

DEMAND_COLUMN = "pickup_count"


def aggregate_hourly_demand(df: pd.DataFrame, n_zones: int) -> pd.DataFrame:
    """
    Aggregate trip-level data into a complete zone x hour grid of
    pickup counts, including hours with zero pickups.

    Args:
        df: Trip-level DataFrame with 'pickup_datetime' and
            'pickup_zone' columns (output of Phase 6's zone assignment).
        n_zones: Total number of zones (config.zones.n_clusters), used
            to build the complete zone axis even for a zone that
            happens to have zero trips in the entire dataset.

    Returns:
        DataFrame with columns ['pickup_zone', 'hour', 'pickup_count'],
        one row per zone per hour, sorted by zone then hour.
    """
    working = df.copy()
    working["hour"] = working["pickup_datetime"].dt.floor("h")

    observed_counts = (
        working.groupby(["pickup_zone", "hour"], observed=True)
        .size()
        .rename(DEMAND_COLUMN)
        .reset_index()
    )
    # Cast away the categorical dtype before reindexing: mixing a
    # categorical zone column against a plain-integer target index in
    # reindex() silently produces an all-NaN result on some pandas
    # versions instead of the zero-filled grid we actually want.
    observed_counts["pickup_zone"] = observed_counts["pickup_zone"].astype(int)

    full_hour_range = pd.date_range(
        start=working["hour"].min(), end=working["hour"].max(), freq="h"
    )
    full_index = pd.MultiIndex.from_product(
        [range(n_zones), full_hour_range], names=["pickup_zone", "hour"]
    )

    complete = (
        observed_counts.set_index(["pickup_zone", "hour"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    complete = complete.sort_values(["pickup_zone", "hour"]).reset_index(drop=True)

    logger.info(
        "Aggregated %s trips into %s zone-hour rows (%d zones x %d hours)",
        f"{len(df):,}", f"{len(complete):,}", n_zones, len(full_hour_range),
    )
    return complete


def add_lag_features(df: pd.DataFrame, lag_hours: List[int]) -> pd.DataFrame:
    """
    Add lagged pickup_count features, computed independently per zone.

    Args:
        df: Complete zone-hour grid from aggregate_hourly_demand,
            sorted by pickup_zone then hour.
        lag_hours: Hours to look back, e.g. [1, 2, 3, 24].

    Returns:
        A copy of df with one 'demand_lag_{h}h' column per entry in
        lag_hours. Rows at the start of each zone's series without
        enough history will contain NaN - expected, and handled by
        drop_insufficient_history below rather than being filled with
        0, which would fabricate a fake "no demand" signal where the
        truth is simply "not observed yet".
    """
    df = df.copy()
    grouped = df.groupby("pickup_zone")[DEMAND_COLUMN]

    for lag in lag_hours:
        df[f"demand_lag_{lag}h"] = grouped.shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame, rolling_windows: List[int]) -> pd.DataFrame:
    """
    Add rolling-mean demand features, computed per zone, strictly
    excluding the current hour.

    THIS IS THE TEMPORAL LEAKAGE GUARD: a naive `.rolling(window).mean()`
    includes the current row in its own average by default, which would
    hand the model a feature partially derived from the exact value
    it's trying to predict - a feature that looks excellent offline and
    cannot be computed the same way in production, since the current
    hour's true demand is precisely the unknown quantity. Calling
    `.shift(1)` BEFORE `.rolling(...)` ensures the window only ever
    looks at hours strictly before the one being predicted.

    Args:
        df: Complete zone-hour grid, already sorted by zone then hour.
        rolling_windows: Window sizes in hours, e.g. [3, 6, 24].

    Returns:
        A copy of df with one 'demand_rolling_mean_{w}h' column per
        entry in rolling_windows.
    """
    df = df.copy()
    grouped = df.groupby("pickup_zone")[DEMAND_COLUMN]

    for window in rolling_windows:
        df[f"demand_rolling_mean_{window}h"] = (
            grouped.shift(1).rolling(window=window, min_periods=1).mean()
        )

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour-of-day and day-of-week cyclic features to the hourly grid,
    reusing the same cyclic_encode helper the trip-level temporal
    features use, so both pipelines encode circular time identically.

    Args:
        df: Complete zone-hour grid with an 'hour' datetime64 column.

    Returns:
        A copy of df with calendar integer and sin/cos columns appended.
    """
    df = df.copy()

    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.weekday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_of_day_sin"], df["hour_of_day_cos"] = cyclic_encode(df["hour_of_day"], period=24)
    df["day_of_week_sin"], df["day_of_week_cos"] = cyclic_encode(df["day_of_week"], period=7)

    return df


def drop_insufficient_history(df: pd.DataFrame, lag_hours: List[int]) -> pd.DataFrame:
    """
    Drop rows lacking enough history for the largest configured lag,
    which would otherwise carry NaN into training across every
    lag/rolling column simultaneously.

    Args:
        df: Zone-hour grid after add_lag_features has run.
        lag_hours: The same lag_hours passed to add_lag_features.

    Returns:
        df with early, history-insufficient rows removed, index reset.
    """
    longest_lag_column = f"demand_lag_{max(lag_hours)}h"
    before = len(df)
    df = df.dropna(subset=[longest_lag_column]).reset_index(drop=True)

    logger.info(
        "Dropped %s rows lacking %d-hour history (%.2f%% of grid)",
        f"{before - len(df):,}", max(lag_hours), (before - len(df)) / before * 100,
    )
    return df


def build_hourly_demand_dataset(df: pd.DataFrame, n_zones: int) -> pd.DataFrame:
    """
    Full pipeline: trip-level zoned data -> model-ready hourly demand
    dataset with lag, rolling, and calendar features.

    Args:
        df: Trip-level DataFrame with 'pickup_datetime' and
            'pickup_zone' (output of Phase 6).
        n_zones: Total number of zones, from config.zones.n_clusters.

    Returns:
        Model-ready DataFrame, one row per (zone, hour) with sufficient
        history, sorted by zone then hour.
    """
    config = load_config()
    lag_hours = config["demand_forecast"]["lag_hours"]
    rolling_windows = config["demand_forecast"]["rolling_windows"]

    hourly = aggregate_hourly_demand(df, n_zones=n_zones)
    hourly = add_lag_features(hourly, lag_hours=lag_hours)
    hourly = add_rolling_features(hourly, rolling_windows=rolling_windows)
    hourly = add_calendar_features(hourly)
    hourly = drop_insufficient_history(hourly, lag_hours=lag_hours)

    logger.info(
        "Built hourly demand dataset: %s rows, %d columns",
        f"{len(hourly):,}", len(hourly.columns),
    )
    return hourly


if __name__ == "__main__":
    from src.utils.config_loader import resolve_path

    logging.basicConfig(level=logging.INFO)

    config = load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])

    zoned_df = pd.read_parquet(processed_dir / "train_zoned.parquet")
    hourly_df = build_hourly_demand_dataset(zoned_df, n_zones=config["zones"]["n_clusters"])

    output_path = processed_dir / "demand_hourly.parquet"
    hourly_df.to_parquet(output_path, index=False)
    logger.info("Saved hourly demand dataset to %s", output_path)
    print(hourly_df.head(10))
    print(hourly_df.describe())