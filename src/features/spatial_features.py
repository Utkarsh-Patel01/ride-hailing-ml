"""
Engineers geospatial features from pickup/dropoff coordinate pairs:
great-circle distance, an NYC-grid-aware approximate distance, compass
bearing, and a coarse trip-direction bucket.

All functions are vectorized over NumPy/pandas Series rather than
applied row-by-row, so they run efficiently across the full 1.45M-row
dataset and equally correctly on a single-row DataFrame at inference
time (Phase 13) — same function, no special-casing for batch vs. single
prediction.
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mean Earth radius in kilometers (IUGG mean radius). Using a constant
# sphere radius rather than a full ellipsoid model (e.g. WGS-84) trades
# a small amount of geodetic precision (~0.3% worst case) for a formula
# simple enough to vectorize cleanly - more than accurate enough for
# taxi-trip-scale distances, where GPS noise itself exceeds this error.
_EARTH_RADIUS_KM = 6371.0088

# Fixed 45-degree compass buckets. This is geometry, not a tunable
# business rule, so unlike rush-hour windows it does not belong in
# config.yaml.
_COMPASS_DIRECTIONS: List[str] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

SPATIAL_FEATURE_COLUMNS: List[str] = [
    "trip_distance_km",
    "manhattan_distance_km",
    "bearing_degrees",
    "bearing_sin",
    "bearing_cos",
    "trip_direction",
]


def haversine_distance_km(
    lat1: Union[pd.Series, float],
    lon1: Union[pd.Series, float],
    lat2: Union[pd.Series, float],
    lon2: Union[pd.Series, float],
) -> Union[pd.Series, float]:
    """
    Compute great-circle distance between two coordinate pairs.

    Vectorized: accepts either scalars (single-trip inference) or
    pandas Series (bulk training) and returns the matching type.

    Args:
        lat1, lon1: Latitude/longitude of the first point(s), degrees.
        lat2, lon2: Latitude/longitude of the second point(s), degrees.

    Returns:
        Distance in kilometers, same shape as the inputs.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)

    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    a = np.clip(a, 0, 1)  # guards against floating-point rounding pushing `a` fractionally outside [0, 1]
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return _EARTH_RADIUS_KM * c


def manhattan_distance_km(
    lat1: Union[pd.Series, float],
    lon1: Union[pd.Series, float],
    lat2: Union[pd.Series, float],
    lon2: Union[pd.Series, float],
) -> Union[pd.Series, float]:
    """
    Approximate grid (non-diagonal) travel distance by summing the
    haversine distance of the pure latitude change and the pure
    longitude change - a closer proxy for actual road distance on
    NYC's largely grid-aligned street network than straight-line
    haversine distance alone.

    Args:
        lat1, lon1: Latitude/longitude of the first point(s), degrees.
        lat2, lon2: Latitude/longitude of the second point(s), degrees.

    Returns:
        Approximate grid distance in kilometers, same shape as inputs.
    """
    lat_component = haversine_distance_km(lat1, lon1, lat2, lon1)
    lon_component = haversine_distance_km(lat2, lon1, lat2, lon2)
    return lat_component + lon_component


def bearing_degrees(
    lat1: Union[pd.Series, float],
    lon1: Union[pd.Series, float],
    lat2: Union[pd.Series, float],
    lon2: Union[pd.Series, float],
) -> Union[pd.Series, float]:
    """
    Compute the initial compass bearing from point 1 to point 2.

    Args:
        lat1, lon1: Latitude/longitude of the origin point(s), degrees.
        lat2, lon2: Latitude/longitude of the destination point(s), degrees.

    Returns:
        Bearing in degrees, range [0, 360), where 0 = due north,
        90 = due east. Same shape as the inputs.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_lambda = np.radians(lon2 - lon1)

    x = np.sin(d_lambda) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lambda)

    theta = np.arctan2(x, y)
    return (np.degrees(theta) + 360) % 360


def _bearing_to_compass_direction(bearing: pd.Series) -> pd.Series:
    """
    Bucket continuous bearing degrees into one of 8 compass directions.

    Args:
        bearing: Series of bearing values in degrees, range [0, 360).

    Returns:
        A pandas 'category' Series with values from _COMPASS_DIRECTIONS.
    """
    # Each bucket spans 45 degrees, centered on its named direction
    # (e.g. "N" covers 337.5-360 and 0-22.5). Adding a half-bucket
    # offset before the modulo/divide centers the buckets correctly.
    bucket_index = (np.floor((bearing + 22.5) / 45) % 8).astype(int)
    return pd.Categorical.from_codes(bucket_index, categories=_COMPASS_DIRECTIONS)


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geospatial features derived from pickup/dropoff coordinates.

    Args:
        df: DataFrame containing 'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude' columns.

    Returns:
        A copy of df with SPATIAL_FEATURE_COLUMNS appended. The input
        DataFrame is not mutated.

    Raises:
        KeyError: If any required coordinate column is missing.
    """
    required = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"add_spatial_features requires column(s) {missing}. "
            "Confirm this DataFrame came from src.data.load_data.load_raw_data()."
        )

    df = df.copy()

    df["trip_distance_km"] = haversine_distance_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["manhattan_distance_km"] = manhattan_distance_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["bearing_degrees"] = bearing_degrees(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )

    # Same cyclic-encoding logic as pickup_hour/pickup_weekday in
    # temporal_features.py: bearing is circular (0 deg == 360 deg), so
    # a linear model needs the sin/cos pair to see that correctly.
    df["bearing_sin"] = np.sin(np.radians(df["bearing_degrees"]))
    df["bearing_cos"] = np.cos(np.radians(df["bearing_degrees"]))

    df["trip_direction"] = _bearing_to_compass_direction(df["bearing_degrees"])

    logger.info("Added %d spatial features", len(SPATIAL_FEATURE_COLUMNS))
    return df


if __name__ == "__main__":
    from src.utils.config_loader import load_config, resolve_path

    logging.basicConfig(level=logging.INFO)

    config = load_config()
    interim_dir = resolve_path(config["paths"]["interim_dir"])
    df = pd.read_parquet(interim_dir / "train_cleaned.parquet")

    df = add_spatial_features(df)
    print(df[SPATIAL_FEATURE_COLUMNS].describe(include="all"))