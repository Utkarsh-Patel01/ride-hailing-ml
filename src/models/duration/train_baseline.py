"""
Establishes the trip-duration modeling contract: which features are
used, how the data is split, and how the target is transformed - then
trains a Linear Regression baseline against it.

Every other duration model (train_tree_models.py, and Phase 9's
stacking ensemble) imports FEATURE_COLUMNS and get_train_test_split
from this file rather than redefining them, so all duration models are
guaranteed to be compared on identical grounds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import regression_metrics
from src.utils.config_loader import load_config, resolve_path

logger = logging.getLogger(__name__)

# The full feature set for the trip-duration pipeline. trip_direction
# (the 8-bucket compass category from Phase 5) is deliberately excluded:
# bearing_sin/bearing_cos already encode the same directional
# information continuously and more precisely, so including the bucket
# too would be redundant and would additionally require one-hot
# encoding for the Linear Regression model below - added complexity for
# no real signal gain.
FEATURE_COLUMNS: List[str] = [
    "pickup_hour", "pickup_weekday", "pickup_is_weekend", "pickup_is_rush_hour",
    "pickup_hour_sin", "pickup_hour_cos", "pickup_weekday_sin", "pickup_weekday_cos",
    "trip_distance_km", "manhattan_distance_km",
    "bearing_degrees", "bearing_sin", "bearing_cos",
    "passenger_count", "vendor_id_code", "store_and_fwd_flag_code",
]

TARGET_COLUMN = "trip_duration"


def prepare_duration_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Select and encode features for the duration pipeline, and produce
    both the log-transformed training target and the raw-seconds target
    used for final, business-interpretable evaluation.

    Args:
        df: Feature-engineered DataFrame, output of
            src.features.build_features.build_feature_dataset (loaded
            here from data/processed/train_features.parquet).

    Returns:
        Tuple of (X, y_log, y_raw):
            X: DataFrame of FEATURE_COLUMNS, fully numeric.
            y_log: log1p(trip_duration) - what models are trained on.
            y_raw: trip_duration in raw seconds - what regression_metrics
                is ultimately scored against, after inverse-transforming
                predictions.
    """
    df = df.copy()

    # vendor_id and store_and_fwd_flag were loaded as pandas 'category'
    # dtype back in Phase 2 - .cat.codes gives a stable integer encoding
    # a linear model can consume directly.
    df["vendor_id_code"] = df["vendor_id"].cat.codes
    df["store_and_fwd_flag_code"] = df["store_and_fwd_flag"].cat.codes

    X = df[FEATURE_COLUMNS].copy()
    y_raw = df[TARGET_COLUMN].copy()
    y_log = np.log1p(y_raw)

    return X, y_log, y_raw


def get_train_test_split(
    X: pd.DataFrame, y_log: pd.Series, y_raw: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split features and both target representations using a random
    split - appropriate here since duration prediction has no
    sequential dependency between rows (contrast with Phase 10's
    necessarily chronological split for demand forecasting).

    Args:
        X: Feature DataFrame from prepare_duration_features.
        y_log: log1p-transformed target.
        y_raw: Raw-seconds target, split identically (same indices) so
            X_test/y_log_test/y_raw_test all refer to the same rows.

    Returns:
        (X_train, X_test, y_log_train, y_log_test, y_raw_train, y_raw_test)
    """
    config = load_config()
    test_size = config["duration_model"]["test_size"]
    random_state = config["project"]["random_seed"]

    X_train, X_test, y_log_train, y_log_test, y_raw_train, y_raw_test = train_test_split(
        X, y_log, y_raw, test_size=test_size, random_state=random_state,
    )

    logger.info(
        "Split %s rows -> %s train / %s test (test_size=%.2f, random_state=%d)",
        f"{len(X):,}", f"{len(X_train):,}", f"{len(X_test):,}", test_size, random_state,
    )
    return X_train, X_test, y_log_train, y_log_test, y_raw_train, y_raw_test


def predict_seconds(model, X: pd.DataFrame, min_duration_seconds: float) -> np.ndarray:
    """
    Predict in log-space and inverse-transform back to seconds, with a
    floor at min_duration_seconds.

    Every duration model in this project must be scored in real seconds,
    not log-seconds - this function is the single place that inverse-
    transform happens, so it can't be forgotten or done inconsistently
    in one script and not another.

    Args:
        model: A fitted regressor with a .predict() method, trained on
            log1p(trip_duration).
        X: Feature DataFrame to predict on.
        min_duration_seconds: Floor applied after inverse-transforming,
            guarding against a model predicting a physically meaningless
            negative duration.

    Returns:
        Array of predicted durations in seconds, clipped to be >= min_duration_seconds.
    """
    log_predictions = model.predict(X)
    seconds_predictions = np.expm1(log_predictions)
    return np.clip(seconds_predictions, a_min=min_duration_seconds, a_max=None)


def train_linear_regression_baseline(
    X_train: pd.DataFrame, y_log_train: pd.Series
) -> LinearRegression:
    """
    Fit the Linear Regression baseline.

    Args:
        X_train: Training features.
        y_log_train: log1p-transformed training target.

    Returns:
        The fitted LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_log_train)
    logger.info("Fitted Linear Regression baseline on %s rows", f"{len(X_train):,}")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"]) / "duration"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_dir / "train_features.parquet")
    X, y_log, y_raw = prepare_duration_features(df)
    X_train, X_test, y_log_train, y_log_test, y_raw_train, y_raw_test = get_train_test_split(
        X, y_log, y_raw
    )

    model = train_linear_regression_baseline(X_train, y_log_train)

    min_duration = config["data"]["min_trip_duration_seconds"]
    predictions_seconds = predict_seconds(model, X_test, min_duration_seconds=min_duration)

    metrics = regression_metrics(y_raw_test, predictions_seconds, label="linear_regression")

    joblib.dump(model, models_dir / "linear_regression.joblib")
    logger.info("Saved baseline model to %s", models_dir / "linear_regression.joblib")