"""
Trains and compares Linear Regression, Random Forest, and XGBoost on
the hourly zone-level demand dataset (Phase 7's output), using a
chronological train/test split - mandatory here because of the
lag/rolling features, unlike trip duration's random split (Phase 8).

Also resolves the MAPE-on-zero-demand caveat metrics.py deferred to
this phase: see mape_excluding_zero_actuals for why standard MAPE alone
is misleading for a target that legitimately includes zero.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.evaluation.metrics import compare_models, mape_excluding_zero_actuals, regression_metrics
from src.utils.config_loader import load_config, resolve_path

logger = logging.getLogger(__name__)

TARGET_COLUMN = "pickup_count"
ZONE_COLUMN = "pickup_zone"

CALENDAR_COLUMNS: List[str] = [
    "hour_of_day", "day_of_week", "is_weekend",
    "hour_of_day_sin", "hour_of_day_cos", "day_of_week_sin", "day_of_week_cos",
]


def get_feature_columns(config: dict) -> List[str]:
    """
    Build lag/rolling column names from config rather than hardcoding
    them - these must exactly match what build_hourly_dataset.py (Phase
    7) generated from these same config values, or the two files could
    silently drift apart if lag_hours/rolling_windows ever change.

    Args:
        config: Loaded project config.

    Returns:
        List of base feature column names (lag + rolling + calendar),
        not including the one-hot zone columns added separately.
    """
    lag_columns = [f"demand_lag_{h}h" for h in config["demand_forecast"]["lag_hours"]]
    rolling_columns = [
        f"demand_rolling_mean_{w}h" for w in config["demand_forecast"]["rolling_windows"]
    ]
    return lag_columns + rolling_columns + CALENDAR_COLUMNS


def prepare_demand_features(
    df: pd.DataFrame, config: dict, n_zones: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select engineered features and one-hot encode pickup_zone.

    pickup_zone is one-hot encoded against a FIXED category range
    (0..n_zones-1), not just the zones present in this particular
    DataFrame - this guarantees train and test splits produce identical
    dummy columns even in the edge case where a zone happened to be
    entirely absent from one split.

    Args:
        df: Hourly demand DataFrame (a train or test split).
        config: Loaded project config.
        n_zones: Total zone count (config.zones.n_clusters), fixing
            the one-hot category range.

    Returns:
        (X, y): feature DataFrame (lag/rolling/calendar + one-hot zone
        columns) and the pickup_count target Series.
    """
    base_columns = get_feature_columns(config)

    zone_categorical = pd.Categorical(df[ZONE_COLUMN].astype(int), categories=range(n_zones))
    zone_dummies = pd.get_dummies(zone_categorical, prefix="zone")

    X = pd.concat(
        [df[base_columns].reset_index(drop=True), zone_dummies.reset_index(drop=True)], axis=1
    )
    y = df[TARGET_COLUMN].reset_index(drop=True)
    return X, y


def get_chronological_split(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the hourly demand grid chronologically: the most recent
    fraction of the time range becomes the test set.

    NOT a random row split, unlike Phase 8's duration split. This
    dataset's lag/rolling features encode genuine dependence on earlier
    hours - a random split could place a training row chronologically
    after a test row it indirectly depends on, leaking future
    information into training. A single global time cutoff avoids this;
    it works cleanly here because Phase 7 built a complete zone x hour
    grid, so every zone has identical time coverage on both sides.

    Args:
        df: Complete hourly demand grid (Phase 7's output), with an
            'hour' datetime64 column.
        config: Loaded project config, read for demand_model.test_size.

    Returns:
        (train_df, test_df), split at one shared cutoff timestamp.
    """
    test_size = config["demand_model"]["test_size"]

    unique_hours = np.sort(df["hour"].unique())
    cutoff_index = int(len(unique_hours) * (1 - test_size))
    cutoff_time = unique_hours[cutoff_index]

    train_df = df[df["hour"] < cutoff_time].reset_index(drop=True)
    test_df = df[df["hour"] >= cutoff_time].reset_index(drop=True)

    logger.info(
        "Chronological split at %s: %s train rows / %s test rows",
        pd.Timestamp(cutoff_time), f"{len(train_df):,}", f"{len(test_df):,}",
    )
    return train_df, test_df


def predict_demand(model, X: pd.DataFrame) -> np.ndarray:
    """
    Predict in log-space and inverse-transform back to raw counts,
    floored at 0.

    Same log1p/expm1 discipline as Phase 8's predict_seconds, for the
    same underlying reason: demand counts are also right-skewed, and
    log1p is safe at demand's actual minimum (log1p(0) == 0), unlike a
    plain log. The floor is 0 here rather than duration's
    min_trip_duration_seconds, since 0 pickups is a valid outcome and
    demand can never be negative.

    Args:
        model: A fitted regressor trained on log1p(pickup_count).
        X: Feature DataFrame to predict on.

    Returns:
        Array of predicted pickup counts, clipped to be >= 0.
    """
    log_predictions = model.predict(X)
    counts = np.expm1(log_predictions)
    return np.clip(counts, a_min=0, a_max=None)


def train_and_evaluate_demand_model(
    model, model_name: str, X_train: pd.DataFrame, y_log_train: pd.Series,
    X_test: pd.DataFrame, y_raw_test: pd.Series,
) -> Dict[str, float]:
    """
    Fit a regressor on log1p(pickup_count), predict via the shared
    inverse-transform, and score against the raw-count test target.

    Args:
        model: An unfitted scikit-learn-compatible regressor.
        model_name: Used as the log label and comparison-table key.
        X_train, y_log_train: Training features and log-transformed target.
        X_test, y_raw_test: Test features and raw-count target.

    Returns:
        Metrics dict from regression_metrics, ready for compare_models().
    """
    model.fit(X_train, y_log_train)
    predictions = predict_demand(model, X_test)
    return regression_metrics(y_raw_test, predictions, label=model_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"]) / "demand"
    models_dir.mkdir(parents=True, exist_ok=True)

    random_seed = config["project"]["random_seed"]
    n_zones = config["zones"]["n_clusters"]

    demand_df = pd.read_parquet(processed_dir / "demand_hourly.parquet")
    train_df, test_df = get_chronological_split(demand_df, config)

    X_train, y_train_raw = prepare_demand_features(train_df, config, n_zones)
    X_test, y_test_raw = prepare_demand_features(test_df, config, n_zones)
    y_train_log = np.log1p(y_train_raw)

    zero_demand_fraction = (y_test_raw == 0).mean()
    logger.info(
        "Test set: %.1f%% of zone-hours have zero true demand - "
        "standard MAPE on these rows is undefined/unstable; see mape_excluding_zero_actuals below.",
        zero_demand_fraction * 100,
    )

    results: Dict[str, Dict[str, float]] = {}
    supplementary_mape: Dict[str, float] = {}

    demand_cfg = config["demand_model"]
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=demand_cfg["random_forest_n_estimators"],
            max_depth=demand_cfg["random_forest_max_depth"],
            random_state=random_seed,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=demand_cfg["xgboost_n_estimators"],
            max_depth=demand_cfg["xgboost_max_depth"],
            learning_rate=demand_cfg["xgboost_learning_rate"],
            random_state=random_seed,
            n_jobs=-1,
            tree_method="hist",
        ),
    }

    for name, model in models.items():
        results[name] = train_and_evaluate_demand_model(
            model, name, X_train, y_train_log, X_test, y_test_raw
        )
        preds = predict_demand(model, X_test)
        nonzero_mape = mape_excluding_zero_actuals(y_test_raw, preds)
        supplementary_mape[name] = nonzero_mape
        logger.info(
            "[%s] MAPE (non-zero-demand hours only) = %s",
            name, f"{nonzero_mape:.4f}" if nonzero_mape is not None else "undefined",
        )
        joblib.dump(model, models_dir / f"{name}.joblib")

    comparison = compare_models(results)
    comparison["mape_nonzero_only"] = comparison.index.map(supplementary_mape)
    logger.info("Demand forecasting model comparison (sorted by RMSE):\n%s", comparison)

    comparison.to_csv(models_dir / "comparison_table.csv")
    logger.info("Saved comparison table to %s", models_dir / "comparison_table.csv")

    best_model_name = comparison.index[0]
    logger.info(
        "Best demand model by RMSE: %s (standard MAPE=%.4f is inflated by %.1f%% zero-demand "
        "test hours; mape_nonzero_only=%.4f is the more honest number to quote).",
        best_model_name,
        comparison.loc[best_model_name, "mape"],
        zero_demand_fraction * 100,
        comparison.loc[best_model_name, "mape_nonzero_only"],
    )