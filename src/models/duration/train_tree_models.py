"""
Trains Random Forest, Extra Trees, and XGBoost on the exact feature set
and train/test split established in train_baseline.py, and assembles
the full trip-duration model comparison table.

Phase 9's stacking ensemble loads the three .joblib artifacts this
script saves as its base estimators - it does not retrain them.
"""

from __future__ import annotations

import logging
from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from src.evaluation.metrics import compare_models, regression_metrics
from src.models.duration.train_baseline import (
    FEATURE_COLUMNS,
    get_train_test_split,
    predict_seconds,
    prepare_duration_features,
    train_linear_regression_baseline,
)
from src.utils.config_loader import load_config, resolve_path

logger = logging.getLogger(__name__)


def train_and_evaluate_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_log_train: pd.Series,
    X_test: pd.DataFrame,
    y_raw_test: pd.Series,
    min_duration_seconds: float,
) -> Dict[str, float]:
    """
    Fit a scikit-learn-compatible regressor on the log-transformed
    target, predict in seconds via the shared inverse-transform, and
    score against the raw-seconds test target.

    Args:
        model: An unfitted estimator implementing .fit()/.predict()
            (RandomForestRegressor, ExtraTreesRegressor, XGBRegressor,
            or anything else with the same interface).
        model_name: Used as the log label and the comparison-table key.
        X_train, y_log_train: Training features and log-transformed target.
        X_test, y_raw_test: Test features and raw-seconds target.
        min_duration_seconds: Passed through to predict_seconds' floor.

    Returns:
        The metrics dict from regression_metrics, ready to feed into
        compare_models() alongside every other model's result.
    """
    model.fit(X_train, y_log_train)
    predictions_seconds = predict_seconds(model, X_test, min_duration_seconds)
    return regression_metrics(y_raw_test, predictions_seconds, label=model_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"]) / "duration"
    models_dir.mkdir(parents=True, exist_ok=True)

    random_seed = config["project"]["random_seed"]
    min_duration = config["data"]["min_trip_duration_seconds"]

    df = pd.read_parquet(processed_dir / "train_features.parquet")
    X, y_log, y_raw = prepare_duration_features(df)
    X_train, X_test, y_log_train, y_log_test, y_raw_train, y_raw_test = get_train_test_split(
        X, y_log, y_raw
    )

    results: Dict[str, Dict[str, float]] = {}

    # Re-fit and re-score the baseline here too, purely so it appears
    # in the same comparison table as the tree models below - training
    # itself already happened once in train_baseline.py; this is not
    # duplicated *logic*, just a second, cheap call to an already-shared
    # training function against data already loaded in this process.
    linear_model = train_linear_regression_baseline(X_train, y_log_train)
    linear_preds = predict_seconds(linear_model, X_test, min_duration)
    results["linear_regression"] = regression_metrics(
        y_raw_test, linear_preds, label="linear_regression"
    )

    random_forest = RandomForestRegressor(
        n_estimators=200, max_depth=16, random_state=random_seed, n_jobs=-1,
    )
    results["random_forest"] = train_and_evaluate_model(
        random_forest, "random_forest", X_train, y_log_train, X_test, y_raw_test, min_duration,
    )
    joblib.dump(random_forest, models_dir / "random_forest.joblib")

    extra_trees = ExtraTreesRegressor(
        n_estimators=200, max_depth=16, random_state=random_seed, n_jobs=-1,
    )
    results["extra_trees"] = train_and_evaluate_model(
        extra_trees, "extra_trees", X_train, y_log_train, X_test, y_raw_test, min_duration,
    )
    joblib.dump(extra_trees, models_dir / "extra_trees.joblib")

    xgboost_model = XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        random_state=random_seed, n_jobs=-1, tree_method="hist",
    )
    results["xgboost"] = train_and_evaluate_model(
        xgboost_model, "xgboost", X_train, y_log_train, X_test, y_raw_test, min_duration,
    )
    joblib.dump(xgboost_model, models_dir / "xgboost.joblib")

    comparison = compare_models(results)
    logger.info("Trip duration model comparison (sorted by RMSE):\n%s", comparison)

    comparison.to_csv(models_dir / "comparison_table.csv")
    logger.info("Saved comparison table to %s", models_dir / "comparison_table.csv")

    top_model_name = comparison.index[0]
    logger.info("Feature importances for top model (%s):", top_model_name)
    top_model = {"random_forest": random_forest, "extra_trees": extra_trees, "xgboost": xgboost_model}.get(
        top_model_name
    )
    if top_model is not None:
        importances = pd.Series(
            top_model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)
        print(importances)