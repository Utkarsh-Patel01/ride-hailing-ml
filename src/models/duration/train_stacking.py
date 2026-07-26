"""
Builds a leakage-free StackingRegressor combining Random Forest, Extra
Trees, and XGBoost as base estimators with a RidgeCV meta-learner, and
appends its result to the duration model comparison table.

See Phase 9's write-up for why this file re-instantiates fresh base
models rather than reusing train_tree_models.py's fitted .joblib
artifacts: StackingRegressor's internal cross-validation is what
prevents the meta-learner from training on artificially-good,
already-seen predictions.
"""

from __future__ import annotations

import logging

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

from src.evaluation.metrics import compare_models, regression_metrics
from src.models.duration.train_baseline import (
    get_train_test_split,
    predict_seconds,
    prepare_duration_features,
)
from src.utils.config_loader import load_config, resolve_path

logger = logging.getLogger(__name__)


def build_stacking_regressor(config: dict, random_state: int) -> StackingRegressor:
    """
    Construct (but do not fit) the stacking ensemble: three tree-based
    base estimators plus a RidgeCV meta-learner, combined via
    scikit-learn's cross-validated stacking mechanism.

    Args:
        config: Loaded project config, read for the stacking
            hyperparameters section.
        random_state: Shared seed for every base estimator, for
            reproducibility.

    Returns:
        An unfitted StackingRegressor.
    """
    stacking_cfg = config["stacking"]

    base_estimators = [
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=stacking_cfg["random_forest_n_estimators"],
                max_depth=stacking_cfg["random_forest_max_depth"],
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=stacking_cfg["extra_trees_n_estimators"],
                max_depth=stacking_cfg["extra_trees_max_depth"],
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "xgboost",
            XGBRegressor(
                n_estimators=stacking_cfg["xgboost_n_estimators"],
                max_depth=stacking_cfg["xgboost_max_depth"],
                learning_rate=stacking_cfg["xgboost_learning_rate"],
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
            ),
        ),
    ]

    # RidgeCV cross-validates its own regularization strength, so the
    # meta-learner's one real hyperparameter is chosen from data rather
    # than guessed - consistent with how n_clusters and contamination
    # were treated as measured, revisitable choices in earlier phases.
    meta_learner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])

    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=stacking_cfg["cv_folds"],
        n_jobs=-1,
        passthrough=False,  # meta-learner sees only the 3 base predictions, not raw features
    )


def update_comparison_table(comparison_path, new_result: dict, model_name: str) -> pd.DataFrame:
    """
    Load the existing duration model comparison table, add or replace
    one model's row, and return the full, re-sorted table.

    Args:
        comparison_path: Path to comparison_table.csv, produced by
            train_tree_models.py in Phase 8.
        new_result: Metrics dict for the model being added, from
            regression_metrics().
        model_name: Row label for this model in the table.

    Returns:
        The full comparison DataFrame, sorted by RMSE ascending.

    Raises:
        FileNotFoundError: If comparison_table.csv doesn't exist yet -
            run train_tree_models.py (Phase 8) first.
    """
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"{comparison_path} not found. Run "
            "src/models/duration/train_tree_models.py (Phase 8) before this script."
        )

    existing = pd.read_csv(comparison_path, index_col=0)
    results = existing.to_dict(orient="index")
    results[model_name] = new_result
    return compare_models(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"]) / "duration"

    random_seed = config["project"]["random_seed"]
    min_duration = config["data"]["min_trip_duration_seconds"]

    df = pd.read_parquet(processed_dir / "train_features.parquet")
    X, y_log, y_raw = prepare_duration_features(df)
    X_train, X_test, y_log_train, y_log_test, y_raw_train, y_raw_test = get_train_test_split(
        X, y_log, y_raw
    )

    stacking_model = build_stacking_regressor(config, random_seed)
    logger.info(
        "Fitting stacking ensemble (cv=%d folds, this takes a while)...",
        config["stacking"]["cv_folds"],
    )
    stacking_model.fit(X_train, y_log_train)

    stacking_predictions = predict_seconds(stacking_model, X_test, min_duration_seconds=min_duration)
    stacking_metrics = regression_metrics(y_raw_test, stacking_predictions, label="stacking_ensemble")

    comparison_path = models_dir / "comparison_table.csv"
    full_comparison = update_comparison_table(comparison_path, stacking_metrics, "stacking_ensemble")
    logger.info("Full duration model comparison (sorted by RMSE):\n%s", full_comparison)
    full_comparison.to_csv(comparison_path)

    # The meta-learner's learned coefficients show how much it trusts
    # each base model's prediction - a direct, inspectable answer to
    # "did stacking actually learn non-equal weights, or did it just
    # rediscover a plain average?"
    base_model_names = [name for name, _ in stacking_model.estimators]
    meta_weights = pd.Series(stacking_model.final_estimator_.coef_, index=base_model_names)
    logger.info("Learned meta-learner weights (higher = more trusted):\n%s", meta_weights)

    joblib.dump(stacking_model, models_dir / "stacking_ensemble.joblib")
    logger.info("Saved stacking ensemble to %s", models_dir / "stacking_ensemble.joblib")