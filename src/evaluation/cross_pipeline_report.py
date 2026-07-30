"""
cross_pipeline_report.py

Ties all three pipelines together into one evaluation report:
- Regenerates real test-set predictions for the winning duration and
  demand models (deterministically, from saved artifacts - no
  intermediate predictions were persisted during training).
- Produces model-comparison, prediction-vs-actual, and residual plots
  for both supervised pipelines.
- Summarizes the anomaly pipeline's flagged-trip output, which has no
  comparable accuracy metric since there is no ground truth.
- Prints a business-framed summary connecting every metric back to the
  specific operational question it answers, as set out in Phase 0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.models.demand.train_demand_models import (
    get_chronological_split,
    predict_demand,
    prepare_demand_features,
)
from src.models.duration.train_baseline import (
    get_train_test_split,
    predict_seconds,
    prepare_duration_features,
)
from src.utils.config_loader import load_config, resolve_path
from src.visualization.plots import plot_model_comparison, plot_predictions_vs_actual, plot_residuals

logger = logging.getLogger(__name__)

# Practical tolerances for the recomputed-vs-saved metric sanity check,
# in each pipeline's native units - not zero, since floating-point
# library version differences can cause tiny, harmless deviations, but
# small enough to catch a genuine drift (e.g. retrained features,
# edited config) rather than noise.
_DURATION_RMSE_TOLERANCE_SECONDS = 1.0
_DEMAND_RMSE_TOLERANCE_PICKUPS = 0.1


def _load_sorted_comparison(path: Path) -> pd.DataFrame:
    """
    Load a saved comparison_table.csv and re-sort by RMSE, rather than
    trusting the file's saved row order - cheap insurance against a
    hand-edited or stale CSV silently reordering the "winner".

    Args:
        path: Path to a comparison_table.csv produced by compare_models().

    Returns:
        DataFrame sorted ascending by 'rmse'.

    Raises:
        FileNotFoundError: With a message naming which training script
            to run first, rather than a bare pandas file-not-found error.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the corresponding pipeline's training "
            "script (Phase 8/9 for duration, Phase 10 for demand) first."
        )
    return pd.read_csv(path, index_col=0).sort_values("rmse")


def _check_metric_drift(recomputed: Dict[str, float], saved: pd.Series, tolerance: float, model_name: str) -> None:
    """
    Warn (not fail) if regenerated RMSE meaningfully disagrees with the
    RMSE saved during training - a signal that train_features.parquet,
    demand_hourly.parquet, or config.yaml changed since that model was
    last trained.

    Args:
        recomputed: Metrics dict from a fresh regression_metrics() call
            on regenerated predictions.
        saved: The corresponding row from the loaded comparison table.
        tolerance: Acceptable absolute RMSE difference, in the
            pipeline's native units.
        model_name: Used in the warning message.
    """
    drift = abs(recomputed["rmse"] - saved["rmse"])
    if drift > tolerance:
        logger.warning(
            "Recomputed RMSE (%.3f) differs from the saved comparison-table RMSE (%.3f) "
            "for '%s' by more than the %.2f tolerance. This usually means the underlying "
            "processed data or config.yaml changed since training last ran - retrain "
            "before trusting these plots.",
            recomputed["rmse"], saved["rmse"], model_name, tolerance,
        )


def evaluate_duration_pipeline(config: dict, reports_dir: Path, models_root: Path) -> Dict:
    """
    Regenerate the duration pipeline's test set and winning model's
    predictions, produce diagnostic plots, and return a summary dict.

    Args:
        config: Loaded project config.
        reports_dir: Directory to save figures into.
        models_root: Root models/ directory (contains duration/ subfolder).

    Returns:
        Dict with 'winning_model' name and its 'metrics' row.
    """
    duration_dir = models_root / "duration"
    comparison = _load_sorted_comparison(duration_dir / "comparison_table.csv")
    winning_name = comparison.index[0]
    logger.info("Duration pipeline winning model (lowest RMSE): %s", winning_name)

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    df = pd.read_parquet(processed_dir / "train_features.parquet")
    X, y_log, y_raw = prepare_duration_features(df)
    _, X_test, _, _, _, y_raw_test = get_train_test_split(X, y_log, y_raw)

    model = joblib.load(duration_dir / f"{winning_name}.joblib")
    min_duration = config["data"]["min_trip_duration_seconds"]
    predictions = predict_seconds(model, X_test, min_duration_seconds=min_duration)

    recomputed = regression_metrics(y_raw_test, predictions, label=f"{winning_name}_recomputed")
    _check_metric_drift(recomputed, comparison.loc[winning_name], _DURATION_RMSE_TOLERANCE_SECONDS, winning_name)

    plot_model_comparison(
        comparison, metric="rmse", title="Trip duration model comparison",
        save_path=reports_dir / "duration_model_comparison.png",
    )
    plot_predictions_vs_actual(
        y_raw_test, predictions, title=f"Duration: predicted vs actual ({winning_name})",
        units="seconds", save_path=reports_dir / "duration_pred_vs_actual.png",
    )
    plot_residuals(
        y_raw_test, predictions, title=f"Duration residuals ({winning_name})",
        units="seconds", save_path=reports_dir / "duration_residuals.png",
    )

    return {"winning_model": winning_name, "metrics": comparison.loc[winning_name].to_dict()}


def evaluate_demand_pipeline(config: dict, reports_dir: Path, models_root: Path) -> Dict:
    """
    Regenerate the demand pipeline's chronological test split and
    winning model's predictions, produce diagnostic plots, and return
    a summary dict.

    Args:
        config: Loaded project config.
        reports_dir: Directory to save figures into.
        models_root: Root models/ directory (contains demand/ subfolder).

    Returns:
        Dict with 'winning_model' name and its 'metrics' row.
    """
    demand_dir = models_root / "demand"
    comparison = _load_sorted_comparison(demand_dir / "comparison_table.csv")
    winning_name = comparison.index[0]
    logger.info("Demand pipeline winning model (lowest RMSE): %s", winning_name)

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    demand_df = pd.read_parquet(processed_dir / "demand_hourly.parquet")
    _, test_df = get_chronological_split(demand_df, config)

    n_zones = config["zones"]["n_clusters"]
    X_test, y_test_raw = prepare_demand_features(test_df, config, n_zones)

    model = joblib.load(demand_dir / f"{winning_name}.joblib")
    predictions = predict_demand(model, X_test)

    recomputed = regression_metrics(y_test_raw, predictions, label=f"{winning_name}_recomputed")
    _check_metric_drift(recomputed, comparison.loc[winning_name], _DEMAND_RMSE_TOLERANCE_PICKUPS, winning_name)

    plot_model_comparison(
        comparison, metric="rmse", title="Demand forecasting model comparison",
        save_path=reports_dir / "demand_model_comparison.png",
    )
    plot_predictions_vs_actual(
        y_test_raw, predictions, title=f"Demand: predicted vs actual ({winning_name})",
        units="pickups/hour", save_path=reports_dir / "demand_pred_vs_actual.png",
    )
    plot_residuals(
        y_test_raw, predictions, title=f"Demand residuals ({winning_name})",
        units="pickups/hour", save_path=reports_dir / "demand_residuals.png",
    )

    return {"winning_model": winning_name, "metrics": comparison.loc[winning_name].to_dict()}


def summarize_anomaly_pipeline(config: dict, reports_dir: Path) -> Dict:
    """
    Summarize the anomaly pipeline's output. Unlike duration and
    demand, there is no ground truth to score against, so this reports
    the contamination assumption and points at the human-reviewable
    flagged-trip evidence rather than a fabricated accuracy metric.

    Args:
        config: Loaded project config.
        reports_dir: The reports/figures directory (its parent holds
            top_20_flagged_trips.csv, saved in Phase 11).

    Returns:
        Dict with 'contamination' and a preview of the top flagged trips.

    Raises:
        FileNotFoundError: If Phase 11's training script hasn't run yet.
    """
    top_flagged_path = reports_dir.parent / "top_20_flagged_trips.csv"
    if not top_flagged_path.exists():
        raise FileNotFoundError(
            f"{top_flagged_path} not found. Run "
            "src/models/anomaly/train_isolation_forest.py (Phase 11) first."
        )

    top_flagged = pd.read_csv(top_flagged_path, index_col=0)
    contamination = config["anomaly_detection"]["contamination"]

    return {"contamination": contamination, "top_flagged_preview": top_flagged.head(5)}


def print_business_summary(duration_result: Dict, demand_result: Dict, anomaly_result: Dict) -> None:
    """
    Print a plain-language rollup connecting each pipeline's technical
    metric back to the specific operational decision it informs - the
    same three-way business split Phase 0 opened this project with.
    """
    d = duration_result["metrics"]
    m = demand_result["metrics"]

    print("=" * 72)
    print("CROSS-PIPELINE EVALUATION SUMMARY")
    print("=" * 72)
    print(f"""
[1] TRIP DURATION -> feeds ETA display and time-based pricing
    Winning model: {duration_result['winning_model']}
    MAE:  {d['mae']:.1f} seconds   (typical ETA error a rider would actually see)
    RMSE: {d['rmse']:.1f} seconds  (penalizes rare, large misses more heavily than MAE)
    R2:   {d['r2']:.3f}            (fraction of duration variance the model explains)
""")
    print(f"""
[2] ZONAL DEMAND -> feeds driver repositioning decisions
    Winning model: {demand_result['winning_model']}
    MAE:  {m['mae']:.2f} pickups/hour   (typical zone-hour forecast error)
    RMSE: {m['rmse']:.2f} pickups/hour
    MAPE (non-zero-demand hours only): {m.get('mape_nonzero_only', float('nan')):.3f}
        (standard MAPE is intentionally not quoted alone here - see
        Phase 10 for why it's unreliable on a target with legitimate
        zero values)
""")
    print(f"""
[3] ANOMALY DETECTION -> feeds the ops fraud/data-quality review queue
    Contamination assumption: {anomaly_result['contamination']:.1%} of trips flagged for review
    No accuracy metric applies here - there is no ground-truth label.
    See reports/top_20_flagged_trips.csv for the human-reviewable
    evidence that flagged trips look genuinely implausible.
""")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    models_root = resolve_path(config["paths"]["models_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    duration_result = evaluate_duration_pipeline(config, reports_dir, models_root)
    demand_result = evaluate_demand_pipeline(config, reports_dir, models_root)
    anomaly_result = summarize_anomaly_pipeline(config, reports_dir)

    print_business_summary(duration_result, demand_result, anomaly_result)