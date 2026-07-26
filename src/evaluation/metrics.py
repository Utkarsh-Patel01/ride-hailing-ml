"""
Shared regression evaluation metrics used across every model-comparison
phase in this project: trip duration (Phase 8-9), demand forecasting
(Phase 10), and the cross-pipeline rollup (Phase 12).

Centralizing this here guarantees that "RMSE" (or MAE, MAPE, R2) means
the exact same computation everywhere it's reported - a model
comparison table is only trustworthy if every column in it was computed
by the same code.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error
)

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ["mae", "rmse", "mape", "r2"]


def regression_metrics(y_true, y_pred, label: str = "") -> Dict[str, float]:
    """
    Compute the standard regression metric suite used throughout this
    project: MAE, RMSE, MAPE, R2.

    Note on MAPE: well-behaved for trip duration, where every target
    value is >= 10 seconds after Phase 4's cleaning (never zero). It is
    NOT well-behaved for demand forecasting, where a correct prediction
    is legitimately "0 pickups" in many zone-hours - Phase 10 discusses
    how to interpret this metric for that pipeline specifically.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions, same length/order as y_true.
        label: Optional name (e.g. model name), included in the log
            line only, purely for readability when several models'
            results are logged in sequence by one training script.

    Returns:
        Dict with keys 'mae', 'rmse', 'mape', 'r2'.

    Raises:
        ValueError: If y_true and y_pred have mismatched lengths.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    prefix = f"[{label}] " if label else ""
    logger.info(
        "%sMAE=%.3f  RMSE=%.3f  MAPE=%.4f  R2=%.4f",
        prefix, metrics["mae"], metrics["rmse"], metrics["mape"], metrics["r2"],
    )
    return metrics


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Assemble multiple models' metric dicts into one comparison table.

    Args:
        results: Mapping of model name -> metrics dict, each produced
            by regression_metrics(), e.g.
            {"linear_regression": {...}, "random_forest": {...}}.

    Returns:
        DataFrame indexed by model name, columns in a fixed order
        (mae, rmse, mape, r2), sorted ascending by RMSE so the best
        model (by RMSE) appears first.

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("compare_models received an empty results dict.")

    table = pd.DataFrame(results).T
    table = table[METRIC_COLUMNS]
    return table.sort_values("rmse")

def mape_excluding_zero_actuals(y_true, y_pred) -> Optional[float]:
    """
    Compute MAPE only over rows where the true value is nonzero.

    Standard MAPE (the 'mape' key from regression_metrics) divides by
    y_true, which is undefined at y_true == 0 and unstable near it.
    This is a non-issue for trip duration (every value is >= 10 seconds
    after Phase 4's cleaning) but a real one for any target that can
    legitimately be zero - such as hourly pickup demand in a low-
    traffic zone overnight, which Phase 10 uses this for.

    This is a generic utility, not demand-specific logic: any future
    pipeline with a zero-inclusive target should reach for this rather
    than trusting standard MAPE blindly.

    Args:
        y_true: Ground-truth values, some of which may be zero.
        y_pred: Model predictions, same length/order as y_true.

    Returns:
        MAPE computed only over rows where y_true != 0, as a float;
        or None if every y_true value is zero (undefined).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    nonzero_mask = y_true != 0

    if not nonzero_mask.any():
        return None

    return float(mean_absolute_percentage_error(y_true[nonzero_mask], y_pred[nonzero_mask]))