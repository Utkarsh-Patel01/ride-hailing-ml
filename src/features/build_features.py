"""
Orchestrates the full feature engineering pipeline: applies temporal
and spatial feature functions in sequence, validates the result, and
(when run as a script) persists a processed, model-ready dataset.

This is the single entry point every pipeline (duration, demand,
anomaly) and the inference layer should use to go from cleaned data to
feature-engineered data - never call add_temporal_features or
add_spatial_features independently outside this module, so training
and inference can never silently drift apart.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.spatial_features import SPATIAL_FEATURE_COLUMNS, add_spatial_features
from src.features.temporal_features import TEMPORAL_FEATURE_COLUMNS, add_temporal_features

logger = logging.getLogger(__name__)

ENGINEERED_FEATURE_COLUMNS = TEMPORAL_FEATURE_COLUMNS + SPATIAL_FEATURE_COLUMNS


class FeatureValidationError(Exception):
    """Raised when engineered features contain NaN or infinite values."""


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a cleaned DataFrame.

    Args:
        df: A cleaned DataFrame, typically loaded from
            data/interim/train_cleaned.parquet (or, at inference time,
            a single-row DataFrame built from a request payload).

    Returns:
        A copy of df with all temporal and spatial feature columns
        appended, validated to contain no NaN or infinite values.

    Raises:
        FeatureValidationError: If any engineered column contains NaN
            or infinite values after both feature steps run.
    """
    df = add_temporal_features(df)
    df = add_spatial_features(df)

    _validate_engineered_features(df)

    logger.info(
        "Feature engineering complete: %d total features (%d engineered)",
        len(df.columns),
        len(ENGINEERED_FEATURE_COLUMNS),
    )
    return df


def _validate_engineered_features(df: pd.DataFrame) -> None:
    """
    Verify no engineered column contains NaN or infinite values.

    Args:
        df: DataFrame after both add_temporal_features and
            add_spatial_features have run.

    Raises:
        FeatureValidationError: Naming the specific offending column(s),
            so a failure here points directly at the feature function
            responsible rather than requiring a manual column-by-column
            search.
    """
    numeric_engineered = [
        col for col in ENGINEERED_FEATURE_COLUMNS
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    nan_columns = [col for col in numeric_engineered if df[col].isna().any()]
    if nan_columns:
        raise FeatureValidationError(
            f"NaN values found in engineered column(s): {nan_columns}. "
            "This usually indicates a floating-point edge case (e.g. "
            "identical pickup/dropoff coordinates) not being guarded "
            "against in the responsible feature function."
        )

    inf_columns = [
        col for col in numeric_engineered
        if np.isinf(df[col].to_numpy()).any()
    ]
    if inf_columns:
        raise FeatureValidationError(
            f"Infinite values found in engineered column(s): {inf_columns}."
        )


if __name__ == "__main__":
    from src.utils.config_loader import load_config, resolve_path

    logging.basicConfig(level=logging.INFO)

    config = load_config()
    interim_dir = resolve_path(config["paths"]["interim_dir"])
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    cleaned_df = pd.read_parquet(interim_dir / "train_cleaned.parquet")
    featured_df = build_feature_dataset(cleaned_df)

    output_path = processed_dir / "train_features.parquet"
    featured_df.to_parquet(output_path, index=False)

    logger.info("Saved feature-engineered dataset to %s", output_path)
    print(featured_df[ENGINEERED_FEATURE_COLUMNS].describe(include="all"))