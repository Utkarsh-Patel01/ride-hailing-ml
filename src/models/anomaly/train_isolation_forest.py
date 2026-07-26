"""
Builds the trip anomaly detection pipeline: an Isolation Forest that
flags trips likely to reflect GPS errors, meter failures, or abnormal
routing - unsupervised, since no labeled "this trip was anomalous"
column exists in the data.

See Phase 11's write-up for why Isolation Forest is preferred here over
Local Outlier Factor or One-Class SVM, and for the reasoning behind
each engineered feature below.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils.config_loader import load_config, resolve_path

logger = logging.getLogger(__name__)

ANOMALY_FEATURE_COLUMNS: List[str] = [
    "trip_duration",
    "trip_distance_km",
    "average_speed_kmh",
    "route_efficiency_ratio",
    "passenger_count",
]

_EPSILON = 1e-6  # guards divide-by-zero for trips with ~0 net displacement


def add_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer average_speed_kmh and route_efficiency_ratio - the two
    anomaly-specific features this pipeline needs. These live here,
    not in src/features/, since they have exactly one consumer (this
    pipeline), matching the single-consumer-logic call Phase 8 made
    for the duration pipeline's FEATURE_COLUMNS.

    Args:
        df: Feature-engineered DataFrame with trip_duration,
            trip_distance_km, and manhattan_distance_km already present
            (output of src.features.build_features.build_feature_dataset).

    Returns:
        A copy of df with 'average_speed_kmh' and 'route_efficiency_ratio' added.
    """
    df = df.copy()

    hours = df["trip_duration"] / 3600
    df["average_speed_kmh"] = df["trip_distance_km"] / hours

    # trip_distance_km (straight-line) vs manhattan_distance_km (grid
    # approximation): a ratio well below 1 suggests an indirect route;
    # a ratio >= 1 shouldn't be physically possible (grid distance is
    # never shorter than straight-line distance) and flags a likely
    # coordinate or routing data error.
    df["route_efficiency_ratio"] = df["trip_distance_km"] / (
        df["manhattan_distance_km"] + _EPSILON
    )

    return df


class TripAnomalyDetector:
    """
    Wraps a fitted IsolationForest together with the min/max anomaly
    score bounds observed at training time, so scores normalize to an
    intuitive [0, 1] range consistently between training and inference
    - the same "bundle preprocessing metadata with the fitted model"
    pattern PickupZoneClusterer used for its longitude correction
    factor in Phase 6.
    """

    def __init__(self, contamination: float, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "TripAnomalyDetector":
        """
        Fit the Isolation Forest and record the training set's raw
        anomaly score range for later normalization.

        Args:
            df: DataFrame containing ANOMALY_FEATURE_COLUMNS (call
                add_anomaly_features first if not already present).

        Returns:
            self, for chaining.
        """
        X = df[ANOMALY_FEATURE_COLUMNS]
        self.model.fit(X)

        raw_scores = self._raw_anomaly_scores(X)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())
        self.is_fitted = True

        logger.info(
            "Fitted Isolation Forest on %s trips (contamination=%.3f, raw score range=[%.4f, %.4f])",
            f"{len(df):,}", self.contamination, self._score_min, self._score_max,
        )
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score trips for anomalousness.

        Args:
            df: DataFrame containing ANOMALY_FEATURE_COLUMNS.

        Returns:
            DataFrame (same index as df) with columns:
                anomaly_score_raw: higher = more anomalous, unbounded,
                    the sign-flipped version of sklearn's score_samples.
                anomaly_score: anomaly_score_raw min-max normalized
                    against the TRAINING set's range and clipped to
                    [0, 1] - a point more extreme than anything seen
                    in training clips to 1.0 rather than exceeding it.
                is_anomaly: boolean, True if the model's contamination-
                    thresholded prediction flags this trip.

        Raises:
            RuntimeError: If called before fit() or load().
        """
        if not self.is_fitted:
            raise RuntimeError("TripAnomalyDetector must be fit() or load()ed before score().")

        X = df[ANOMALY_FEATURE_COLUMNS]
        raw_scores = self._raw_anomaly_scores(X)

        normalized = (raw_scores - self._score_min) / (self._score_max - self._score_min + _EPSILON)
        normalized = np.clip(normalized, 0.0, 1.0)

        return pd.DataFrame(
            {
                "anomaly_score_raw": raw_scores,
                "anomaly_score": normalized,
                "is_anomaly": self.model.predict(X) == -1,
            },
            index=df.index,
        )

    def _raw_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Flip sklearn's score_samples (higher = more NORMAL, per
        sklearn's convention) into an intuitive score where higher =
        more ANOMALOUS - done once, here, so nobody downstream has to
        remember the inverted convention.
        """
        return -self.model.score_samples(X)

    def save(self, path: Path) -> None:
        """Persist this object (model + normalization bounds) as a single artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved anomaly detector to %s", path)

    @staticmethod
    def load(path: Path) -> "TripAnomalyDetector":
        """Load a previously saved TripAnomalyDetector."""
        detector = joblib.load(path)
        if not isinstance(detector, TripAnomalyDetector):
            raise TypeError(f"{path} did not contain a TripAnomalyDetector instance.")
        return detector


def summarize_flagged_trips(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Compare feature distributions between flagged and non-flagged
    trips - the primary interpretability tool for this pipeline, since
    (unlike the tree regressors in Phases 8-10) IsolationForest exposes
    no feature_importances_ attribute to inspect directly.

    Args:
        df: DataFrame containing ANOMALY_FEATURE_COLUMNS.
        scores: Output of TripAnomalyDetector.score() on the same df.

    Returns:
        DataFrame with one row per feature, columns 'mean_normal' and
        'mean_flagged', comparing average values between the two groups.
    """
    combined = df[ANOMALY_FEATURE_COLUMNS].copy()
    combined["is_anomaly"] = scores["is_anomaly"].values

    summary = combined.groupby("is_anomaly")[ANOMALY_FEATURE_COLUMNS].mean().T
    summary.columns = ["mean_normal", "mean_flagged"]
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"]) / "anomaly"
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_dir / "train_features.parquet")
    df = add_anomaly_features(df)

    detector = TripAnomalyDetector(
        contamination=config["anomaly_detection"]["contamination"],
        random_state=config["anomaly_detection"]["random_state"],
    )
    detector.fit(df)

    scores = detector.score(df)
    flagged_fraction = scores["is_anomaly"].mean()
    logger.info(
        "Flagged %s trips as anomalous (%.2f%% of dataset, contamination=%.3f)",
        f"{scores['is_anomaly'].sum():,}", flagged_fraction * 100,
        config["anomaly_detection"]["contamination"],
    )

    summary = summarize_flagged_trips(df, scores)
    logger.info("Feature means, flagged vs. normal trips:\n%s", summary)

    detector.save(models_dir / "isolation_forest.joblib")

    scored_output = df[ANOMALY_FEATURE_COLUMNS].copy()
    scored_output["anomaly_score"] = scores["anomaly_score"]
    scored_output["is_anomaly"] = scores["is_anomaly"]
    top_flagged_path = reports_dir.parent / "top_20_flagged_trips.csv"
    scored_output.sort_values("anomaly_score", ascending=False).head(20).to_csv(top_flagged_path)
    logger.info("Saved top 20 most anomalous trips to %s for manual review", top_flagged_path)