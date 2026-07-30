"""
demand_predictor.py

Wraps the saved winning demand-forecasting model in a class that
answers "how many pickups will this zone see in the next hour" from a
raw zone/time request.

See this phase's write-up for the central architectural honesty this
file embodies: lag/rolling features are served from a STATIC snapshot
of src.models.demand.build_hourly_dataset's output (Phase 7), standing
in for what a production system would serve from a live-updated
feature store. Predictions are only servable for hours the snapshot's
history actually covers, plus lead time for the longest configured
lag/rolling window - a real, named limitation, not a bug.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.features.temporal_features import cyclic_encode
from src.inference.errors import InferenceValidationError
from src.models.demand.train_demand_models import (
    CALENDAR_COLUMNS,
    TARGET_COLUMN,
    ZONE_COLUMN,
    get_feature_columns,
    prepare_demand_features,
    predict_demand,
)
from src.utils.config_loader import load_config, resolve_path
from src.zones.kmeans_zones import PickupZoneClusterer

logger = logging.getLogger(__name__)

# This value is never used for anything - prepare_demand_features()
# (Phase 10) requires a pickup_count column to exist so it can return
# (X, y), but at inference time the true count is exactly what we're
# predicting. Attaching a placeholder here avoids duplicating that
# function's one-hot-encoding logic a second time in this file.
_UNUSED_TARGET_PLACEHOLDER = 0


def _load_winning_model_name(comparison_path: Path) -> str:
    """
    Read comparison_table.csv and return the lowest-RMSE model's name.

    Deliberately NOT imported from src.evaluation.cross_pipeline_report,
    even though the logic is nearly identical - that module pulls in
    matplotlib for plotting, which a production inference path has no
    reason to depend on. See this phase's write-up for the full
    reasoning.

    Args:
        comparison_path: Path to demand/comparison_table.csv.

    Returns:
        The model name (row index) with the lowest RMSE.

    Raises:
        FileNotFoundError: If the comparison table doesn't exist yet.
    """
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"{comparison_path} not found. Run "
            "src/models/demand/train_demand_models.py (Phase 10) first."
        )
    comparison = pd.read_csv(comparison_path, index_col=0).sort_values("rmse")
    return comparison.index[0]


class DemandPredictor:
    """
    Loads the saved zone clusterer and winning demand model, and serves
    zone-hour pickup demand forecasts from raw requests.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Args:
            model_path: Override for the demand model artifact. Defaults
                to the winning model per demand/comparison_table.csv.
        """
        self.config = load_config()
        models_root = resolve_path(self.config["paths"]["models_dir"])

        resolved_model_path = model_path or (
            models_root / "demand" / f"{_load_winning_model_name(models_root / 'demand' / 'comparison_table.csv')}.joblib"
        )
        self.model = joblib.load(resolved_model_path)
        self.n_zones: int = self.config["zones"]["n_clusters"]
        self.lag_hours = self.config["demand_forecast"]["lag_hours"]
        self.rolling_windows = self.config["demand_forecast"]["rolling_windows"]
        self._max_offset_hours = max(self.lag_hours + self.rolling_windows)

        self.zone_clusterer = PickupZoneClusterer.load(models_root / "kmeans_zones.joblib")

        self._history, self._history_min, self._history_max = self._load_history_index()

        logger.info(
            "Loaded demand model from %s (history snapshot covers %s to %s)",
            resolved_model_path, self._history_min, self._history_max,
        )

    def _load_history_index(self) -> Tuple[Dict[Tuple[int, pd.Timestamp], int], pd.Timestamp, pd.Timestamp]:
        """
        Load Phase 7's complete hourly demand grid once and index it
        for fast (zone, hour) -> pickup_count lookups.

        This is the static-snapshot stand-in for a live feature store,
        named explicitly in this phase's write-up.

        Returns:
            (index, min_hour, max_hour): the lookup dict, and the
            snapshot's covered time range, used to give informative
            errors when a request falls outside it.
        """
        processed_dir = resolve_path(self.config["paths"]["processed_dir"])
        demand_df = pd.read_parquet(processed_dir / "demand_hourly.parquet")

        index = {
            (int(row.pickup_zone), row.hour): int(row.pickup_count)
            for row in demand_df.itertuples(index=False)
        }
        return index, demand_df["hour"].min(), demand_df["hour"].max()

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict pickup demand for a single zone-hour request.

        Args:
            request: Dict with keys:
                target_datetime (str or datetime-parseable value) -
                    floored to the hour internally; sub-hour precision
                    is intentionally discarded since demand is
                    forecast at hourly granularity.
                Either zone_id (int, 0..n_zones-1) directly, OR
                pickup_latitude + pickup_longitude (floats), resolved
                to a zone via the same K-Means clusterer training used.

        Returns:
            Dict with 'predicted_pickup_count' (float), 'zone_id',
            'target_hour' (ISO string), and 'model_name'.

        Raises:
            InferenceValidationError: If the zone is invalid, or if
                the historical snapshot lacks enough data before
                target_datetime to compute the required lag/rolling
                features (see this phase's write-up).
        """
        zone_id = self._resolve_zone_id(request)
        target_hour = self._parse_target_hour(request)

        recent_counts = self._fetch_recent_counts(zone_id, target_hour)
        row = self._build_feature_row(zone_id, target_hour, recent_counts)

        X, _ = prepare_demand_features(row, self.config, self.n_zones)
        prediction = predict_demand(self.model, X)[0]

        return {
            "predicted_pickup_count": float(prediction),
            "zone_id": zone_id,
            "target_hour": target_hour.isoformat(),
            "model_name": type(self.model).__name__,
        }

    def _resolve_zone_id(self, request: Dict[str, Any]) -> int:
        """Resolve zone_id directly, or derive it from pickup coordinates."""
        if "zone_id" in request and request["zone_id"] is not None:
            zone_id = int(request["zone_id"])
        elif "pickup_latitude" in request and "pickup_longitude" in request:
            coord_df = pd.DataFrame([{
                "pickup_latitude": float(request["pickup_latitude"]),
                "pickup_longitude": float(request["pickup_longitude"]),
            }])
            zone_id = int(self.zone_clusterer.predict(coord_df)[0])
        else:
            raise InferenceValidationError(
                "Request must include either 'zone_id', or both "
                "'pickup_latitude' and 'pickup_longitude'."
            )

        if not (0 <= zone_id < self.n_zones):
            raise InferenceValidationError(
                f"zone_id must be between 0 and {self.n_zones - 1}, got {zone_id}."
            )
        return zone_id

    def _parse_target_hour(self, request: Dict[str, Any]) -> pd.Timestamp:
        """Parse and floor target_datetime to the top of the hour."""
        if "target_datetime" not in request:
            raise InferenceValidationError("Request must include 'target_datetime'.")
        return pd.Timestamp(request["target_datetime"]).floor("h")

    def _fetch_recent_counts(self, zone_id: int, target_hour: pd.Timestamp) -> Dict[int, int]:
        """
        Look up this zone's actual pickup counts for each hour offset
        (1 to the longest configured lag/rolling window) before
        target_hour, from the static history snapshot.

        Args:
            zone_id: Resolved zone id.
            target_hour: The hour being forecast.

        Returns:
            Dict mapping hour-offset (1, 2, 3, ...) -> pickup_count,
            containing only offsets actually present in the snapshot.
        """
        counts = {}
        for offset in range(1, self._max_offset_hours + 1):
            key = (zone_id, target_hour - pd.Timedelta(hours=offset))
            if key in self._history:
                counts[offset] = self._history[key]
        return counts

    def _build_feature_row(
        self, zone_id: int, target_hour: pd.Timestamp, recent_counts: Dict[int, int]
    ) -> pd.DataFrame:
        """
        Assemble a single-row DataFrame with every column
        prepare_demand_features expects: lag features, rolling-mean
        features, calendar features, and zone/target columns.

        Raises:
            InferenceValidationError: Naming exactly which lag or
                rolling feature couldn't be computed, and the
                snapshot's covered date range, if history is missing.
        """
        row: Dict[str, Any] = {ZONE_COLUMN: zone_id, "hour": target_hour, TARGET_COLUMN: _UNUSED_TARGET_PLACEHOLDER}

        for lag in self.lag_hours:
            if lag not in recent_counts:
                raise InferenceValidationError(
                    f"Cannot compute demand_lag_{lag}h for zone {zone_id} at {target_hour}: "
                    f"missing historical data. This demo serves predictions from a static "
                    f"snapshot covering {self._history_min} to {self._history_max} - "
                    f"target_datetime must fall within that range, with at least "
                    f"{self._max_offset_hours}h of preceding history available."
                )
            row[f"demand_lag_{lag}h"] = recent_counts[lag]

        for window in self.rolling_windows:
            available = [recent_counts[o] for o in range(1, window + 1) if o in recent_counts]
            if not available:
                raise InferenceValidationError(
                    f"Cannot compute demand_rolling_mean_{window}h for zone {zone_id} at "
                    f"{target_hour}: no historical data available in that window. Snapshot "
                    f"covers {self._history_min} to {self._history_max}."
                )
            row[f"demand_rolling_mean_{window}h"] = float(np.mean(available))

        row["hour_of_day"] = target_hour.hour
        row["day_of_week"] = target_hour.weekday()
        row["is_weekend"] = int(target_hour.weekday() in (5, 6))
        hour_sin, hour_cos = cyclic_encode(pd.Series([row["hour_of_day"]]), period=24)
        weekday_sin, weekday_cos = cyclic_encode(pd.Series([row["day_of_week"]]), period=7)
        row["hour_of_day_sin"], row["hour_of_day_cos"] = hour_sin.iloc[0], hour_cos.iloc[0]
        row["day_of_week_sin"], row["day_of_week_cos"] = weekday_sin.iloc[0], weekday_cos.iloc[0]

        return pd.DataFrame([row])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    predictor = DemandPredictor()
    # Use a timestamp comfortably inside the training data's covered
    # range plus lead time, since this demo can't forecast genuinely
    # future hours beyond its static snapshot (see write-up above).
    sample_target = predictor._history_max - pd.Timedelta(hours=2)
    result = predictor.predict({"zone_id": 0, "target_datetime": sample_target.isoformat()})
    print(result)