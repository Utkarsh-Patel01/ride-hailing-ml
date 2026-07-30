"""
anomaly_scorer.py

Wraps the saved Isolation Forest anomaly detector in a class that
scores a single COMPLETED trip - not a pre-trip request, since anomaly
detection needs the trip's actual outcome (real duration, real
distance) to have already happened.

See this phase's write-up for why validation here is deliberately much
thinner than duration_predictor.py's: this pipeline's entire job is to
flag implausible trips, so rejecting implausible-looking input before
it reaches the model would defeat the purpose. Only genuinely
impossible input (non-positive elapsed time, physically invalid
coordinates) is rejected; everything else - including out-of-NYC
coordinates and unusual durations - is scored, not filtered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.features.spatial_features import haversine_distance_km, manhattan_distance_km
from src.inference.errors import InferenceValidationError
from src.models.anomaly.train_isolation_forest import TripAnomalyDetector, add_anomaly_features
from src.utils.config_loader import load_config, resolve_path

logger = logging.getLogger(__name__)


class AnomalyScorer:
    """
    Loads the saved Isolation Forest anomaly detector and scores
    completed trips for anomalousness.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Args:
            model_path: Override for the model artifact location.
                Defaults to models/anomaly/isolation_forest.joblib.
        """
        config = load_config()
        resolved_path = model_path or (
            resolve_path(config["paths"]["models_dir"]) / "anomaly" / "isolation_forest.joblib"
        )
        self.detector = TripAnomalyDetector.load(resolved_path)
        logger.info("Loaded anomaly detector from %s", resolved_path)

    def score(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single completed trip for anomalousness.

        Args:
            request: Dict with keys:
                pickup_datetime, dropoff_datetime (str or
                    datetime-parseable values) - trip_duration is
                    derived from these directly, the same structural
                    definition Phase 4's cleaning used, rather than
                    accepted as a separately-supplied number that
                    could disagree with the timestamps.
                pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude (floats),
                passenger_count (int).

        Returns:
            Dict with 'anomaly_score' (float, 0-1), 'is_anomaly'
            (bool), and the underlying feature values
            (trip_duration_seconds, trip_distance_km,
            average_speed_kmh, route_efficiency_ratio) so a caller -
            an ops reviewer, or unified_pipeline.py's operational
            insight - can see exactly why a trip was or wasn't flagged.

        Raises:
            InferenceValidationError: If elapsed time is not strictly
                positive, a coordinate is not a physically valid
                lat/lon value, or passenger_count is negative.
                Deliberately does NOT reject out-of-NYC-bounds
                coordinates or unusual durations - see this phase's
                write-up for why.
        """
        trip_duration_seconds = self._compute_duration_seconds(request)
        self._validate_coordinates(request)

        passenger_count = int(request.get("passenger_count", 0))
        if passenger_count < 0:
            raise InferenceValidationError(f"passenger_count cannot be negative, got {passenger_count}.")

        row = self._build_feature_row(request, trip_duration_seconds, passenger_count)
        scored = self.detector.score(row)

        return {
            "anomaly_score": float(scored["anomaly_score"].iloc[0]),
            "is_anomaly": bool(scored["is_anomaly"].iloc[0]),
            "trip_duration_seconds": float(row["trip_duration"].iloc[0]),
            "trip_distance_km": float(row["trip_distance_km"].iloc[0]),
            "average_speed_kmh": float(row["average_speed_kmh"].iloc[0]),
            "route_efficiency_ratio": float(row["route_efficiency_ratio"].iloc[0]),
        }

    def _compute_duration_seconds(self, request: Dict[str, Any]) -> float:
        """
        Derive trip_duration from pickup/dropoff timestamps directly,
        mirroring Phase 4's duration_mismatch structural check, applied
        here as a hard requirement rather than a row-dropping rule.

        Raises:
            InferenceValidationError: If dropoff is not strictly after
                pickup. A non-positive elapsed time isn't a candidate
                anomaly to score - it produces an undefined speed
                (division by zero or negative), so it can't be scored
                at all, not even as an outlier.
        """
        if "pickup_datetime" not in request or "dropoff_datetime" not in request:
            raise InferenceValidationError(
                "Request must include both 'pickup_datetime' and 'dropoff_datetime'."
            )

        pickup = pd.Timestamp(request["pickup_datetime"])
        dropoff = pd.Timestamp(request["dropoff_datetime"])
        duration = (dropoff - pickup).total_seconds()

        if duration <= 0:
            raise InferenceValidationError(
                f"dropoff_datetime must be strictly after pickup_datetime, "
                f"got a computed duration of {duration}s."
            )
        return duration

    def _validate_coordinates(self, request: Dict[str, Any]) -> None:
        """
        Reject only physically impossible coordinates (outside valid
        global lat/lon ranges) - NOT outside-NYC-bounds coordinates,
        which are exactly the kind of GPS anomaly this pipeline exists
        to detect and score, not filter out beforehand.
        """
        required = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
        missing = [f for f in required if f not in request]
        if missing:
            raise InferenceValidationError(f"Request missing coordinate field(s): {missing}.")

        for lat_field in ("pickup_latitude", "dropoff_latitude"):
            value = request[lat_field]
            if not (-90 <= value <= 90):
                raise InferenceValidationError(f"{lat_field}={value!r} is not a valid latitude.")
        for lon_field in ("pickup_longitude", "dropoff_longitude"):
            value = request[lon_field]
            if not (-180 <= value <= 180):
                raise InferenceValidationError(f"{lon_field}={value!r} is not a valid longitude.")

    def _build_feature_row(
        self, request: Dict[str, Any], trip_duration_seconds: float, passenger_count: int
    ) -> pd.DataFrame:
        """
        Assemble the single-row DataFrame the detector expects, calling
        the two specific vectorized distance functions this pipeline
        needs directly (Phase 5) rather than the full
        add_spatial_features/build_feature_dataset orchestration -
        bearing and trip_direction contribute nothing to
        ANOMALY_FEATURE_COLUMNS, so computing them here would be pure
        waste on every request.
        """
        pickup_lat = float(request["pickup_latitude"])
        pickup_lon = float(request["pickup_longitude"])
        dropoff_lat = float(request["dropoff_latitude"])
        dropoff_lon = float(request["dropoff_longitude"])

        row = pd.DataFrame([{
            "trip_duration": trip_duration_seconds,
            "trip_distance_km": haversine_distance_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon),
            "manhattan_distance_km": manhattan_distance_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon),
            "passenger_count": passenger_count,
        }])

        return add_anomaly_features(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scorer = AnomalyScorer()

    plausible = scorer.score({
        "pickup_datetime": "2016-03-14 17:00:00",
        "dropoff_datetime": "2016-03-14 17:10:00",
        "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7489, "dropoff_longitude": -73.9680,
        "passenger_count": 1,
    })
    print("Plausible trip:", plausible)

    implausible = scorer.score({
        "pickup_datetime": "2016-03-14 17:00:00",
        "dropoff_datetime": "2016-03-14 17:01:00",  # 1 minute
        "pickup_latitude": 40.6413, "pickup_longitude": -73.7781,   # JFK
        "dropoff_latitude": 40.7580, "dropoff_longitude": -73.9855,  # Times Square, 1 min later
        "passenger_count": 1,
    })
    print("Implausible trip:", implausible)