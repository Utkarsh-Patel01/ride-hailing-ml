"""
duration_predictor.py

Wraps the saved trip-duration stacking ensemble in a class that accepts
a raw trip request - the shape an API caller would actually send - and
returns a duration prediction in seconds.

Reuses build_feature_dataset (Phase 5), FEATURE_COLUMNS, and
predict_seconds (Phase 8) exactly as training used them, so training
and inference can never silently compute features or the inverse
transform differently.

See this phase's write-up for why VENDOR_ID_CODES and
STORE_AND_FWD_FLAG_CODES are hardcoded rather than derived via
.cat.codes on a single-row DataFrame - the latter silently produces
wrong codes at inference time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.features.build_features import build_feature_dataset
from src.models.duration.train_baseline import FEATURE_COLUMNS, predict_seconds
from src.utils.config_loader import load_config, resolve_path
from src.inference.errors import InferenceValidationError
logger = logging.getLogger(__name__)

# Fixed category encodings mirroring what pandas' .cat.codes assigned
# during training on the full dataset (see load_data.py: vendor_id and
# store_and_fwd_flag were loaded as 'category' dtype there). Pandas
# sorts categories by value (numeric ascending / string alphabetical)
# when a column is cast to 'category' with no explicit category order,
# so these codes are a fixed, verifiable fact about this dataset - see
# this phase's "How to test" section for a test that proves it, rather
# than trusting this comment alone.
VENDOR_ID_CODES: Dict[int, int] = {1: 0, 2: 1}
STORE_AND_FWD_FLAG_CODES: Dict[str, int] = {"N": 0, "Y": 1}

# store_and_fwd_flag reflects whether the vehicle lost connectivity
# DURING the ride - it cannot be honestly known before a trip starts.
# Defaulting to the training set's overwhelming majority value is a
# small, documented approximation, not a silent assumption.
_DEFAULT_STORE_AND_FWD_FLAG = "N"



class DurationPredictor:
    """
    Loads the saved trip-duration stacking ensemble and serves
    predictions from raw trip requests.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Args:
            model_path: Override for the model artifact location.
                Defaults to models/duration/stacking_ensemble.joblib,
                the artifact Phase 9 established as this pipeline's
                final model.
        """
        config = load_config()

        resolved_path = model_path or (
            resolve_path(config["paths"]["models_dir"]) / "duration" / "stacking_ensemble.joblib"
        )
        self.model = joblib.load(resolved_path)

        self.min_duration_seconds: float = config["data"]["min_trip_duration_seconds"]
        self.max_passenger_count: int = config["data"]["max_passenger_count"]
        self.nyc_lat_bounds = tuple(config["data"]["nyc_lat_bounds"])
        self.nyc_lon_bounds = tuple(config["data"]["nyc_lon_bounds"])

        logger.info("Loaded duration model from %s", resolved_path)

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict trip duration for a single raw trip request.

        Args:
            request: Dict with keys:
                pickup_datetime (str or datetime-parseable value),
                pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude (floats),
                passenger_count (int),
                vendor_id (int, 1 or 2),
                store_and_fwd_flag (optional str, "N" or "Y" -
                    defaults to "N" if omitted; see this phase's
                    write-up for why this is honest, not a shortcut).

        Returns:
            Dict with 'predicted_duration_seconds' (float),
            'predicted_duration_minutes' (float), and 'model_name'.

        Raises:
            InferenceValidationError: If any input value is out of the
                range this model was trained on (see _validate_request).
        """
        self._validate_request(request)

        df = self._build_request_dataframe(request)
        df = build_feature_dataset(df)
        df = self._encode_categoricals(df)

        X = df[FEATURE_COLUMNS]
        predicted_seconds = predict_seconds(self.model, X, self.min_duration_seconds)[0]

        return {
            "predicted_duration_seconds": float(predicted_seconds),
            "predicted_duration_minutes": float(predicted_seconds) / 60.0,
            "model_name": "stacking_ensemble",
        }

    def _validate_request(self, request: Dict[str, Any]) -> None:
        """
        Reject a request outside the ranges this model was trained on.

        Reuses the same NYC bounding box and passenger-count limits
        Phase 4 used to clean training data - here as input validation
        rather than row-dropping, since a live request can't just be
        silently discarded the way a bad training row could be.

        Raises:
            InferenceValidationError: Naming exactly which check failed.
        """
        passenger_count = request.get("passenger_count")
        if passenger_count is None or not (1 <= passenger_count <= self.max_passenger_count):
            raise InferenceValidationError(
                f"passenger_count must be between 1 and {self.max_passenger_count}, "
                f"got {passenger_count!r}."
            )

        vendor_id = request.get("vendor_id")
        if vendor_id not in VENDOR_ID_CODES:
            raise InferenceValidationError(
                f"vendor_id must be one of {list(VENDOR_ID_CODES)}, got {vendor_id!r}."
            )

        flag = request.get("store_and_fwd_flag", _DEFAULT_STORE_AND_FWD_FLAG)
        if flag not in STORE_AND_FWD_FLAG_CODES:
            raise InferenceValidationError(
                f"store_and_fwd_flag must be one of {list(STORE_AND_FWD_FLAG_CODES)}, got {flag!r}."
            )

        for coord_name, bounds in [
            ("pickup_latitude", self.nyc_lat_bounds), ("dropoff_latitude", self.nyc_lat_bounds),
        ]:
            value = request.get(coord_name)
            if value is None or not (bounds[0] <= value <= bounds[1]):
                raise InferenceValidationError(
                    f"{coord_name}={value!r} is outside expected NYC range {bounds}."
                )

        for coord_name, bounds in [
            ("pickup_longitude", self.nyc_lon_bounds), ("dropoff_longitude", self.nyc_lon_bounds),
        ]:
            value = request.get(coord_name)
            if value is None or not (bounds[0] <= value <= bounds[1]):
                raise InferenceValidationError(
                    f"{coord_name}={value!r} is outside expected NYC range {bounds}."
                )

    def _build_request_dataframe(self, request: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert a validated raw request dict into a single-row
        DataFrame with the columns build_feature_dataset expects.
        """
        row = {
            "pickup_datetime": pd.to_datetime(request["pickup_datetime"]),
            "pickup_latitude": float(request["pickup_latitude"]),
            "pickup_longitude": float(request["pickup_longitude"]),
            "dropoff_latitude": float(request["dropoff_latitude"]),
            "dropoff_longitude": float(request["dropoff_longitude"]),
            "passenger_count": int(request["passenger_count"]),
            "vendor_id": int(request["vendor_id"]),
            "store_and_fwd_flag": request.get("store_and_fwd_flag", _DEFAULT_STORE_AND_FWD_FLAG),
        }
        return pd.DataFrame([row])

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fixed VENDOR_ID_CODES / STORE_AND_FWD_FLAG_CODES
        mappings - NOT .astype('category').cat.codes, which would
        silently produce wrong codes on a single-row DataFrame (see
        this phase's write-up).
        """
        df = df.copy()
        df["vendor_id_code"] = df["vendor_id"].map(VENDOR_ID_CODES)
        df["store_and_fwd_flag_code"] = df["store_and_fwd_flag"].map(STORE_AND_FWD_FLAG_CODES)
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    predictor = DurationPredictor()
    result = predictor.predict({
        "pickup_datetime": "2016-03-14 17:30:00",
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.6413,
        "dropoff_longitude": -73.7781,
        "passenger_count": 1,
        "vendor_id": 2,
    })
    print(result)