"""
unified_pipeline.py

Composes the three independent inference pipelines (duration, demand,
anomaly) into the single response your original spec's "Final
Inference Layer" asked for.

See this phase's write-up for the two ideas this file is built around:
  1. The three pipelines answer questions at genuinely different
     points in a trip's lifecycle (pre-trip vs post-trip), so which
     ones can even run depends on what the request actually contains.
  2. Each component can fail for its own domain-specific reason
     without taking the other two down - this file reports partial
     results with per-component status, and in the post-trip case,
     actively cross-references results between components (a rejected
     duration prediction becomes evidence for the anomaly result, not
     just a gap in the response).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from src.inference.anomaly_scorer import AnomalyScorer
from src.inference.demand_predictor import DemandPredictor
from src.inference.duration_predictor import DurationPredictor
from src.inference.errors import InferenceValidationError

logger = logging.getLogger(__name__)

_DURATION_REQUIRED_FIELDS = [
    "pickup_datetime", "pickup_latitude", "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude", "passenger_count", "vendor_id",
]
_ANOMALY_REQUIRED_FIELDS = [
    "pickup_datetime", "dropoff_datetime", "pickup_latitude",
    "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
]

# Illustrative demo thresholds only - a real deployment would calibrate
# these against each zone's actual demand distribution and each route's
# actual duration variance, not one hardcoded global cutoff.
_DEMAND_QUIET_THRESHOLD = 5.0
_DEMAND_BUSY_THRESHOLD = 20.0
_DURATION_DEVIATION_FLAG_PCT = 50.0

ComponentResult = Tuple[Optional[Dict[str, Any]], Dict[str, str]]


class UnifiedTripPipeline:
    """
    Loads all three inference pipelines once and composes their
    results into a single response per request, with graceful
    per-component degradation and cross-pipeline synthesis for
    completed trips.
    """

    def __init__(self) -> None:
        self.duration_predictor = DurationPredictor()
        self.demand_predictor = DemandPredictor()
        self.anomaly_scorer = AnomalyScorer()
        logger.info("Unified trip pipeline ready (duration + demand + anomaly).")

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a composed prediction for a single trip request.

        Args:
            request: A raw request dict. Which fields are required
                depends on the request's lifecycle stage:
                - Always: pickup_datetime, pickup_latitude/longitude.
                - For duration: also dropoff_latitude/longitude,
                  passenger_count, vendor_id.
                - For demand: also either zone_id, or the pickup
                  coordinates above (already required); optionally a
                  separate target_datetime (defaults to pickup_datetime).
                - For anomaly (only runs if dropoff_datetime is
                  present): also dropoff_latitude/longitude.

        Returns:
            Dict with 'request_mode' ('pre_trip' or 'post_trip'),
            'duration', 'demand', 'anomaly' (each a result dict or
            None), 'component_status' (per-component 'ok'/'unavailable'
            + reason), and 'operational_insight' (a synthesized,
            human-readable summary).
        """
        is_completed_trip = bool(request.get("dropoff_datetime"))

        duration_result, duration_status = self._run_duration(request)
        demand_result, demand_status = self._run_demand(request)

        if is_completed_trip:
            anomaly_result, anomaly_status = self._run_anomaly(request)
        else:
            anomaly_result, anomaly_status = None, {
                "status": "unavailable",
                "reason": (
                    "Trip not yet completed - no dropoff_datetime supplied. "
                    "Anomaly detection requires a completed trip's actual outcome."
                ),
            }

        insight = self._build_operational_insight(
            duration_result, duration_status,
            demand_result, demand_status,
            anomaly_result, anomaly_status,
            is_completed_trip,
        )

        return {
            "request_mode": "post_trip" if is_completed_trip else "pre_trip",
            "duration": duration_result,
            "demand": demand_result,
            "anomaly": anomaly_result,
            "component_status": {
                "duration": duration_status,
                "demand": demand_status,
                "anomaly": anomaly_status,
            },
            "operational_insight": insight,
        }

    def _run_duration(self, request: Dict[str, Any]) -> ComponentResult:
        """Call DurationPredictor if required fields are present, catching validation failures."""
        missing = [f for f in _DURATION_REQUIRED_FIELDS if request.get(f) is None]
        if missing:
            return None, {"status": "unavailable", "reason": f"missing required field(s): {missing}"}
        try:
            return self.duration_predictor.predict(request), {"status": "ok"}
        except InferenceValidationError as e:
            return None, {"status": "unavailable", "reason": str(e)}

    def _run_demand(self, request: Dict[str, Any]) -> ComponentResult:
        """
        Call DemandPredictor, defaulting target_datetime to
        pickup_datetime (forecasting demand for the pickup hour itself)
        when not separately supplied.
        """
        target_datetime = request.get("target_datetime") or request.get("pickup_datetime")
        has_zone = request.get("zone_id") is not None
        has_coords = (
            request.get("pickup_latitude") is not None
            and request.get("pickup_longitude") is not None
        )

        if target_datetime is None or not (has_zone or has_coords):
            return None, {
                "status": "unavailable",
                "reason": (
                    "missing required field(s): need target_datetime (or pickup_datetime as "
                    "a fallback) plus either zone_id or pickup_latitude/pickup_longitude."
                ),
            }

        demand_request: Dict[str, Any] = {"target_datetime": target_datetime}
        if has_zone:
            demand_request["zone_id"] = request["zone_id"]
        else:
            demand_request["pickup_latitude"] = request["pickup_latitude"]
            demand_request["pickup_longitude"] = request["pickup_longitude"]

        try:
            return self.demand_predictor.predict(demand_request), {"status": "ok"}
        except InferenceValidationError as e:
            return None, {"status": "unavailable", "reason": str(e)}

    def _run_anomaly(self, request: Dict[str, Any]) -> ComponentResult:
        """Call AnomalyScorer if required fields are present, catching validation failures."""
        missing = [f for f in _ANOMALY_REQUIRED_FIELDS if request.get(f) is None]
        if missing:
            return None, {"status": "unavailable", "reason": f"missing required field(s): {missing}"}
        try:
            return self.anomaly_scorer.score(request), {"status": "ok"}
        except InferenceValidationError as e:
            return None, {"status": "unavailable", "reason": str(e)}

    def _build_operational_insight(
        self,
        duration_result: Optional[Dict], duration_status: Dict[str, str],
        demand_result: Optional[Dict], demand_status: Dict[str, str],
        anomaly_result: Optional[Dict], anomaly_status: Dict[str, str],
        is_completed_trip: bool,
    ) -> str:
        """
        Synthesize the available component results into one
        human-readable insight. Pre-trip and post-trip requests get
        structurally different insights, since they're answering
        different operational questions (see this phase's write-up).
        """
        parts = []

        if not is_completed_trip:
            if duration_result:
                parts.append(
                    f"Estimated trip duration: {duration_result['predicted_duration_minutes']:.1f} minutes."
                )
            else:
                parts.append(f"Duration prediction unavailable ({duration_status['reason']}).")

            if demand_result:
                count = demand_result["predicted_pickup_count"]
                tier = (
                    "quiet" if count < _DEMAND_QUIET_THRESHOLD
                    else "busy" if count >= _DEMAND_BUSY_THRESHOLD
                    else "moderate"
                )
                parts.append(
                    f"Pickup zone {demand_result['zone_id']} is forecast at ~{count:.1f} "
                    f"pickups in the next hour ({tier} demand)."
                )
            else:
                parts.append(f"Zonal demand forecast unavailable ({demand_status['reason']}).")

            return " ".join(parts)

        # Post-trip: anomaly is the headline signal; duration (when
        # available) becomes a cross-check against it rather than an
        # independent second number.
        if anomaly_result:
            score = anomaly_result["anomaly_score"]
            flagged = anomaly_result["is_anomaly"]
            actual_seconds = anomaly_result["trip_duration_seconds"]
            parts.append(
                f"Anomaly score: {score:.3f} "
                f"({'FLAGGED for review' if flagged else 'within normal range'})."
            )

            if duration_result:
                predicted_seconds = duration_result["predicted_duration_seconds"]
                deviation_pct = abs(actual_seconds - predicted_seconds) / predicted_seconds * 100

                if deviation_pct >= _DURATION_DEVIATION_FLAG_PCT and flagged:
                    parts.append(
                        f"Actual duration ({actual_seconds:.0f}s) deviated {deviation_pct:.0f}% "
                        f"from the pre-trip estimate ({predicted_seconds:.0f}s), reinforcing the "
                        f"anomaly flag - this looks like a genuine data/meter issue, not just "
                        f"normal trip variance."
                    )
                elif deviation_pct >= _DURATION_DEVIATION_FLAG_PCT:
                    parts.append(
                        f"Actual duration deviated {deviation_pct:.0f}% from the pre-trip "
                        f"estimate, but was not flagged anomalous - likely explainable variance "
                        f"(traffic, routing choice) rather than a data error."
                    )
                elif flagged:
                    parts.append(
                        f"Duration matched the pre-trip estimate closely ({deviation_pct:.0f}% "
                        f"deviation), so the anomaly flag is likely driven by distance/speed/"
                        f"routing rather than timing - see route_efficiency_ratio and "
                        f"average_speed_kmh in the anomaly result."
                    )
                else:
                    parts.append(
                        f"Duration matched the pre-trip estimate closely ({deviation_pct:.0f}% "
                        f"deviation) and no anomaly signals were raised - a normal, "
                        f"well-recorded trip."
                    )
            else:
                # The composition payoff this phase's write-up describes:
                # a rejected duration prediction is corroborating context
                # here, not just a gap in the response.
                consistency_note = (
                    "the anomaly flag raised above" if flagged
                    else "this trip, even though it was not flagged as anomalous"
                )
                parts.append(
                    f"Duration prediction unavailable ({duration_status['reason']}). This "
                    f"commonly happens when a trip's coordinates fall outside the range the "
                    f"duration model trained on - notably consistent with {consistency_note}."
                )
        else:
            parts.append(f"Anomaly scoring unavailable ({anomaly_status['reason']}).")

        if demand_result:
            parts.append(
                f"(Context: pickup zone {demand_result['zone_id']} was forecast at "
                f"~{demand_result['predicted_pickup_count']:.1f} pickups/hour around this "
                f"trip's start.)"
            )

        return " ".join(parts)


if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    pipeline = UnifiedTripPipeline()

    # Demo pickup time inside the demand predictor's static snapshot
    # range (see Phase 13's demand_predictor.py for why this matters).
    demo_pickup_time = pipeline.demand_predictor._history_max - pd.Timedelta(hours=2)

    print("\n--- 1. Pre-trip request (rider requesting a ride) ---")
    pre_trip = pipeline.predict({
        "pickup_datetime": demo_pickup_time.isoformat(),
        "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,
        "passenger_count": 1, "vendor_id": 2,
    })
    print(pre_trip["operational_insight"])

    print("\n--- 2. Post-trip: a normal, plausible completed trip ---")
    normal_completed = pipeline.predict({
        "pickup_datetime": demo_pickup_time.isoformat(),
        "dropoff_datetime": (demo_pickup_time + pd.Timedelta(minutes=25)).isoformat(),
        "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,
        "passenger_count": 1, "vendor_id": 2,
    })
    print(normal_completed["operational_insight"])

    print("\n--- 3. Post-trip: implausible timing (real JFK<->Times Sq route, compressed into 1 minute) ---")
    implausible_timing = pipeline.predict({
        "pickup_datetime": demo_pickup_time.isoformat(),
        "dropoff_datetime": (demo_pickup_time + pd.Timedelta(minutes=1)).isoformat(),
        "pickup_latitude": 40.6413, "pickup_longitude": -73.7781,
        "dropoff_latitude": 40.7580, "dropoff_longitude": -73.9855,
        "passenger_count": 1, "vendor_id": 2,
    })
    print(implausible_timing["operational_insight"])

    print("\n--- 4. Post-trip: dropoff far outside NYC (duration can't extrapolate; anomaly still scores it) ---")
    out_of_range = pipeline.predict({
        "pickup_datetime": demo_pickup_time.isoformat(),
        "dropoff_datetime": (demo_pickup_time + pd.Timedelta(minutes=20)).isoformat(),
        "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,
        "dropoff_latitude": 34.0522, "dropoff_longitude": -118.2437,  # Los Angeles
        "passenger_count": 1, "vendor_id": 2,
    })
    print(out_of_range["operational_insight"])
    print("component_status:", out_of_range["component_status"])