"""
schemas.py

Pydantic request/response models forming the HTTP boundary in front of
src.inference's plain-dict-based predictor classes.

See Phase 14's write-up for why coordinate fields here validate only
globally-physical lat/lon ranges, not NYC-specific bounds - the
NYC-specific and anomaly-permissive validation decisions already made
in Phase 13 must not be silently re-implemented (and in AnomalyScorer's
case, contradicted) at this layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# ── Shared location mixins ──────────────────────────────────────────
# Composed via multiple inheritance into the request schemas below, so
# every schema needing pickup/dropoff coordinates gets identical field
# constraints from one place rather than four independently-typed copies.


class PickupLocation(BaseModel):
    pickup_latitude: float = Field(ge=-90, le=90, description="Pickup latitude, degrees.")
    pickup_longitude: float = Field(ge=-180, le=180, description="Pickup longitude, degrees.")


class DropoffLocation(BaseModel):
    dropoff_latitude: float = Field(ge=-90, le=90, description="Dropoff latitude, degrees.")
    dropoff_longitude: float = Field(ge=-180, le=180, description="Dropoff longitude, degrees.")


# ── Request schemas ──────────────────────────────────────────────────


class DurationRequest(PickupLocation, DropoffLocation):
    """Request body for POST /predict/duration - a pre-trip ETA request."""

    pickup_datetime: datetime
    passenger_count: int = Field(ge=1, description="Must be >= 1; a taxi cannot dispatch empty.")
    vendor_id: int
    store_and_fwd_flag: Optional[Literal["N", "Y"]] = Field(
        default=None,
        description="Unknowable before a trip starts; defaults to 'N' inside DurationPredictor if omitted.",
    )


class DemandRequest(BaseModel):
    """
    Request body for POST /predict/demand.

    Requires EITHER zone_id OR both pickup_latitude/pickup_longitude -
    enforced below via a model_validator, since Field() constraints
    alone can't express a cross-field "at least one of these" rule.
    """

    target_datetime: datetime
    zone_id: Optional[int] = Field(default=None, ge=0)
    pickup_latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    pickup_longitude: Optional[float] = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def _require_zone_or_coordinates(self) -> "DemandRequest":
        has_coords = self.pickup_latitude is not None and self.pickup_longitude is not None
        if self.zone_id is None and not has_coords:
            raise ValueError(
                "Provide either 'zone_id', or both 'pickup_latitude' and 'pickup_longitude'."
            )
        return self


class AnomalyRequest(PickupLocation, DropoffLocation):
    """
    Request body for POST /predict/anomaly - a COMPLETED trip.

    Deliberately does NOT constrain coordinates beyond the global
    physical range inherited from PickupLocation/DropoffLocation - see
    this phase's write-up for why an NYC-bounds constraint here would
    silently defeat AnomalyScorer's design.
    """

    pickup_datetime: datetime
    dropoff_datetime: datetime
    passenger_count: int = Field(default=0, ge=0)


class TripRequest(PickupLocation, DropoffLocation):
    """
    Request body for POST /predict - the unified endpoint.

    dropoff_datetime is the ONLY field that toggles pre-trip vs
    post-trip mode (see UnifiedTripPipeline, Phase 13); dropoff
    coordinates are required unconditionally, since DurationPredictor
    needs a destination even for a pre-trip ETA request.
    """

    pickup_datetime: datetime
    dropoff_datetime: Optional[datetime] = Field(
        default=None, description="Omit for a pre-trip request; supply for a completed-trip request."
    )
    passenger_count: int = Field(ge=1)
    vendor_id: int
    store_and_fwd_flag: Optional[Literal["N", "Y"]] = None
    zone_id: Optional[int] = Field(default=None, ge=0, description="Optional demand-zone override.")
    target_datetime: Optional[datetime] = Field(
        default=None, description="Optional demand-forecast hour override; defaults to pickup_datetime."
    )


# ── Response schemas ─────────────────────────────────────────────────
# Field names below match each predictor's returned dict keys exactly
# (see Phase 13) - see this phase's write-up for why that's deliberate.


class DurationPredictionResponse(BaseModel):
    predicted_duration_seconds: float
    predicted_duration_minutes: float
    model_name: str


class DemandPredictionResponse(BaseModel):
    predicted_pickup_count: float
    zone_id: int
    target_hour: str
    model_name: str


class AnomalyScoreResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    trip_duration_seconds: float
    trip_distance_km: float
    average_speed_kmh: float
    route_efficiency_ratio: float


class ComponentStatus(BaseModel):
    status: Literal["ok", "unavailable"]
    reason: Optional[str] = None


class UnifiedPredictionResponse(BaseModel):
    request_mode: Literal["pre_trip", "post_trip"]
    duration: Optional[DurationPredictionResponse] = None
    demand: Optional[DemandPredictionResponse] = None
    anomaly: Optional[AnomalyScoreResponse] = None
    component_status: Dict[str, ComponentStatus]
    operational_insight: str


class ErrorResponse(BaseModel):
    """Shape returned for any InferenceValidationError, across every endpoint."""

    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"