"""
demand.py

Single-pipeline endpoint: POST /predict/demand.

Reuses DemandPredictor from the already-loaded UnifiedTripPipeline
(app.state.pipeline.demand_predictor), following the identical pattern
established in api/routers/duration.py: no exception handling here,
letting InferenceValidationError propagate to the handler registered
in api/main.py.

DemandRequest's cross-field validation (zone_id OR coordinates,
enforced in api/schemas.py via model_validator) means this endpoint
body has nothing left to check by the time it runs - what remains
(an invalid zone_id, an out-of-history target_datetime) is business
logic that correctly still lives inside DemandPredictor, not here.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request

from api.schemas import DemandPredictionResponse, DemandRequest

router = APIRouter(prefix="/predict", tags=["demand"])


@router.post("/demand", response_model=DemandPredictionResponse)
async def predict_demand(payload: DemandRequest, request: Request) -> Dict[str, Any]:
    """
    Forecast pickup demand for a single zone-hour request.

    Raises:
        InferenceValidationError: If the resolved zone id is invalid,
            or if src.inference.demand_predictor.DemandPredictor's
            static history snapshot doesn't cover enough time before
            target_datetime to compute the required lag/rolling
            features - translated into an HTTP 422 by the handler
            registered in api/main.py.
    """
    demand_predictor = request.app.state.pipeline.demand_predictor
    return demand_predictor.predict(payload.model_dump())