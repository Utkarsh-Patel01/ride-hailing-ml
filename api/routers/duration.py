"""
duration.py

Single-pipeline endpoint: POST /predict/duration.

Reuses DurationPredictor from the already-loaded UnifiedTripPipeline
(app.state.pipeline.duration_predictor) rather than instantiating a
second, independent DurationPredictor - avoiding loading the same
model artifact into memory twice.

Unlike the unified /predict endpoint (api/main.py), this endpoint does
NOT catch InferenceValidationError - letting it propagate is what
exercises the handler registered in api/main.py for the first time.
See Phase 14's write-up for why that's the deliberate point of this
file, not an oversight.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request

from api.schemas import DurationPredictionResponse, DurationRequest

router = APIRouter(prefix="/predict", tags=["duration"])


@router.post("/duration", response_model=DurationPredictionResponse)
async def predict_duration(payload: DurationRequest, request: Request) -> Dict[str, Any]:
    """
    Predict trip duration for a single pre-trip request.

    Raises:
        InferenceValidationError: If any input falls outside the range
            src.inference.duration_predictor.DurationPredictor was
            trained on - translated into an HTTP 422 by the handler
            registered in api/main.py, since nothing here catches it.
    """
    duration_predictor = request.app.state.pipeline.duration_predictor
    return duration_predictor.predict(payload.model_dump())