"""
anomaly.py

Single-pipeline endpoint: POST /predict/anomaly.

Reuses AnomalyScorer from the already-loaded UnifiedTripPipeline
(app.state.pipeline.anomaly_scorer), following the identical pattern
established in api/routers/duration.py.

This endpoint is the one genuinely worth stress-testing against an
out-of-NYC-bounds request: AnomalyRequest (api/schemas.py) was
deliberately built without NYC-bounds coordinate constraints, and
AnomalyScorer (Phase 13) deliberately does not reject out-of-NYC
coordinates either. A request like that should return 200 with a high
anomaly_score - proving the permissive-validation design holds all the
way through the real HTTP boundary, not just inside the Python classes.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request

from api.schemas import AnomalyRequest, AnomalyScoreResponse

router = APIRouter(prefix="/predict", tags=["anomaly"])


@router.post("/anomaly", response_model=AnomalyScoreResponse)
async def score_anomaly(payload: AnomalyRequest, request: Request) -> Dict[str, Any]:
    """
    Score a single completed trip for anomalousness.

    Raises:
        InferenceValidationError: Only for structurally impossible
            input (non-positive elapsed time, an invalid lat/lon
            value) - see src.inference.anomaly_scorer.AnomalyScorer
            for why validation here is deliberately thin. Translated
            into an HTTP 422 by the handler registered in api/main.py.
    """
    anomaly_scorer = request.app.state.pipeline.anomaly_scorer
    return anomaly_scorer.score(payload.model_dump())