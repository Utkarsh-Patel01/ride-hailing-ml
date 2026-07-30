"""
main.py

The FastAPI application: loads UnifiedTripPipeline once at startup
(via a lifespan hook, not per-request or at import time), exposes a
health check and the unified /predict endpoint, and translates
src.inference.errors.InferenceValidationError into a proper HTTP 422
response.

api/routers/duration.py, demand.py, and anomaly.py (Phase 14's next
three files) each register one additional endpoint against this same
`app` object via app.include_router(...).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.schemas import ErrorResponse, HealthResponse, TripRequest, UnifiedPredictionResponse
from api.routers import anomaly as anomaly_router
from api.routers import demand as demand_router
from api.routers import duration as duration_router
from src.inference.errors import InferenceValidationError
from src.inference.unified_pipeline import UnifiedTripPipeline
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Load UnifiedTripPipeline exactly once when the server starts, not
    per-request and not merely on module import - see this phase's
    write-up for why both of those alternatives are wrong.
    """
    logger.info("Loading unified trip pipeline (duration + demand + anomaly models)...")
    app.state.pipeline = UnifiedTripPipeline()
    logger.info("Startup complete - ready to serve requests.")
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Ride-Hailing ML API",
    description="Trip duration, zonal demand forecasting, and anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(duration_router.router)
app.include_router(demand_router.router)
app.include_router(anomaly_router.router)

@app.exception_handler(InferenceValidationError)
async def handle_inference_validation_error(
    request: Request, exc: InferenceValidationError
) -> JSONResponse:
    """
    Translate a business-logic rejection from src.inference into a 422
    response. Deliberately separate from Pydantic's own request-
    validation errors - see this phase's write-up for why the two
    failure layers intentionally produce different response shapes.
    """
    return JSONResponse(status_code=422, content=ErrorResponse(detail=str(exc)).model_dump())


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for anything not already handled: logs the full
    exception server-side for debugging, but never leaks internal
    details (a stack trace, a library error message) across the HTTP
    boundary to the caller.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "An unexpected internal error occurred."})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check - returns 200 the moment the process is up, independent of model state."""
    return HealthResponse()


@app.post("/predict", response_model=UnifiedPredictionResponse)
async def predict_unified(payload: TripRequest, request: Request) -> Dict[str, Any]:
    """
    Unified prediction endpoint: pre-trip (duration + demand) or
    post-trip (+ anomaly, cross-referenced against duration) depending
    on whether dropoff_datetime is supplied - see
    src.inference.unified_pipeline.UnifiedTripPipeline for the full
    lifecycle-stage logic.
    """
    pipeline: UnifiedTripPipeline = request.app.state.pipeline
    return pipeline.predict(payload.model_dump())


if __name__ == "__main__":
    config = load_config()
    uvicorn.run(
        "api.main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True,
    )