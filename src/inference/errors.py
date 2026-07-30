"""
errors.py

Shared exception type for the inference layer. Every predictor class
(duration, demand, anomaly) raises this same InferenceValidationError
for bad input, so src/inference/unified_pipeline.py can catch
validation failures from any of the three pipelines with one except
clause, rather than needing to import and check three separate
exception types.
"""


class InferenceValidationError(Exception):
    """Raised when a live prediction request fails input validation, by any inference pipeline."""