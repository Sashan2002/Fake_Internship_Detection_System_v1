"""
Service layer wrapping ml/uncertainty/ for use by the Flask routes.
Thresholds are loaded from backend/config.py, which in turn reads
DEFER_THRESHOLD_LOW/HIGH from the environment - these values must have been
selected on validation data and frozen (Section 13/17), not tuned here.
"""
from ml.uncertainty.uncertainty_estimation import DeferThresholds, classify_with_defer
from backend.config import config


class UncertaintyService:
    def __init__(self):
        self.thresholds = DeferThresholds(
            low=config.DEFER_THRESHOLD_LOW, high=config.DEFER_THRESHOLD_HIGH
        )

    def assess(self, fraud_probability: float) -> dict:
        return classify_with_defer(fraud_probability, self.thresholds)


uncertainty_service = UncertaintyService()
