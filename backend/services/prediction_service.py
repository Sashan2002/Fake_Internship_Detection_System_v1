"""
Prediction service: loads the currently active model artefact and exposes
a single `predict(record)` used by the /api/predict route.

Model selection is intentionally simple and explicit: the baseline
TF-IDF+LogReg model is the default "active" model until the integrated
model has been trained and evaluated per the planned experiment sequence
(Section 10, dev sequence). This avoids ever silently serving an
untrained/placeholder model as if it were the evaluated research model.
"""
import logging

from backend.config import config
from backend.utils.preprocessing import build_combined_text

logger = logging.getLogger(__name__)


class ModelNotTrainedError(RuntimeError):
    """Raised when a prediction is requested but no trained model artefact exists yet."""


class PredictionService:
    def __init__(self):
        self._baseline_model = None
        self._load_error = None

    def _load_baseline(self):
        if self._baseline_model is not None:
            return self._baseline_model
        if not config.BASELINE_MODEL_PATH.exists():
            self._load_error = (
                f"No trained baseline model found at {config.BASELINE_MODEL_PATH}. "
                f"Run `python -m ml.train_baseline` first (see dev sequence step 7-8)."
            )
            return None
        from ml.models.baseline_tfidf import TfidfLogRegBaseline
        self._baseline_model = TfidfLogRegBaseline.load(str(config.BASELINE_MODEL_PATH))
        return self._baseline_model

    def predict(self, advertisement_record: dict) -> dict:
        """
        Returns {"fraud_probability": float, "model_version": str}.
        Raises ModelNotTrainedError if no model artefact is available yet -
        callers (routes) should turn this into a clear 503-style API error
        rather than fabricating a number.
        """
        model = self._load_baseline()
        if model is None:
            raise ModelNotTrainedError(self._load_error)

        text = build_combined_text(advertisement_record)
        fraud_probability = float(model.predict_proba([text])[0])
        return {
            "fraud_probability": fraud_probability,
            "model_version": config.ACTIVE_MODEL_VERSION,
        }


prediction_service = PredictionService()
