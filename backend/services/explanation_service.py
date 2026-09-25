"""
Explanation service: produces user-readable, model-behaviour-only
explanations (Section 12). Falls back to a transparent "not yet available"
response rather than fabricating an explanation when no trained model or
explainer background data is available - honesty about limitations matters
as much here as the explanation itself (Section 12: 'Treat explanations as
descriptions of model behaviour, not proof that an advertisement is
fraudulent.').
"""
import logging

from backend.config import config

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This explanation describes which factors most influenced the model's "
    "own prediction for this specific submission. It is not independent "
    "verification of the employer or proof that the advertisement is "
    "fraudulent."
)


class ExplanationService:
    def __init__(self):
        self._lime_explainer = None

    def _get_lime_explainer(self):
        if self._lime_explainer is None:
            if not config.BASELINE_MODEL_PATH.exists():
                return None
            from ml.models.baseline_tfidf import TfidfLogRegBaseline
            from ml.explainability.lime_explainer import TextLimeExplainer

            model = TfidfLogRegBaseline.load(str(config.BASELINE_MODEL_PATH))
            self._lime_explainer = TextLimeExplainer(model.pipeline.predict_proba)
        return self._lime_explainer

    def explain_text(self, combined_text: str) -> dict:
        explainer = self._get_lime_explainer()
        if explainer is None:
            return {
                "available": False,
                "reason": "No trained model available yet for explanation generation.",
                "disclaimer": DISCLAIMER,
            }
        try:
            explanation = explainer.explain_instance(combined_text, num_features=8)
            result = explanation.to_dict()
            result["available"] = True
            result["disclaimer"] = DISCLAIMER
            return result
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("LIME explanation failed")
            return {"available": False, "reason": str(exc), "disclaimer": DISCLAIMER}

    def explain_credibility(self, credibility_vector: dict) -> dict:
        """
        Lightweight, explainer-free fallback: ranks credibility indicators
        by whether they are 'missing' completeness signals, since a full
        SHAP KernelExplainer needs a fitted credibility-only model and
        background dataset that may not exist yet in a fresh setup. Once
        ml/credibility/credibility_model.py has been trained, swap this for
        ml/explainability/shap_explainer.CredibilityShapExplainer.
        """
        missing = [k for k, v in credibility_vector.items() if k.endswith("_present") and not v]
        return {
            "available": True,
            "method": "rule_based_fallback",
            "missing_indicators": missing,
            "disclaimer": DISCLAIMER,
        }


explanation_service = ExplanationService()
