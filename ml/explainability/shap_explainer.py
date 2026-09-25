"""
SHAP-based explanation for structured credibility features
(Section 12: 'Use SHAP or an appropriate feature-attribution method for
structured credibility features.').

Explanations produced here describe MODEL BEHAVIOUR only - per Section 12
they must be presented as descriptions of what drove the model's output,
not as independent proof that an advertisement is fraudulent, and clearly
distinguished from any external evidence.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ShapExplanation:
    feature_names: list
    shap_values: list
    base_value: float

    def top_factors(self, n: int = 5):
        pairs = sorted(zip(self.feature_names, self.shap_values), key=lambda p: abs(p[1]), reverse=True)
        return [
            {"feature": name, "contribution": float(val), "direction": "increases_fraud_signal" if val > 0 else "decreases_fraud_signal"}
            for name, val in pairs[:n]
        ]

    def to_dict(self):
        return {
            "base_value": self.base_value,
            "top_factors": self.top_factors(),
            "note": "SHAP values describe this model's behaviour on this input; "
                    "they are not independent proof of fraud.",
        }


class CredibilityShapExplainer:
    """
    Wraps a fitted sklearn-style credibility model (any object exposing
    .predict_proba) with a SHAP explainer. Uses KernelExplainer as a
    model-agnostic fallback so this works whether the underlying model is
    logistic regression (credibility-only) or the integrated fusion head.
    """

    def __init__(self, model, background_data, feature_names):
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self._explainer = None

    def _build(self):
        if self._explainer is None:
            import shap
            predict_fn = lambda X: self.model.predict_proba(X)[:, 1]
            self._explainer = shap.KernelExplainer(predict_fn, self.background_data)
        return self._explainer

    def explain_instance(self, x_row: np.ndarray) -> ShapExplanation:
        explainer = self._build()
        shap_values = explainer.shap_values(x_row.reshape(1, -1), nsamples=100)
        values = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        base_value = float(np.atleast_1d(explainer.expected_value)[0])
        return ShapExplanation(
            feature_names=list(self.feature_names),
            shap_values=list(np.ravel(values)),
            base_value=base_value,
        )
