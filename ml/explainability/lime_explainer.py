"""
LIME-based explanation for text-level decisions
(Section 12: 'Use LIME or another suitable token/text-level explanation
method for textual decisions where technically appropriate.').

Same interpretation caveat as shap_explainer.py: outputs describe model
behaviour, not proof of fraud.
"""
from dataclasses import dataclass


@dataclass
class LimeExplanation:
    tokens_with_weights: list  # list of (token, weight)

    def top_factors(self, n: int = 5):
        ranked = sorted(self.tokens_with_weights, key=lambda t: abs(t[1]), reverse=True)
        return [
            {"token": tok, "contribution": float(w), "direction": "increases_fraud_signal" if w > 0 else "decreases_fraud_signal"}
            for tok, w in ranked[:n]
        ]

    def to_dict(self):
        return {
            "top_factors": self.top_factors(),
            "note": "LIME highlights tokens influential to this model's prediction "
                    "on this input; it is not independent proof of fraud.",
        }


class TextLimeExplainer:
    """
    Wraps any callable `predict_proba_fn(list[str]) -> np.ndarray[n, 2]`
    (e.g. TfidfLogRegBaseline.pipeline.predict_proba or a transformer
    wrapper's predict_proba adapted to 2-column output) with LIME's
    text explainer.
    """

    def __init__(self, predict_proba_fn, class_names=("legitimate", "fraudulent")):
        self.predict_proba_fn = predict_proba_fn
        self.class_names = list(class_names)
        self._explainer = None

    def _build(self):
        if self._explainer is None:
            from lime.lime_text import LimeTextExplainer
            self._explainer = LimeTextExplainer(class_names=self.class_names)
        return self._explainer

    def explain_instance(self, text: str, num_features: int = 10) -> LimeExplanation:
        explainer = self._build()
        exp = explainer.explain_instance(
            text, self.predict_proba_fn, num_features=num_features, labels=(1,)
        )
        pairs = exp.as_list(label=1)
        return LimeExplanation(tokens_with_weights=pairs)
