"""
Service layer wrapping ml/credibility/ for use by the Flask routes.
Keeps routes thin and keeps the ml/ package framework-agnostic (Appendix B:
'Maintain separate research and application layers so experiments can be
reproduced independently of the web UI.').
"""
from ml.credibility.credibility_features import build_credibility_vector, summarise_indicators


class CredibilityService:
    def analyse(self, advertisement_record: dict) -> dict:
        vector = build_credibility_vector(advertisement_record)
        summary = summarise_indicators(vector)
        return {"feature_vector": vector, "summary": summary}


credibility_service = CredibilityService()
