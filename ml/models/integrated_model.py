"""
Integrated NLP + employer-credibility model (Section 8.3, Experiment E6).

Architecture: late fusion. The transformer produces a fraud-probability-
style representation from text; the credibility model (or raw credibility
vector) contributes the structured signal; a small fusion classifier
combines both into a final probability. This is deliberately implemented
as a controlled, swappable pipeline so that:
  * E6 (integrated)         -> use_nlp=True,  use_credibility=True
  * E7 (ablation, no cred.) -> use_nlp=True,  use_credibility=False
  * E8 (ablation, no NLP)   -> use_nlp=False, use_credibility=True
can all be run from the same class (Section 8.3: 'implemented as a
controlled experiment so that the contribution of employer credibility
features can be measured against NLP-only and credibility-only
alternatives.').
"""
from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.credibility.credibility_features import CREDIBILITY_INDICATOR_COLUMNS


@dataclass
class IntegratedModelConfig:
    use_nlp: bool = True
    use_credibility: bool = True
    random_seed: int = 42
    max_iter: int = 1000
    credibility_columns: list = field(default_factory=lambda: list(CREDIBILITY_INDICATOR_COLUMNS))


class IntegratedFusionModel:
    """
    Fusion layer only. It expects the NLP probability/embedding and the raw
    credibility vector to already have been produced upstream (by a
    transformer model and credibility_features.build_credibility_vector
    respectively) and combines them via a small logistic-regression fusion
    head - keeping the expensive transformer forward pass decoupled from
    the fusion experiment so E6/E7/E8 can reuse cached NLP outputs.
    """

    def __init__(self, config: IntegratedModelConfig = None):
        self.config = config or IntegratedModelConfig()
        if not (self.config.use_nlp or self.config.use_credibility):
            raise ValueError("At least one of use_nlp/use_credibility must be True")
        self.fusion_clf = LogisticRegression(
            max_iter=self.config.max_iter, random_state=self.config.random_seed
        )
        self._fitted = False

    def _assemble_features(self, nlp_probs: np.ndarray, credibility_df) -> np.ndarray:
        blocks = []
        if self.config.use_nlp:
            blocks.append(np.asarray(nlp_probs).reshape(-1, 1))
        if self.config.use_credibility:
            cred = credibility_df[self.config.credibility_columns].fillna(0.0).to_numpy(dtype=float)
            blocks.append(cred)
        return np.hstack(blocks)

    def fit(self, nlp_probs, credibility_df, y):
        X = self._assemble_features(nlp_probs, credibility_df)
        self.fusion_clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, nlp_probs, credibility_df):
        if not self._fitted:
            raise RuntimeError("IntegratedFusionModel must be fit before predicting")
        X = self._assemble_features(nlp_probs, credibility_df)
        return self.fusion_clf.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump({"fusion_clf": self.fusion_clf, "config": self.config}, path)

    @classmethod
    def load(cls, path: str):
        payload = joblib.load(path)
        instance = cls(payload["config"])
        instance.fusion_clf = payload["fusion_clf"]
        instance._fitted = True
        return instance
