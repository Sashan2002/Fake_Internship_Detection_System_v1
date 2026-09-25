"""
Credibility-only classifier (Experiment E4, Section 10) - a simple
supervised model trained solely on the structured credibility vector
(Section 8.2), used to measure how much signal these structured indicators
carry on their own, independent of NLP text representations.
"""
from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ml.credibility.credibility_features import CREDIBILITY_INDICATOR_COLUMNS


@dataclass
class CredibilityModelConfig:
    random_seed: int = 42
    max_iter: int = 1000
    class_weight: str = "balanced"
    feature_columns: list = field(default_factory=lambda: list(CREDIBILITY_INDICATOR_COLUMNS))


class CredibilityOnlyModel:
    def __init__(self, config: CredibilityModelConfig = None):
        self.config = config or CredibilityModelConfig()
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=self.config.max_iter,
                class_weight=self.config.class_weight,
                random_state=self.config.random_seed,
            )),
        ])
        self._fitted = False

    def _select(self, df):
        X = df[self.config.feature_columns].copy()
        return X.fillna(0.0).to_numpy(dtype=float)

    def fit(self, df_train, y_train):
        X = self._select(df_train)
        self.pipeline.fit(X, y_train)
        self._fitted = True
        return self

    def predict_proba(self, df):
        if not self._fitted:
            raise RuntimeError("CredibilityOnlyModel must be fit before predicting")
        X = self._select(df)
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump({"pipeline": self.pipeline, "config": self.config}, path)

    @classmethod
    def load(cls, path: str):
        payload = joblib.load(path)
        instance = cls(payload["config"])
        instance.pipeline = payload["pipeline"]
        instance._fitted = True
        return instance
