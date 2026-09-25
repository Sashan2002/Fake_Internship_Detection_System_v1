"""
Baseline: TF-IDF + Logistic Regression (Experiment E2, Section 10; dev
sequence step 7). This is the first trainable model in the implementation
sequence and the reference point every later model must beat before being
described as an improvement (Section 18: 'Do not claim model superiority
before controlled evaluation.').
"""
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class TfidfBaselineConfig:
    max_features: int = 20000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    random_seed: int = 42
    max_iter: int = 1000
    class_weight: str = "balanced"


class TfidfLogRegBaseline:
    def __init__(self, config: TfidfBaselineConfig = None):
        self.config = config or TfidfBaselineConfig()
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
            )),
            ("clf", LogisticRegression(
                max_iter=self.config.max_iter,
                class_weight=self.config.class_weight,
                random_state=self.config.random_seed,
            )),
        ])
        self._fitted = False

    def fit(self, texts, y):
        self.pipeline.fit(texts, y)
        self._fitted = True
        return self

    def predict_proba(self, texts):
        if not self._fitted:
            raise RuntimeError("Model must be fit before predicting")
        return self.pipeline.predict_proba(texts)[:, 1]

    def save(self, path: str):
        joblib.dump({"pipeline": self.pipeline, "config": self.config}, path)

    @classmethod
    def load(cls, path: str):
        payload = joblib.load(path)
        instance = cls(payload["config"])
        instance.pipeline = payload["pipeline"]
        instance._fitted = True
        return instance
