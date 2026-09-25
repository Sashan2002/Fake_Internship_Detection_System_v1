"""
Additional classical ML baselines on TF-IDF features (Experiment E3,
Section 10), for comparison against plain Logistic Regression. Kept
separate from baseline_tfidf.py so E2 stays a minimal, auditable reference
implementation while this module can be extended with more classifiers
without touching that reference.
"""
from dataclasses import dataclass

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

SUPPORTED_CLASSIFIERS = ("random_forest", "linear_svm")


@dataclass
class ClassicalMlConfig:
    classifier: str = "random_forest"
    max_features: int = 20000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    random_seed: int = 42
    class_weight: str = "balanced"


def _build_classifier(config: ClassicalMlConfig):
    if config.classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight=config.class_weight,
            random_state=config.random_seed, n_jobs=-1,
        )
    if config.classifier == "linear_svm":
        # LinearSVC has no predict_proba; calibrate to get probabilities so
        # this model can be evaluated with the same probability-based
        # metrics (ROC-AUC, PR-AUC, calibration) as the others.
        base = LinearSVC(class_weight=config.class_weight, random_state=config.random_seed)
        return CalibratedClassifierCV(base, cv=3)
    raise ValueError(f"Unsupported classifier '{config.classifier}'. Choose from {SUPPORTED_CLASSIFIERS}")


class ClassicalMlBaseline:
    def __init__(self, config: ClassicalMlConfig = None):
        self.config = config or ClassicalMlConfig()
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
            )),
            ("clf", _build_classifier(self.config)),
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
