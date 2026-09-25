"""
Model-facing unit tests: baseline model shape, save/load, metrics
computation (Section 11), and reproducibility (Section 17).
"""
import numpy as np
import pytest

from ml.models.baseline_tfidf import TfidfLogRegBaseline, TfidfBaselineConfig
from ml.evaluation.metrics import compute_metrics


LEGIT_TEXTS = [
    "Great structured internship programme with mentorship and training.",
    "Join our engineering team as a summer intern, real projects, senior mentors.",
    "Marketing internship with clear responsibilities and a defined salary range.",
] * 5

FRAUD_TEXTS = [
    "Earn money fast, no experience needed, send your bank details today!!!",
    "Work from home urgent hiring, contact immediately for easy money.",
    "Limited slots, no interview required, send personal information now.",
] * 5


def _toy_dataset():
    texts = LEGIT_TEXTS + FRAUD_TEXTS
    labels = [0] * len(LEGIT_TEXTS) + [1] * len(FRAUD_TEXTS)
    return texts, labels


def test_baseline_model_predicts_probabilities_in_range():
    texts, labels = _toy_dataset()
    model = TfidfLogRegBaseline(TfidfBaselineConfig(max_features=500, min_df=1))
    model.fit(texts, labels)

    probs = model.predict_proba(texts)
    assert len(probs) == len(texts)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_baseline_model_raises_before_fit():
    model = TfidfLogRegBaseline()
    with pytest.raises(RuntimeError):
        model.predict_proba(["some text"])


def test_baseline_model_save_and_load(tmp_path):
    texts, labels = _toy_dataset()
    model = TfidfLogRegBaseline(TfidfBaselineConfig(max_features=500, min_df=1))
    model.fit(texts, labels)

    out_path = tmp_path / "model.joblib"
    model.save(str(out_path))
    assert out_path.exists()

    loaded = TfidfLogRegBaseline.load(str(out_path))
    original_probs = model.predict_proba(texts)
    loaded_probs = loaded.predict_proba(texts)
    assert np.allclose(original_probs, loaded_probs)


def test_compute_metrics_reasonable_on_separable_toy_data():
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.05, 0.1, 0.2, 0.8, 0.9, 0.95]
    report = compute_metrics(y_true, y_prob)
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.roc_auc == 1.0
    assert report.n_samples == 6


def test_metrics_do_not_crash_on_single_class_edge_case():
    report = compute_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert report.n_samples == 3
