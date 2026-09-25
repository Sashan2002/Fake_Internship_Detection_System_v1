"""
Evaluation metrics (Section 11). Accuracy is intentionally NOT the primary
headline metric: 'Because the EMSCAD target is strongly imbalanced,
accuracy alone should not be treated as the primary performance
indicator.'
"""
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix, accuracy_score,
    brier_score_loss,
)


@dataclass
class MetricsReport:
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float
    accuracy: float
    brier_score: float
    confusion_matrix: list
    n_samples: int

    def to_dict(self):
        return asdict(self)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> MetricsReport:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred).tolist()

    return MetricsReport(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        roc_auc=float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        pr_auc=float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        accuracy=float(accuracy_score(y_true, y_pred)),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        confusion_matrix=cm,
        n_samples=int(len(y_true)),
    )


def defer_coverage_report(y_true, y_prob, defer_mask) -> dict:
    """
    Reports accuracy/metrics separately for deferred vs non-deferred cases
    (Section 11: 'Uncertainty/defer coverage and error analysis where
    applicable').
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    defer_mask = np.asarray(defer_mask, dtype=bool)

    coverage = float(1 - defer_mask.mean())
    report = {"coverage": coverage, "n_deferred": int(defer_mask.sum())}

    if (~defer_mask).sum() > 0:
        report["non_deferred_metrics"] = compute_metrics(
            y_true[~defer_mask], y_prob[~defer_mask]
        ).to_dict()
    if defer_mask.sum() > 0:
        report["deferred_metrics"] = compute_metrics(
            y_true[defer_mask], y_prob[defer_mask]
        ).to_dict()
    return report
