"""
Confidence, uncertainty and defer-mechanism logic (Section 13, Experiment
E9). The result interface must support three outcome states: Potentially
Legitimate, Potentially Fraudulent, and Requires Review.

CRITICAL SAFEGUARD (Section 13 / 17): 'The defer threshold should be
selected using validation data and fixed before final test evaluation. It
must not be tuned on the test set.' -> fit_thresholds() below is only ever
meant to be called on the validation split; the resulting thresholds are
then frozen and reused as constants (see backend/config.py
DEFER_THRESHOLD_LOW/HIGH) for all subsequent test-set and live inference.
"""
from dataclasses import dataclass

import numpy as np

POTENTIALLY_LEGITIMATE = "Potentially Legitimate"
POTENTIALLY_FRAUDULENT = "Potentially Fraudulent"
REQUIRES_REVIEW = "Requires Review"


@dataclass
class DeferThresholds:
    low: float = 0.40   # below this -> confidently legitimate
    high: float = 0.60  # above this -> confidently fraudulent
    # between low and high -> Requires Review


def confidence_from_probability(p_fraud: float) -> float:
    """
    Confidence is defined as distance from the decision boundary (0.5),
    rescaled to [0, 1]. This is a simple, interpretable confidence measure;
    entropy-based or MC-dropout-based estimates could be substituted here
    later as long as they are validated in Experiment E9.
    """
    return abs(p_fraud - 0.5) * 2


def uncertainty_from_probability(p_fraud: float) -> float:
    """Binary predictive entropy, normalised to [0, 1]."""
    eps = 1e-9
    p = min(max(p_fraud, eps), 1 - eps)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(entropy)  # already in [0, 1] for binary entropy


def classify_with_defer(p_fraud: float, thresholds: DeferThresholds = None) -> dict:
    thresholds = thresholds or DeferThresholds()
    confidence = confidence_from_probability(p_fraud)
    uncertainty = uncertainty_from_probability(p_fraud)

    if thresholds.low <= p_fraud <= thresholds.high:
        predicted_class = REQUIRES_REVIEW
        defer_flag = True
    elif p_fraud > thresholds.high:
        predicted_class = POTENTIALLY_FRAUDULENT
        defer_flag = False
    else:
        predicted_class = POTENTIALLY_LEGITIMATE
        defer_flag = False

    return {
        "fraud_probability": float(p_fraud),
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "uncertainty": float(uncertainty),
        "defer_flag": defer_flag,
    }


def fit_thresholds(y_true_val, p_fraud_val, target_defer_rate: float = 0.15) -> DeferThresholds:
    """
    Select symmetric thresholds around 0.5 on VALIDATION data only, such
    that approximately `target_defer_rate` of validation cases fall into
    the Requires Review band. This is one reasonable, documented selection
    rule; whatever rule is used, per Section 13/17 it must be fixed here
    and not re-tuned against the test set.
    """
    p_fraud_val = np.asarray(p_fraud_val)
    distances = np.abs(p_fraud_val - 0.5)
    if len(distances) == 0:
        return DeferThresholds()
    cutoff_distance = np.quantile(distances, target_defer_rate)
    return DeferThresholds(low=0.5 - cutoff_distance, high=0.5 + cutoff_distance)
