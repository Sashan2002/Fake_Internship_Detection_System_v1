"""
Unit tests for employer credibility feature construction (Section 9) and
the uncertainty / defer mechanism (Section 13).
"""
from ml.credibility.credibility_features import build_credibility_vector, summarise_indicators
from ml.uncertainty.uncertainty_estimation import (
    classify_with_defer, DeferThresholds, fit_thresholds,
    POTENTIALLY_LEGITIMATE, POTENTIALLY_FRAUDULENT, REQUIRES_REVIEW,
)


def test_build_credibility_vector_flags_missing_indicators():
    record = {
        "title": "Unpaid Internship",
        "description": "Work from home, no experience needed.",
        "company_profile": "",
        "requirements": "",
        "benefits": "",
        "salary_range": "",
        "location": "",
        "has_company_logo": 0,
        "has_questions": 0,
    }
    vector = build_credibility_vector(record)
    assert vector["company_profile_present"] == 0
    assert vector["salary_disclosed"] == 0
    assert vector["has_company_logo"] == 0


def test_build_credibility_vector_detects_contact_info():
    record = {
        "title": "Marketing Intern",
        "description": "Apply now! Email us at jobs@example.com",
        "company_profile": "An established firm.",
        "requirements": "Degree in progress",
        "benefits": "Mentorship",
        "salary_range": "$1,500/month",
        "location": "Remote",
        "has_company_logo": 1,
        "has_questions": 1,
    }
    vector = build_credibility_vector(record)
    assert vector["email_present"] == 1
    assert vector["contact_information_present"] == 1


def test_summarise_indicators_is_non_judgemental_text():
    vector = {"company_profile_present": 1, "salary_disclosed": 0}
    summary = summarise_indicators(vector)
    assert "not proof" in summary


def test_classify_with_defer_boundaries():
    thresholds = DeferThresholds(low=0.4, high=0.6)

    legit = classify_with_defer(0.1, thresholds)
    assert legit["predicted_class"] == POTENTIALLY_LEGITIMATE
    assert legit["defer_flag"] is False

    fraud = classify_with_defer(0.9, thresholds)
    assert fraud["predicted_class"] == POTENTIALLY_FRAUDULENT
    assert fraud["defer_flag"] is False

    review = classify_with_defer(0.5, thresholds)
    assert review["predicted_class"] == REQUIRES_REVIEW
    assert review["defer_flag"] is True


def test_fit_thresholds_on_validation_only():
    y_true = [0] * 50 + [1] * 50
    p_fraud = [0.05] * 40 + [0.5] * 20 + [0.95] * 40
    thresholds = fit_thresholds(y_true, p_fraud, target_defer_rate=0.2)
    assert thresholds.low < 0.5 < thresholds.high
