"""
Unit tests: text cleaning, feature engineering, schema validation and
leakage safeguards (Section 18/19 checklist items).
"""
import pandas as pd
import pytest

from ml.preprocessing.clean_text import minimal_normalise, detect_url, detect_email, detect_phone
from ml.preprocessing.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLUMNS
from ml.preprocessing.dataset_split import (
    validate_schema, get_model_input_columns, assert_no_leaked_columns,
    NON_PREDICTIVE_COLUMNS, APPROVED_ANALYTICAL_SCHEMA, compute_duplicate_group_id,
    duplicate_aware_split, check_group_split_integrity,
)


def test_minimal_normalise_strips_html_and_whitespace():
    dirty = "<p>Hello   World</p>\n\n<br/>Great <b>job</b>!!"
    cleaned = minimal_normalise(dirty)
    assert "<" not in cleaned
    assert "  " not in cleaned
    assert "Hello World" in cleaned


def test_detect_url_email_phone():
    text = "Contact us at jobs@example.com or visit https://example.com, call 020-7946-0000."
    assert detect_url(text)
    assert detect_email(text)
    assert detect_phone(text)
    assert not detect_phone("no numbers here")


def test_engineer_features_produces_24_columns():
    df = pd.DataFrame([{
        "title": "Marketing Intern",
        "location": "London",
        "department": "",
        "salary_range": "$1,000-$2,000",
        "company_profile": "We are a company.",
        "description": "Great internship opportunity for students.",
        "requirements": "Currently enrolled",
        "benefits": "Mentorship",
        "function": "Marketing",
    }])
    engineered = engineer_features(df)
    for col in ENGINEERED_FEATURE_COLUMNS:
        assert col in engineered.columns
    assert engineered.loc[0, "salary_lower"] == 1000
    assert engineered.loc[0, "salary_upper"] == 2000
    assert engineered.loc[0, "currency_present"] == 1


def test_schema_validation_flags_missing_columns():
    df = pd.DataFrame({c: [] for c in APPROVED_ANALYTICAL_SCHEMA[:-1]})  # drop one column
    result = validate_schema(df)
    assert not result.valid
    assert len(result.missing_columns) == 1


def test_schema_validation_accepts_research_controls():
    df = pd.DataFrame({c: [] for c in APPROVED_ANALYTICAL_SCHEMA + ["duplicate_group_id", "split"]})
    result = validate_schema(df, allow_research_controls=True)
    assert result.valid
    assert result.has_research_controls


def test_model_input_columns_exclude_non_predictive_fields():
    df = pd.DataFrame({c: [] for c in APPROVED_ANALYTICAL_SCHEMA + ["duplicate_group_id", "split"]})
    inputs = get_model_input_columns(df)
    assert not set(inputs) & NON_PREDICTIVE_COLUMNS


def test_assert_no_leaked_columns_raises_on_target_or_controls():
    with pytest.raises(ValueError):
        assert_no_leaked_columns(["title", "fraudulent"])
    with pytest.raises(ValueError):
        assert_no_leaked_columns(["title", "duplicate_group_id"])
    with pytest.raises(ValueError):
        assert_no_leaked_columns(["title", "split"])
    assert_no_leaked_columns(["title", "description_word_count"])  # should not raise


def test_duplicate_group_split_never_crosses_splits():
    rows = []
    for i in range(20):
        rows.append({
            "job_id": i, "title": f"Intern {i % 5}", "company_profile": "Same company",
            "description": f"desc {i % 5}", "fraudulent": i % 2,
        })
    df = pd.DataFrame(rows)
    df["duplicate_group_id"] = compute_duplicate_group_id(df)
    split_df = duplicate_aware_split(df, test_size=0.2, val_size=0.2, random_seed=1)
    offending = check_group_split_integrity(split_df)
    assert offending == []
    assert set(split_df["split"].unique()) <= {"train", "validation", "test"}
