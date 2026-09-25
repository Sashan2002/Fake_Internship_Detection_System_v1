"""
Employer/advertisement credibility feature vector construction (Section 9).

IMPORTANT INTERPRETATION RULE (verbatim intent of Section 9): these
features are indicators derived from advertisement/employer information.
They are NOT independent proof that an employer is legitimate or
fraudulent. Nothing in this module should be read, displayed, or logged as
a legitimacy verdict - it produces measurable indicators only
(Appendix B: 'Use structured credibility features as measurable indicators
rather than a hard-coded legitimacy score unless a separate validated
scoring methodology is defined.').
"""
import pandas as pd

from ml.preprocessing.feature_engineering import ENGINEERED_FEATURE_COLUMNS, engineer_features

# The subset of the 24 engineered features (Table 5 / Appendix A 19-42) that
# specifically constitute the "structured credibility vector" referenced in
# Section 8.2, as opposed to purely descriptive engineered features like
# word counts used elsewhere.
CREDIBILITY_INDICATOR_COLUMNS = [
    "company_profile_present",
    "company_profile_word_count",
    "has_company_logo",
    "has_questions",
    "salary_disclosed",
    "benefits_present",
    "requirements_present",
    "location_present",
    "employment_type_present",
    "experience_requirement_present",
    "education_requirement_present",
    "url_present",
    "email_present",
    "phone_present",
    "contact_information_present",
    "salary_lower",
    "salary_upper",
    "salary_range_width",
    "currency_present",
]


def build_credibility_vector(record: dict) -> dict:
    """
    Build the structured credibility feature vector for a single
    advertisement record (as assembled by
    backend/utils/preprocessing.build_advertisement_record).
    """
    df = pd.DataFrame([record])
    engineered = engineer_features(df)
    row = engineered.iloc[0]

    vector = {}
    for col in CREDIBILITY_INDICATOR_COLUMNS:
        if col in row.index:
            val = row[col]
            vector[col] = None if pd.isna(val) else _to_native(val)
    # has_company_logo / has_questions come straight from the raw record
    # if not already present via engineering.
    vector.setdefault("has_company_logo", int(bool(record.get("has_company_logo", 0))))
    vector.setdefault("has_questions", int(bool(record.get("has_questions", 0))))
    return vector


def _to_native(value):
    """
    Convert numpy scalar types (int64/float64/bool_) to plain Python types
    so the vector is safely JSON-serializable by Flask's jsonify, which
    does not know how to encode numpy scalars.
    """
    if hasattr(value, "item"):
        return value.item()
    return value


def summarise_indicators(vector: dict) -> str:
    """
    Produce a short, human-readable, non-judgemental summary string for
    storage in credibility_analysis.indicator_summary (Table 2). Phrased
    descriptively ("X of Y indicators present"), not as a verdict.
    """
    presence_keys = [k for k in vector if k.endswith("_present")]
    present_count = sum(1 for k in presence_keys if vector.get(k))
    return (
        f"{present_count} of {len(presence_keys)} completeness/contact "
        f"indicators present. These are descriptive indicators, not proof "
        f"of employer legitimacy."
    )
