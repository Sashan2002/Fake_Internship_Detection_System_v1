"""
Dataset schema validation, leakage safeguards, and duplicate-group-aware
train/validation/test splitting.

Implements:
  * Section 7 - the approved 42-column schema, plus optional
    duplicate_group_id/split research-control columns (44 columns total).
  * Section 18 / Table "Field group | Rule" - job_id, duplicate_group_id and
    split must never be used as model inputs; a duplicate group must not
    cross splits; class balancing and preprocessing must be fit only on
    training data.
  * Section 17 - fixed random seeds, recorded dataset version, no tuning
    against the final test set.
"""
from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Appendix A - Approved Dataset Schema v1.0
ORIGINAL_EMSCAD_FIELDS = [
    "job_id", "title", "location", "department", "salary_range",
    "company_profile", "description", "requirements", "benefits",
    "telecommuting", "has_company_logo", "has_questions", "employment_type",
    "required_experience", "required_education", "industry", "function",
    "fraudulent",
]

ENGINEERED_FIELDS = [
    "title_word_count", "location_present", "department_present", "salary_disclosed",
    "company_profile_present", "company_profile_word_count",
    "description_present", "description_word_count",
    "requirements_present", "requirements_word_count",
    "benefits_present", "benefits_word_count",
    "employment_type_present", "experience_requirement_present", "education_requirement_present",
    "url_present", "email_present", "phone_present", "contact_information_present",
    "internship_relevance", "salary_lower", "salary_upper", "salary_range_width",
    "currency_present",
]

APPROVED_ANALYTICAL_SCHEMA = ORIGINAL_EMSCAD_FIELDS + ENGINEERED_FIELDS  # 42 columns
RESEARCH_CONTROL_FIELDS = ["duplicate_group_id", "split"]               # +2 = 44 columns

TARGET_COLUMN = "fraudulent"
IDENTIFIER_COLUMN = "job_id"

# Columns that must NEVER be passed to a model as a feature (Section 18 / 21)
NON_PREDICTIVE_COLUMNS = {IDENTIFIER_COLUMN, TARGET_COLUMN, *RESEARCH_CONTROL_FIELDS}

assert len(APPROVED_ANALYTICAL_SCHEMA) == 42
assert len(APPROVED_ANALYTICAL_SCHEMA) + len(RESEARCH_CONTROL_FIELDS) == 44


@dataclass
class SchemaValidationResult:
    valid: bool
    missing_columns: list = field(default_factory=list)
    unexpected_columns: list = field(default_factory=list)
    has_research_controls: bool = False
    notes: list = field(default_factory=list)


def validate_schema(df: pd.DataFrame, allow_research_controls: bool = True) -> SchemaValidationResult:
    """
    Verify the dataframe matches the approved 42-column schema, optionally
    plus the two research-control columns (44 total). Extra/unknown columns
    are flagged rather than silently ignored, so schema drift is caught
    early (Section 7, dev-sequence step 4).
    """
    columns = set(df.columns)
    expected = set(APPROVED_ANALYTICAL_SCHEMA)
    missing = sorted(expected - columns)

    has_controls = set(RESEARCH_CONTROL_FIELDS).issubset(columns)
    allowed = expected | (set(RESEARCH_CONTROL_FIELDS) if allow_research_controls else set())
    unexpected = sorted(columns - allowed)

    notes = []
    if has_controls and not allow_research_controls:
        notes.append("research-control columns present but not permitted in this context")

    valid = not missing and not unexpected
    return SchemaValidationResult(
        valid=valid,
        missing_columns=missing,
        unexpected_columns=unexpected,
        has_research_controls=has_controls,
        notes=notes,
    )


def get_model_input_columns(df: pd.DataFrame) -> list:
    """
    The single, explicit definition of which columns may reach the model
    (Section 21: 'The final model input list must be explicitly defined in
    code so that research-control metadata cannot accidentally enter
    training.'). Excludes identifier, target and research-control columns.
    """
    return [c for c in df.columns if c not in NON_PREDICTIVE_COLUMNS]


def assert_no_leaked_columns(feature_columns: list):
    leaked = set(feature_columns) & NON_PREDICTIVE_COLUMNS
    if leaked:
        raise ValueError(
            f"Leakage guard triggered: {sorted(leaked)} must never be used as "
            f"model features (job_id/duplicate_group_id/split/fraudulent)."
        )


def compute_duplicate_group_id(df: pd.DataFrame, text_columns=("title", "company_profile", "description")) -> pd.Series:
    """
    Derive a duplicate_group_id when one is not already present, by hashing
    a normalised concatenation of key text fields. Near-duplicate postings
    (e.g. the same scam re-posted) should hash identically after
    normalisation; this is a conservative exact-match grouping - it is
    documented as such rather than claimed to catch all near-duplicates.
    """
    def _key(row):
        parts = [str(row.get(c, "") or "").strip().lower() for c in text_columns]
        joined = "||".join(parts)
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()

    return df.apply(_key, axis=1)


def check_group_split_integrity(df: pd.DataFrame, group_col: str = "duplicate_group_id",
                                 split_col: str = "split") -> list:
    """
    Verify no duplicate group spans more than one split (Section 18:
    'Do not allow duplicate/similar groups to cross data splits.').
    Returns a list of offending group ids (empty list == passes).
    """
    if group_col not in df.columns or split_col not in df.columns:
        raise KeyError(f"Both '{group_col}' and '{split_col}' must be present to check integrity")
    grouped = df.groupby(group_col)[split_col].nunique()
    return grouped[grouped > 1].index.tolist()


def duplicate_aware_split(df: pd.DataFrame, group_col: str = "duplicate_group_id",
                           test_size: float = 0.15, val_size: float = 0.15,
                           random_seed: int = 42) -> pd.DataFrame:
    """
    Produce a `split` column with values train/validation/test, ensuring
    every row sharing a duplicate_group_id stays in the same split. Uses a
    fixed random seed (Section 17) and must be run once and recorded, not
    re-run per experiment.
    """
    if group_col not in df.columns:
        raise KeyError(f"'{group_col}' column is required for duplicate-aware splitting")

    out = df.copy()
    groups = out[group_col].values

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    trainval_idx, test_idx = next(gss1.split(out, groups=groups))

    trainval = out.iloc[trainval_idx]
    relative_val_size = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_seed)
    train_idx_rel, val_idx_rel = next(
        gss2.split(trainval, groups=trainval[group_col].values)
    )

    split = np.empty(len(out), dtype=object)
    split[test_idx] = "test"
    train_abs_idx = trainval.index[train_idx_rel]
    val_abs_idx = trainval.index[val_idx_rel]
    split[out.index.get_indexer(train_abs_idx)] = "train"
    split[out.index.get_indexer(val_abs_idx)] = "validation"

    out["split"] = split

    offending = check_group_split_integrity(out, group_col=group_col, split_col="split")
    if offending:
        raise RuntimeError(f"Duplicate-group leakage detected across splits: {offending[:5]}...")

    return out
