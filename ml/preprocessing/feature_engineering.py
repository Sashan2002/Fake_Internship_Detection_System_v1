"""
Engineered-feature construction for the approved 42-column schema
(Appendix A). Fields 1-18 are the original EMSCAD fields; this module
derives fields 19-42 from them.

These functions are used both to build the offline research dataset
(ml/preprocessing/dataset_split.py -> training) and, via
ml/credibility/credibility_features.py, to score a single live submission
from the web app - so the same logic must be reusable at both scales. Every
function here therefore accepts either a pandas Series/DataFrame column or
a single scalar value where relevant.
"""
import re
import pandas as pd

from ml.preprocessing.clean_text import detect_url, detect_email, detect_phone

CURRENCY_RE = re.compile(r"(\$|USD|GBP|EUR|£|€|AUD|CAD)", re.IGNORECASE)
SALARY_NUM_RE = re.compile(r"(\d[\d,]*)(?:\.\d+)?")


def _word_count(text) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def _present(value) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and pd.isna(value):
        return 0
    return int(bool(str(value).strip()))


def parse_salary_range(salary_range):
    """
    Best-effort parse of a free-text salary_range field into
    (lower, upper, width). Returns (None, None, None) when unparsable -
    absence of a clean numeric salary is itself informative and is captured
    separately by `salary_disclosed`.
    """
    if not isinstance(salary_range, str) or not salary_range.strip():
        return None, None, None
    numbers = SALARY_NUM_RE.findall(salary_range.replace(",", ""))
    numbers = [float(n) for n in numbers if n]
    if not numbers:
        return None, None, None
    if len(numbers) == 1:
        return numbers[0], numbers[0], 0.0
    lower, upper = min(numbers), max(numbers)
    return lower, upper, upper - lower


def internship_relevance(title: str, function_field: str = None) -> int:
    """
    Simple keyword indicator for whether the posting is internship-related,
    used only as a descriptive/filter feature - not as a legitimacy signal.
    """
    text = f"{title or ''} {function_field or ''}".lower()
    keywords = ("intern", "internship", "trainee", "placement", "co-op", "coop")
    return int(any(k in text for k in keywords))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame containing at least the original 18 EMSCAD fields
    (Appendix A, 1-18), return a copy with the 24 engineered columns
    (Appendix A, 19-42) added. Column order matches the appendix.
    """
    out = df.copy()

    out["title_word_count"] = out.get("title", pd.Series(dtype=str)).apply(_word_count)
    out["location_present"] = out.get("location", pd.Series(dtype=str)).apply(_present)
    out["department_present"] = out.get("department", pd.Series(dtype=str)).apply(_present)
    out["salary_disclosed"] = out.get("salary_range", pd.Series(dtype=str)).apply(_present)

    out["company_profile_present"] = out.get("company_profile", pd.Series(dtype=str)).apply(_present)
    out["company_profile_word_count"] = out.get("company_profile", pd.Series(dtype=str)).apply(_word_count)

    out["description_present"] = out.get("description", pd.Series(dtype=str)).apply(_present)
    out["description_word_count"] = out.get("description", pd.Series(dtype=str)).apply(_word_count)

    out["requirements_present"] = out.get("requirements", pd.Series(dtype=str)).apply(_present)
    out["requirements_word_count"] = out.get("requirements", pd.Series(dtype=str)).apply(_word_count)

    out["benefits_present"] = out.get("benefits", pd.Series(dtype=str)).apply(_present)
    out["benefits_word_count"] = out.get("benefits", pd.Series(dtype=str)).apply(_word_count)

    out["employment_type_present"] = out.get("employment_type", pd.Series(dtype=str)).apply(_present)
    out["experience_requirement_present"] = out.get("required_experience", pd.Series(dtype=str)).apply(_present)
    out["education_requirement_present"] = out.get("required_education", pd.Series(dtype=str)).apply(_present)

    combined_text = (
        out.get("description", "").fillna("") + " " +
        out.get("company_profile", "").fillna("") + " " +
        out.get("requirements", "").fillna("") + " " +
        out.get("benefits", "").fillna("")
    )
    out["url_present"] = combined_text.apply(detect_url).astype(int)
    out["email_present"] = combined_text.apply(detect_email).astype(int)
    out["phone_present"] = combined_text.apply(detect_phone).astype(int)
    out["contact_information_present"] = (
        out["url_present"] | out["email_present"] | out["phone_present"]
    ).astype(int)

    out["internship_relevance"] = out.apply(
        lambda r: internship_relevance(r.get("title"), r.get("function")), axis=1
    )

    salary_parsed = out.get("salary_range", pd.Series(dtype=str)).apply(parse_salary_range)
    out["salary_lower"] = salary_parsed.apply(lambda t: t[0])
    out["salary_upper"] = salary_parsed.apply(lambda t: t[1])
    out["salary_range_width"] = salary_parsed.apply(lambda t: t[2])

    out["currency_present"] = out.get("salary_range", pd.Series(dtype=str)).apply(
        lambda s: int(bool(CURRENCY_RE.search(s))) if isinstance(s, str) else 0
    )

    return out


ENGINEERED_FEATURE_COLUMNS = [
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

assert len(ENGINEERED_FEATURE_COLUMNS) == 24, "Appendix A specifies exactly 24 engineered features"
