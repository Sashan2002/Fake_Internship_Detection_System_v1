"""
Request-time helpers that assemble an incoming advertisement submission into
the same field shape the ML pipeline expects (Appendix A schema fields 1-17,
excluding the target `fraudulent`).

Heavier, model-facing preprocessing (tokenisation, normalisation used for
training/inference) lives in ml/preprocessing/clean_text.py so that the
backend and the research pipeline always use identical logic - the backend
imports from ml/ rather than re-implementing it.
"""
from ml.preprocessing.clean_text import minimal_normalise


def build_advertisement_record(payload: dict) -> dict:
    """
    Normalise a raw API payload into the field set used across the schema
    (Appendix A, fields 1-17). Unset optional fields become None/0 rather
    than being silently dropped, so downstream feature engineering sees a
    consistent shape.
    """
    record = {
        "title": payload.get("title", "").strip(),
        "location": payload.get("location", "").strip() or None,
        "department": payload.get("department", "").strip() or None,
        "salary_range": payload.get("salary_range", "").strip() or None,
        "company_profile": payload.get("company_profile", "").strip() or None,
        "description": payload.get("description", "").strip(),
        "requirements": payload.get("requirements", "").strip() or None,
        "benefits": payload.get("benefits", "").strip() or None,
        "telecommuting": int(bool(payload.get("telecommuting", 0))),
        "has_company_logo": int(bool(payload.get("has_company_logo", 0))),
        "has_questions": int(bool(payload.get("has_questions", 0))),
        "employment_type": payload.get("employment_type", "").strip() or None,
        "required_experience": payload.get("required_experience", "").strip() or None,
        "required_education": payload.get("required_education", "").strip() or None,
        "industry": payload.get("industry", "").strip() or None,
        "function_field": payload.get("function", payload.get("function_field", "")).strip() or None,
        "employer_name": payload.get("employer_name", "").strip() or None,
    }
    return record


def build_combined_text(record: dict) -> str:
    """
    Concatenate the free-text fields into a single document for NLP
    representation, applying only minimal, conservative normalisation
    (Section 8.1: avoid aggressive stemming/stop-word removal for
    transformer inputs).
    """
    parts = [
        record.get("title") or "",
        record.get("company_profile") or "",
        record.get("description") or "",
        record.get("requirements") or "",
        record.get("benefits") or "",
    ]
    combined = " ".join(p for p in parts if p)
    return minimal_normalise(combined)
