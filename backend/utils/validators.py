"""
Request-level validation for API inputs.

This is deliberately separate from ml/preprocessing/, which validates the
research dataset schema (Section 7). This module validates data coming in
through the web API at request time.
"""
import re

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

REQUIRED_ADVERTISEMENT_FIELDS = ["title", "description"]


def validate_registration(payload: dict) -> list:
    errors = []
    if not payload.get("name", "").strip():
        errors.append("name is required")
    email = payload.get("email", "").strip()
    if not email or not EMAIL_RE.match(email):
        errors.append("a valid email is required")
    password = payload.get("password", "")
    if len(password) < 8:
        errors.append("password must be at least 8 characters")
    return errors


def validate_login(payload: dict) -> list:
    errors = []
    if not payload.get("email", "").strip():
        errors.append("email is required")
    if not payload.get("password", ""):
        errors.append("password is required")
    return errors


def validate_advertisement(payload: dict) -> list:
    errors = []
    for field in REQUIRED_ADVERTISEMENT_FIELDS:
        if not str(payload.get(field, "")).strip():
            errors.append(f"{field} is required")

    description = str(payload.get("description", ""))
    if description and len(description) < 20:
        errors.append("description is too short to analyse meaningfully (min 20 characters)")

    for bool_field in ("telecommuting", "has_company_logo", "has_questions"):
        if bool_field in payload and payload[bool_field] not in (0, 1, True, False, None):
            errors.append(f"{bool_field} must be boolean")

    return errors


def sanitize_for_logging(payload: dict) -> dict:
    """
    Strip fields that should never appear in logs (Section 5 safeguard:
    'Logs should avoid sensitive information and unnecessary personal data').
    """
    redacted = dict(payload)
    for sensitive in ("password", "password_hash", "email"):
        if sensitive in redacted:
            redacted[sensitive] = "***redacted***"
    return redacted
