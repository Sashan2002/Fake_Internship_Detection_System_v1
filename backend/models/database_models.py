"""
Lightweight data-access layer over SQLite.

A full ORM (e.g. SQLAlchemy) is intentionally avoided to keep the schema in
database/schema.sql as the single source of truth, matching Section 5 and
the leakage/safeguard requirements in Section 18. Each function below maps
directly onto one table defined there.
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from backend.config import config


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCHEMA_PATH = _PROJECT_ROOT / "database" / "schema.sql"
_DEFAULT_SEED_PATH = _PROJECT_ROOT / "database" / "seed.sql"


def init_db(schema_path: Path = None, seed_path: Path = None, run_seed: bool = False):
    """
    Create the database from schema.sql (idempotent - uses IF NOT EXISTS).
    schema.sql/seed.sql always live in database/, independent of where
    DATABASE_PATH points (e.g. a temporary path used in tests), so this
    does not derive their location from DATABASE_PATH's parent.
    """
    schema_path = schema_path or _DEFAULT_SCHEMA_PATH
    config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript(schema_path.read_text())
        if run_seed:
            seed_path = seed_path or _DEFAULT_SEED_PATH
            if seed_path.exists():
                conn.executescript(seed_path.read_text())


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------
def create_user(name: str, email: str, password_hash: str, role: str = "user") -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, role),
        )
        return cur.lastrowid


def get_user_by_email(email: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# advertisements
# ---------------------------------------------------------------------------
def create_advertisement(user_id: int, fields: dict) -> int:
    columns = [
        "title", "description", "requirements", "benefits", "company_profile",
        "employer_name", "location", "department", "salary_range",
        "employment_type", "required_experience", "required_education",
        "industry", "function_field", "telecommuting", "has_company_logo",
        "has_questions",
    ]
    values = [fields.get(c) for c in columns]
    placeholders = ", ".join("?" for _ in columns)
    with get_db() as conn:
        cur = conn.execute(
            f"INSERT INTO advertisements (user_id, {', '.join(columns)}) "
            f"VALUES (?, {placeholders})",
            (user_id, *values),
        )
        return cur.lastrowid


def get_advertisement(advertisement_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM advertisements WHERE advertisement_id = ?",
            (advertisement_id,),
        ).fetchone()
        return dict(row) if row else None


def list_advertisements_for_user(user_id: int):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM advertisements WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# credibility_analysis
# ---------------------------------------------------------------------------
def create_credibility_analysis(advertisement_id: int, feature_dict: dict, summary: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO credibility_analysis (advertisement_id, feature_json, indicator_summary) "
            "VALUES (?, ?, ?)",
            (advertisement_id, json.dumps(feature_dict), summary),
        )
        return cur.lastrowid


def get_credibility_analysis(advertisement_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM credibility_analysis WHERE advertisement_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (advertisement_id,),
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------
def create_prediction(advertisement_id: int, model_version: str, predicted_class: str,
                       fraud_probability: float, confidence: float, uncertainty: float,
                       defer_flag: bool) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO predictions "
            "(advertisement_id, model_version, predicted_class, fraud_probability, "
            "confidence, uncertainty, defer_flag) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (advertisement_id, model_version, predicted_class, fraud_probability,
             confidence, uncertainty, int(defer_flag)),
        )
        return cur.lastrowid


def get_prediction(prediction_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,)
        ).fetchone()
        return dict(row) if row else None


def get_latest_prediction_for_ad(advertisement_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE advertisement_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (advertisement_id,),
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# explanations
# ---------------------------------------------------------------------------
def create_explanation(prediction_id: int, explanation_dict: dict, summary: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO explanations (prediction_id, explanation_json, evidence_summary) "
            "VALUES (?, ?, ?)",
            (prediction_id, json.dumps(explanation_dict), summary),
        )
        return cur.lastrowid


def get_explanation(prediction_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM explanations WHERE prediction_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (prediction_id,),
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# model_versions
# ---------------------------------------------------------------------------
def register_model_version(model_name: str, version: str, training_dataset: str,
                            metrics: dict, file_path: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT OR REPLACE INTO model_versions "
            "(model_name, version, training_dataset, metrics_json, file_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (model_name, version, training_dataset, json.dumps(metrics), file_path),
        )
        return cur.lastrowid
