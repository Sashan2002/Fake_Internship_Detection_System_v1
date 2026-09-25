"""
Application configuration.

Values are read from environment variables (see .env.example). Nothing
sensitive is hard-coded here; copy .env.example to .env and fill it in.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    # Database
    DATABASE_PATH = BASE_DIR / "database" / "app.db"
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")

    # Model artefact paths
    BASELINE_MODEL_PATH = BASE_DIR / os.getenv(
        "BASELINE_MODEL_PATH", "models/baseline/tfidf_logreg.joblib"
    )
    TRANSFORMER_MODEL_PATH = BASE_DIR / os.getenv(
        "TRANSFORMER_MODEL_PATH", "models/transformer/"
    )
    INTEGRATED_MODEL_PATH = BASE_DIR / os.getenv(
        "INTEGRATED_MODEL_PATH", "models/integrated/integrated_model.joblib"
    )
    ACTIVE_MODEL_VERSION = os.getenv("ACTIVE_MODEL_VERSION", "baseline-v0")

    # Uncertainty / defer thresholds (Section 13). These must be selected on
    # validation data and frozen before final test evaluation - see
    # ml/uncertainty/uncertainty_estimation.py and Section 17.
    DEFER_THRESHOLD_LOW = float(os.getenv("DEFER_THRESHOLD_LOW", "0.40"))
    DEFER_THRESHOLD_HIGH = float(os.getenv("DEFER_THRESHOLD_HIGH", "0.60"))

    RAW_DATASET_PATH = BASE_DIR / os.getenv(
        "RAW_DATASET_PATH", "ml/data/raw/emscad_research_ready.csv"
    )

    DEBUG = os.getenv("FLASK_ENV", "development") == "development"


config = Config()
