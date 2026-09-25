"""
Trains and evaluates the TF-IDF + Logistic Regression baseline (E2) end to
end: load -> schema validate -> leakage guard -> split-aware subset ->
train on `train` -> evaluate on `validation` -> save model + metrics +
register in the model registry.

Usage:
    python -m ml.train_baseline --dataset ml/data/raw/emscad_research_ready.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ml.preprocessing.dataset_split import (
    validate_schema, assert_no_leaked_columns, TARGET_COLUMN,
)
from ml.preprocessing.clean_text import minimal_normalise
from ml.models.baseline_tfidf import TfidfLogRegBaseline, TfidfBaselineConfig
from ml.evaluation.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[1]


def build_combined_text_column(df: pd.DataFrame) -> pd.Series:
    parts = df[["title", "company_profile", "description", "requirements", "benefits"]].fillna("")
    combined = parts.agg(" ".join, axis=1)
    return combined.apply(minimal_normalise)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "ml/data/raw/emscad_research_ready.csv"))
    parser.add_argument("--output_model", default=str(ROOT / "models/baseline/tfidf_logreg.joblib"))
    parser.add_argument("--run_name", default="E2_tfidf_logreg")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run "
            f"`python -m ml.generate_synthetic_data` for a dev-only stand-in, "
            f"or place the real research-ready dataset there (Section 19)."
        )

    df = pd.read_csv(dataset_path)

    validation = validate_schema(df, allow_research_controls=True)
    if not validation.valid:
        raise ValueError(f"Dataset schema invalid: {validation}")
    if "split" not in df.columns:
        raise ValueError("Dataset is missing the 'split' research-control column required for training.")

    df["combined_text"] = build_combined_text_column(df)
    feature_columns = ["combined_text"]
    assert_no_leaked_columns(feature_columns)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]

    print(f"Train rows: {len(train_df)} | Validation rows: {len(val_df)}")

    model = TfidfLogRegBaseline(TfidfBaselineConfig())
    model.fit(train_df["combined_text"], train_df[TARGET_COLUMN])

    val_probs = model.predict_proba(val_df["combined_text"])
    report = compute_metrics(val_df[TARGET_COLUMN], val_probs)
    print(json.dumps(report.to_dict(), indent=2))

    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Saved baseline model to {out_path}")

    # Register in model registry (Table "model_versions")
    try:
        from backend.models.database_models import init_db, register_model_version
        init_db()
        register_model_version(
            model_name="baseline_tfidf_logreg",
            version="v1",
            training_dataset=str(dataset_path),
            metrics=report.to_dict(),
            file_path=str(out_path),
        )
        print("Registered model version in database.")
    except Exception as exc:  # pragma: no cover - registry is best-effort here
        print(f"Note: could not register model in database ({exc}). "
              f"Run backend/app.py once first to initialise the DB, or ignore for now.")


if __name__ == "__main__":
    main()
