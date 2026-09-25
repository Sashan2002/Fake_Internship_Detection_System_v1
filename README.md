# Fake Internship Detection System

An AI-based decision-support system that reviews internship advertisements
using NLP analysis of the advertisement text combined with structured
employer-credibility indicators, producing a prediction with a confidence
score, an uncertainty estimate, and an explanation of the influential
factors — implemented per the accompanying Implementation Specification
v1.0.

**This tool predicts, it does not verify.** Every result is presented as a
model output ("Potentially Legitimate" / "Potentially Fraudulent" /
"Requires Review"), never as confirmed fact — see `docs/model_documentation.md`.

## Quick start

```bash
# 1. Set up environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 2. Generate a small SYNTHETIC dataset so you can run the whole pipeline
#    immediately. Replace with the real research-ready dataset before
#    reporting any result — see docs/dataset_documentation.md.
python -m ml.generate_synthetic_data

# 3. Train the baseline model (E2: TF-IDF + Logistic Regression)
python -m ml.train_baseline

# 4. Run the web app
python backend/app.py
# → http://localhost:5000
```

## Run the tests

```bash
pip install -r requirements.txt
pytest
```

33 tests cover text cleaning, feature engineering, schema/leakage guards,
credibility indicators, uncertainty/defer logic, the baseline model,
evaluation metrics, the full Flask API, and frontend template rendering.

## Project layout

```
backend/        Flask app: routes, services, DB access, request validation
ml/             Research layer: preprocessing, models, uncertainty, XAI, evaluation
frontend/       Server-rendered HTML/CSS/JS pages
database/       SQLite schema + seed data
models/         Saved model artefacts (baseline/, transformer/, integrated/)
notebooks/      Exploration and experiment notebooks (01-06)
tests/          pytest suite
docs/           Architecture, dataset, API, model, and implementation docs
```

See `docs/system_architecture.md` for the full processing flow and
`docs/implementation_notes.md` for exactly what's fully implemented versus
scaffolded pending the real dataset/compute.

## Key documents

- [`docs/system_architecture.md`](docs/system_architecture.md) — processing flow, layering
- [`docs/dataset_documentation.md`](docs/dataset_documentation.md) — schema, leakage safeguards, how to place the real dataset
- [`docs/api_documentation.md`](docs/api_documentation.md) — every endpoint, request/response shapes
- [`docs/model_documentation.md`](docs/model_documentation.md) — experiment sequence E1-E10, architecture, evaluation protocol
- [`docs/implementation_notes.md`](docs/implementation_notes.md) — what's done vs. scaffolded, reproducibility/safeguard checklists

## Important: before reporting any result

1. Replace the synthetic dataset with the real, approved EMSCAD-derived
   research-ready dataset at `ml/data/raw/emscad_research_ready.csv`.
2. Run the full experiment sequence (E1–E10) on it — see
   `docs/model_documentation.md`.
3. Only evaluate the test split once, after freezing the model and
   evaluation protocol (`docs/model_documentation.md` §6).
