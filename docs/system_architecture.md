# System Architecture

## 1. Overview

The Fake Internship Detection System is a decision-support tool that combines
NLP analysis of advertisement text with structured employer credibility
indicators to help a human reviewer assess an internship advertisement. It
does **not** issue a verdict of fact — every prediction is presented as a
model output, alongside a confidence score, an uncertainty estimate, and an
explanation of the factors that most influenced it.

## 2. Processing flow

```
Internship Advertisement + Employer Information
            │
            ▼
        Validation                (backend/utils/validators.py)
            │
            ▼
   ┌────────┴─────────┐
   ▼                   ▼
NLP Analysis     Employer Credibility Analysis
(ml/nlp/,        (ml/credibility/,
 ml/models/       backend/services/credibility_service.py)
 baseline_tfidf/
 transformer_model.py)
   │                   │
   └─────────┬─────────┘
             ▼
   Feature/Representation Integration
   (ml/models/integrated_model.py)
             │
             ▼
       Fraud Classification
             │
             ▼
  Confidence / Uncertainty Assessment
  (ml/uncertainty/uncertainty_estimation.py)
             │
             ▼
   Explainability / Evidence
   (ml/explainability/shap_explainer.py,
    ml/explainability/lime_explainer.py)
             │
             ▼
   Evidence-Based User Result
   (frontend/templates/result.html)
```

## 3. Layers

The codebase is split into three layers that can be developed, tested, and
reasoned about independently:

- **`ml/`** — the research layer. Framework-agnostic: no Flask, no
  database imports. Contains preprocessing, feature engineering, models,
  uncertainty, explainability, and evaluation. Anything here should be
  runnable from a notebook or a script with no web server involved.
- **`backend/`** — the application layer. A Flask app that validates
  requests, calls into `ml/` for inference, persists results via
  `backend/models/database_models.py`, and exposes a small JSON API.
- **`frontend/`** — server-rendered HTML/CSS/JS pages that call the JSON
  API and render results. No build step required.

This separation exists so that experiments in `ml/` can be reproduced
independently of the web UI, and so the web app never has to re-implement
preprocessing or modeling logic (train/serve skew is avoided by having
`backend/utils/preprocessing.py` import directly from
`ml/preprocessing/clean_text.py`).

## 4. Request lifecycle (submit → result)

1. User submits an advertisement via `POST /api/advertisements`
   (`backend/routes/advertisement_routes.py`) → persisted in `advertisements`.
2. Frontend calls `POST /api/predict` with the same payload plus
   `advertisement_id`.
   - `backend/services/prediction_service.py` loads the active model
     artefact (currently the TF-IDF + Logistic Regression baseline; swap
     to the integrated model once trained and evaluated) and produces a
     fraud probability.
   - `backend/services/uncertainty_service.py` converts that probability
     into `predicted_class` (`Potentially Legitimate` /
     `Potentially Fraudulent` / `Requires Review`), `confidence`, and
     `uncertainty`, using thresholds frozen from validation-set tuning.
   - `backend/services/credibility_service.py` builds the structured
     credibility indicator vector for the same submission.
   - Results are persisted to `predictions` and `credibility_analysis`.
3. Frontend navigates to `/result/<prediction_id>`, which calls
   `GET /api/predictions/<id>`, `POST /api/analyse-credibility`, and
   `GET /api/explanations/<id>` to render the full evidence-based result.

## 5. Model versioning

`model_versions` (see `database/schema.sql`) records which artefact, trained
on which dataset version, with which metrics, is active. `ACTIVE_MODEL_VERSION`
in `.env` controls which artefact `prediction_service.py` reports as having
produced a given prediction — keep this in sync when promoting a newly
trained model.

## 6. Known simplifications (documented, not hidden)

- The active model served by the backend today is the TF-IDF + Logistic
  Regression baseline (E2). The integrated NLP + credibility model (E6) is
  implemented in `ml/models/integrated_model.py` but requires a trained
  transformer checkpoint before it can be wired in as the active model —
  see `docs/model_documentation.md`.
- Authentication is intentionally minimal (a `localStorage` session token
  keyed by `user_id`) — this is a decision-support demo, not a
  production-grade auth system.
