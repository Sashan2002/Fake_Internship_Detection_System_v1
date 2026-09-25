# API Documentation

Base URL (local development): `http://localhost:5000/api`

All request/response bodies are JSON. All endpoints return
`{"errors": [...]}` with a 4xx/5xx status on failure.

---

## Auth

### `POST /api/auth/register`
Register a new user. Password is hashed (pbkdf2:sha256) before storage.

**Body**
```json
{ "name": "Ada Lovelace", "email": "ada@example.com", "password": "at-least-8-chars" }
```
**201** → `{ "user_id": 1, "email": "ada@example.com" }`
**400** → validation errors. **409** → email already registered.

### `POST /api/auth/login`
**Body** `{ "email": "...", "password": "..." }`
**200** → `{ "user_id": 1, "name": "...", "email": "...", "role": "user" }`
**401** → invalid credentials.

---

## Advertisements

### `POST /api/advertisements`
Submit an advertisement for storage (does not run prediction — call
`/api/predict` separately).

**Body** (only `title`/`description` required; all others optional)
```json
{
  "user_id": 1, "title": "Marketing Intern", "description": "...",
  "requirements": "...", "benefits": "...", "company_profile": "...",
  "employer_name": "...", "location": "...", "department": "...",
  "salary_range": "...", "employment_type": "...", "required_experience": "...",
  "required_education": "...", "industry": "...", "function": "...",
  "telecommuting": false, "has_company_logo": false, "has_questions": false
}
```
**201** → `{ "advertisement_id": 1 }`

### `GET /api/advertisements/<id>`
**200** → the stored advertisement row. **404** if not found.

### `GET /api/advertisements/user/<user_id>`
**200** → array of the user's advertisements, most recent first.

---

## Prediction & credibility

### `POST /api/analyse-credibility`
Computes the structured credibility indicator vector for a submission
(does not require the advertisement to have been saved first, though
passing `advertisement_id` will persist the analysis if the advertisement
exists).

**200**
```json
{
  "feature_vector": { "company_profile_present": 0, "salary_disclosed": 1, "...": "..." },
  "summary": "6 of 12 completeness/contact indicators present. These are descriptive indicators, not proof of employer legitimacy.",
  "disclaimer": "These are descriptive indicators derived from the submitted information, not independent proof of employer legitimacy or fraud."
}
```

### `POST /api/predict`
Runs the active model and the uncertainty/defer logic on a submission.

**503** if no trained model artefact exists yet (run
`python -m ml.train_baseline` first). This is deliberate: the API refuses
to fabricate a prediction rather than silently serving an untrained model.

**200**
```json
{
  "fraud_probability": 0.83,
  "predicted_class": "Potentially Fraudulent",
  "confidence": 0.66,
  "uncertainty": 0.63,
  "defer_flag": false,
  "model_version": "baseline-v0",
  "credibility": { "feature_vector": { "...": "..." }, "summary": "..." },
  "prediction_id": 12,
  "disclaimer": "'Potentially Legitimate' and 'Potentially Fraudulent' describe model predictions, not proof of an organisation's legitimacy or fraud."
}
```

`predicted_class` is one of `Potentially Legitimate`,
`Potentially Fraudulent`, `Requires Review` (Section 13's three-state
result interface).

### `GET /api/predictions/<id>`
**200** → the stored prediction row. **404** if not found.

### `GET /api/explanations/<id>`
Returns (and caches) a SHAP/LIME-based explanation for the given
prediction.

**200**
```json
{
  "available": true,
  "top_factors": [
    { "token": "urgent", "contribution": 0.041, "direction": "increases_fraud_signal" }
  ],
  "disclaimer": "This explanation describes which factors most influenced the model's own prediction for this specific submission. It is not independent verification of the employer or proof that the advertisement is fraudulent."
}
```
If no model is available yet, `available` is `false` with a `reason`.

### `GET /api/health`
**200** → `{ "status": "ok" }` — liveness check, no auth required.

---

## Error format

All validation and not-found errors use the same shape:
```json
{ "errors": ["title is required", "description is required"] }
```
