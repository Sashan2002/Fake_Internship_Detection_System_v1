# Model Documentation

## 1. Experiment sequence (Section 10)

All experiments are registered in `ml/experiments/experiment_config.yaml`
and should be run in this order, each compared against the previous:

| ID | Name | Purpose | Module |
|---|---|---|---|
| E1 | Majority reference | Naive baseline; anything not beating this is not "predictive" | `sklearn.dummy.DummyClassifier` (see notebook 03) |
| E2 | TF-IDF + Logistic Regression | First real, auditable baseline | `ml/models/baseline_tfidf.py` |
| E3 | TF-IDF + classical ML | Random forest / linear SVM comparison | `ml/models/baseline_ml.py` |
| E4 | Credibility-only | How much signal do structured indicators carry alone? | `ml/credibility/credibility_model.py` |
| E5 | Transformer-only | BERT vs RoBERTa, controlled comparison | `ml/nlp/bert_model.py`, `ml/nlp/roberta_model.py`, `ml/models/transformer_model.py` |
| E6 | Integrated (NLP + credibility) | The system's target model | `ml/models/integrated_model.py` (`use_nlp=True, use_credibility=True`) |
| E7 | Ablation: no credibility | Isolates the NLP contribution | same class, `use_credibility=False` |
| E8 | Ablation: no NLP | Isolates the credibility contribution | same class, `use_nlp=False` |
| E9 | + Uncertainty | Adds the defer mechanism on top of E6 | `ml/uncertainty/uncertainty_estimation.py` |
| E10 | + Explainability | Adds SHAP/LIME on top of E9 | `ml/explainability/` |

**Rule (Section 18):** do not claim model superiority before controlled
evaluation. Each later experiment must be compared to earlier ones on the
same validation split before being described as "better."

## 2. Architecture: integrated model (E6)

Late fusion:

```
text  ──► transformer (BERT/RoBERTa) ──► fraud-probability-style signal ──┐
                                                                            ├─► fusion Logistic Regression ──► p(fraud)
structured fields ──► credibility feature vector (24 engineered cols) ────┘
```

`IntegratedFusionModel` (`ml/models/integrated_model.py`) is deliberately a
thin fusion head that consumes *already-computed* NLP outputs and
credibility vectors, rather than owning the transformer forward pass
itself. This keeps the (expensive) transformer inference decoupled from
the (cheap) fusion experiment, so E6/E7/E8 can all reuse cached NLP outputs
without re-running the transformer three times.

## 3. Transformer configuration

Default checkpoints: `bert-base-uncased` and `roberta-base`
(`ml/nlp/bert_model.py`, `ml/nlp/roberta_model.py`). Configurable via
`BertTrainingConfig`/`RobertaTrainingConfig`: `max_length` (default 256),
`learning_rate` (2e-5), `batch_size` (16), `epochs` (3), `random_seed` (42).

Per Section 8.1: text fed to the tokenizer receives only *minimal*
normalisation (`ml/preprocessing/clean_text.py::minimal_normalise`) — no
aggressive stemming or stop-word removal, since that would strip context a
transformer relies on.

## 4. Uncertainty / defer mechanism (Section 13)

Three-state result interface:
- **Potentially Legitimate** — `p(fraud) < DEFER_THRESHOLD_LOW`
- **Requires Review** — `DEFER_THRESHOLD_LOW ≤ p(fraud) ≤ DEFER_THRESHOLD_HIGH`
- **Potentially Fraudulent** — `p(fraud) > DEFER_THRESHOLD_HIGH`

`confidence` = distance from the 0.5 decision boundary, rescaled to
`[0, 1]`. `uncertainty` = binary predictive entropy.

**Threshold-fitting discipline (Section 13/17):** `fit_thresholds()` must
only ever be called on the **validation** split. Once chosen, thresholds
are frozen into `DEFER_THRESHOLD_LOW`/`DEFER_THRESHOLD_HIGH` in `.env` and
reused unchanged for all test-set evaluation and live inference. They must
never be re-tuned against test-set results.

## 5. Explainability (Section 12)

- **SHAP** (`ml/explainability/shap_explainer.py`) — structured credibility
  features, via a model-agnostic `KernelExplainer`.
- **LIME** (`ml/explainability/lime_explainer.py`) — text-level token
  attributions, via `LimeTextExplainer`.

Both explanation types carry an explicit disclaimer in their `to_dict()`
output: they describe **model behaviour on this input**, not independent
proof of fraud. This disclaimer must be preserved wherever explanations are
surfaced (API responses, UI) — see `backend/services/explanation_service.py`.

## 6. Evaluation protocol (Section 11, 17, 18)

- Primary metrics: precision, recall, F1, balanced accuracy, ROC-AUC,
  PR-AUC, Brier score (calibration). Plain accuracy is reported but is
  **not** the headline metric, given EMSCAD's class imbalance.
- Confusion matrix and defer-coverage breakdown
  (`ml/evaluation/metrics.py::defer_coverage_report`) for the
  "Requires Review" band.
- Fixed random seed (42) throughout; dataset version recorded in
  `ml/experiments/experiment_config.yaml`.
- **The test split is evaluated exactly once**, after the model and
  evaluation protocol are frozen. `ml/evaluation/evaluate.py` attaches a
  `_warning` field to any report run against the `test` split as a
  reminder of this rule — it is not an automatic enforcement, so follow it
  deliberately.

## 7. Currently active model

The backend currently serves **E2 (TF-IDF + Logistic Regression)** as the
active model (`ACTIVE_MODEL_VERSION=baseline-v0` in `.env.example`) because
it is the first model in the sequence that produces a real trained
artefact without requiring a GPU or the transformer stack. To promote the
integrated model (E6) to production:

1. Train and evaluate E2 → E9 in order, on the real dataset, recording
   metrics for each.
2. Save the final integrated model artefact to
   `models/integrated/integrated_model.joblib`.
3. Update `.env`: `INTEGRATED_MODEL_PATH`, `ACTIVE_MODEL_VERSION`, and the
   frozen `DEFER_THRESHOLD_LOW`/`HIGH`.
4. Update `backend/services/prediction_service.py` to load the integrated
   model (currently hard-wired to the baseline) and assemble both the NLP
   probability and credibility vector before calling `predict_proba`.
5. Register the new version via
   `backend/models/database_models.register_model_version()`.
