# Implementation Notes

## 1. What's fully implemented vs. what's scaffolded

**Fully implemented and tested** (run `pytest` — 33 tests):
- Text cleaning, 24-feature engineering, schema validation, leakage
  guards, duplicate-aware splitting
- Employer credibility feature vector + rule-based summary
- Uncertainty/defer classification (three-state result)
- TF-IDF + Logistic Regression baseline (trainable, saveable, loadable)
- Classical ML baseline (Random Forest / Linear SVM)
- Evaluation metrics
- Full Flask API (auth, advertisements, predict, analyse-credibility,
  explanations, health) backed by SQLite
- Full server-rendered frontend (login/register, dashboard, analyse form,
  result screen, history, about)
- LIME text explanations (end-to-end tested against the trained baseline)

**Scaffolded, needs real data/compute to complete**:
- BERT/RoBERTa fine-tuning (`ml/nlp/bert_model.py`,
  `ml/nlp/roberta_model.py`) — model/tokenizer wrappers are complete and
  functional (`predict_proba`, `save`, `load`), but the actual fine-tuning
  loop is deliberately left to a dedicated training script once
  checkpointing/early-stopping decisions are made (see notebook 04 and
  `docs/model_documentation.md` §3).
- Integrated fusion model (`ml/models/integrated_model.py`) — fully
  implemented and unit-testable in isolation, but not yet wired as the
  backend's active model because it depends on a trained transformer (see
  `docs/model_documentation.md` §7 for the exact promotion steps).
- SHAP credibility explanations
  (`ml/explainability/shap_explainer.py::CredibilityShapExplainer`) —
  implemented, but needs a trained `CredibilityOnlyModel` plus background
  data to instantiate; the backend currently uses a simpler rule-based
  fallback (`explanation_service.explain_credibility`) until then.

## 2. Why the backend serves the baseline, not the integrated model, today

Serving an untested or untrained model as if it were the evaluated
research model would misrepresent the system's actual performance. Rather
than wire up `IntegratedFusionModel` before it has real trained weights
and a validated defer threshold, `prediction_service.py` serves E2, the
first model in the sequence that can be honestly trained end-to-end
without a GPU. `ModelNotTrainedError` (503) is raised — rather than a
fabricated prediction — if even that baseline hasn't been trained yet.

## 3. Reproducibility checklist (Section 17), as implemented

- [x] Fixed random seed (42) as the default across
      `TfidfBaselineConfig`, `ClassicalMlConfig`, `CredibilityModelConfig`,
      `BertTrainingConfig`, `IntegratedModelConfig`,
      `duplicate_aware_split()`.
- [x] Dataset version field in `ml/experiments/experiment_config.yaml`
      (`dataset.version`) — must be filled in once the real dataset is
      placed.
- [x] Model registry table (`model_versions`) records dataset + metrics +
      file path per trained artefact.
- [ ] Experiment tracking beyond structured JSON files
      (`ml/experiments/runs/<run_name>/*.json`) — `mlflow` is listed in
      `requirements.txt` for teams that want richer tracking, but is not
      wired in by default to keep the reference implementation dependency-light.

## 4. Safeguards checklist (Section 18), as implemented

- [x] `job_id`/`duplicate_group_id`/`split`/`fraudulent` blocked from
      model inputs via `assert_no_leaked_columns()`.
- [x] Duplicate groups cannot cross splits — enforced (raises
      `RuntimeError`) in `duplicate_aware_split()`.
- [x] Model fitting only ever called on the `train` split in
      `ml/train_baseline.py`.
- [x] Credibility features and explanations are framed as indicators of
      model behaviour, never as proof — see the `disclaimer` field
      returned by every relevant API endpoint and enforced string checks
      in `tests/test_credibility.py`.
- [x] No plaintext password storage — `werkzeug.security.generate_password_hash`.
- [x] Sensitive fields excluded from logs — `validators.sanitize_for_logging()`.

## 5. Development sequence followed (matches Section 19)

1. Project skeleton and folder structure — done.
2. Database schema — done (`database/schema.sql`).
3. Dataset placement + schema validation tooling — tooling done; **real
   dataset must be placed by the user** (see `docs/dataset_documentation.md` §6).
4. Preprocessing + feature engineering — done.
5. Duplicate-aware split + leakage guards — done.
6. Baseline models (E1-E4) — implemented; train on real data via
   `python -m ml.train_baseline` (E2) or notebook 03 (E1, E3, E4).
7. Transformer experiments (E5) — wrappers done; fine-tuning loop pending
   real data/compute (notebook 04).
8. Integrated model + ablations (E6-E8) — implemented; pending E5's output.
9. Uncertainty/defer (E9) — done.
10. Explainability (E10) — done (LIME fully working; SHAP pending a
    trained credibility-only model).
11. Backend API — done.
12. Frontend — done.
13. Tests — done (33 passing).
14. Documentation — this folder.

## 6. Known gaps / honest limitations

- The credibility-only SHAP explainer needs `CredibilityOnlyModel` to be
  trained on real data before it can replace the current rule-based
  fallback in the API.
- No automated model-comparison report generator exists yet — running
  each experiment and comparing `ml/evaluation/metrics.py` output across
  `ml/experiments/runs/` is currently a manual step (notebook 03/06).
- Authentication is minimal by design (see `docs/system_architecture.md` §6).
- `ml/preprocessing/dataset_split.py::compute_duplicate_group_id()` uses
  exact-match hashing after normalisation, not fuzzy near-duplicate
  detection — documented in `docs/dataset_documentation.md` §4.
