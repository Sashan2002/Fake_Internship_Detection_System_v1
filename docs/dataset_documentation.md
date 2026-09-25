# Dataset Documentation

## 1. Source

The research dataset is derived from **EMSCAD** (Employment Scam Aegean
Dataset). EMSCAD is a general job-postings fraud dataset — it is **not
internship-specific**, contains a limited number of internship-labelled
postings, and is strongly imbalanced toward legitimate postings. These are
real limitations of the underlying data, not implementation shortcuts, and
they should be stated wherever results are reported.

## 2. Approved schema (Appendix A)

The research-ready dataset used for training has exactly **42 analytical
columns**, plus 2 research-control columns used only during development
(44 total):

### Original EMSCAD fields (18)
`job_id, title, location, department, salary_range, company_profile,
description, requirements, benefits, telecommuting, has_company_logo,
has_questions, employment_type, required_experience, required_education,
industry, function, fraudulent`

`fraudulent` is the target column. `job_id` is an identifier only.

### Engineered fields (24) — see `ml/preprocessing/feature_engineering.py`
`title_word_count, location_present, department_present, salary_disclosed,
company_profile_present, company_profile_word_count, description_present,
description_word_count, requirements_present, requirements_word_count,
benefits_present, benefits_word_count, employment_type_present,
experience_requirement_present, education_requirement_present,
url_present, email_present, phone_present, contact_information_present,
internship_relevance, salary_lower, salary_upper, salary_range_width,
currency_present`

### Research-control fields (2) — training-time only, never model inputs
`duplicate_group_id, split`

`ml/preprocessing/dataset_split.py::validate_schema()` checks a dataframe
against this exact schema and flags both missing and unexpected columns, so
schema drift is caught immediately rather than silently accepted.

## 3. Leakage safeguards

The following rules are enforced in code, not just documentation:

| Rule | Enforced by |
|---|---|
| `job_id`, `duplicate_group_id`, `split`, and `fraudulent` must never be model inputs | `dataset_split.assert_no_leaked_columns()` — raises `ValueError` if violated |
| A duplicate/near-duplicate group must not cross splits | `dataset_split.duplicate_aware_split()` uses `GroupShuffleSplit`; `check_group_split_integrity()` verifies afterward |
| Class balancing and any preprocessing that "learns" from data (e.g. TF-IDF vocabulary) must be fit only on the training split | Enforced by pipeline discipline in `ml/train_baseline.py` — `fit()` is only ever called on `train_df` |
| The model input column list must be explicit in code | `dataset_split.get_model_input_columns()` — no implicit "everything except the target" logic |

## 4. Duplicate grouping

`compute_duplicate_group_id()` derives a group id by hashing a normalised
concatenation of `title`, `company_profile`, and `description`. This is a
**conservative, exact-match-after-normalisation** grouping — it will not
catch near-duplicates with meaningfully different wording. If a more
sophisticated near-duplicate detector (e.g. MinHash/LSH on shingles) is
introduced later, it should replace this function without changing its
interface, so the rest of the pipeline is unaffected.

## 5. Synthetic development data

`ml/generate_synthetic_data.py` produces a small, clearly-labelled
**synthetic** dataset matching the 44-column schema, purely so the rest of
the pipeline (schema validation, splitting, baseline training, API wiring,
tests) can be exercised before the real dataset is available. Synthetic
output is written with an explicit console warning and must never be used
to report a research result — replace it with the genuine, approved
research-ready dataset at `ml/data/raw/emscad_research_ready.csv` before
running any experiment intended for evaluation.

## 6. Placing the real dataset

1. Obtain the EMSCAD-derived, approved research-ready CSV (42 or 44
   columns as above).
2. Place it at `ml/data/raw/emscad_research_ready.csv`.
3. Run `ml/preprocessing/dataset_split.validate_schema()` (see notebook
   `01_data_exploration.ipynb`) to confirm it matches the approved schema.
4. If `duplicate_group_id`/`split` are not already present, compute them
   with `compute_duplicate_group_id()` and `duplicate_aware_split()`
   (notebook `02_preprocessing_feature_engineering.ipynb`) and save the
   result to `ml/data/processed/`.
5. Record the dataset version (e.g. a content hash or release tag) in
   `ml/experiments/experiment_config.yaml` under `dataset.version` — every
   experiment run should reference this.
