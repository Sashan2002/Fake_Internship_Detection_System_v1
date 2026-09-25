-- ============================================================================
-- Fake Internship Detection System - Database Schema
-- Ref: Implementation Specification v1.0, Section 5 (Database Schema)
--
-- Safeguards implemented here (Section 18):
--   * Passwords are stored as password_hash only, never plaintext.
--   * Foreign keys connect advertisements -> credibility_analysis ->
--     predictions -> explanations, as required.
--   * No research-control fields (duplicate_group_id, split) exist in the
--     application database - those belong only to the research dataset
--     used for offline model training (ml/data/).
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------------------------
-- users
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,          -- never store plaintext passwords
    role            TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ----------------------------------------------------------------------------
-- advertisements
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS advertisements (
    advertisement_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              INTEGER NOT NULL,
    title                TEXT NOT NULL,
    description          TEXT NOT NULL,
    requirements         TEXT,
    benefits             TEXT,
    company_profile      TEXT,
    employer_name        TEXT,
    location             TEXT,
    department           TEXT,
    salary_range         TEXT,
    employment_type      TEXT,
    required_experience  TEXT,
    required_education   TEXT,
    industry             TEXT,
    function_field       TEXT,               -- 'function' is reserved in some engines
    telecommuting        INTEGER DEFAULT 0,
    has_company_logo     INTEGER DEFAULT 0,
    has_questions         INTEGER DEFAULT 0,
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- credibility_analysis
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS credibility_analysis (
    credibility_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    advertisement_id     INTEGER NOT NULL,
    feature_json          TEXT NOT NULL,      -- serialized structured credibility feature vector
    indicator_summary     TEXT,               -- human-readable summary of indicators
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (advertisement_id) REFERENCES advertisements(advertisement_id) ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- predictions
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    advertisement_id     INTEGER NOT NULL,
    model_version          TEXT NOT NULL,
    predicted_class         TEXT NOT NULL CHECK (
        predicted_class IN ('Potentially Legitimate', 'Potentially Fraudulent', 'Requires Review')
    ),
    fraud_probability       REAL NOT NULL,
    confidence               REAL NOT NULL,
    uncertainty               REAL,
    defer_flag                 INTEGER NOT NULL DEFAULT 0,
    created_at                  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (advertisement_id) REFERENCES advertisements(advertisement_id) ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- explanations
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS explanations (
    explanation_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id         INTEGER NOT NULL,
    explanation_json        TEXT NOT NULL,     -- SHAP / LIME structured output
    evidence_summary          TEXT,            -- user-readable influential factors
    created_at                  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- model_versions  (model registry, supports reproducibility - Section 17)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_versions (
    model_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name          TEXT NOT NULL,
    version              TEXT NOT NULL,
    training_dataset       TEXT,               -- dataset version identifier
    metrics_json             TEXT,             -- serialized evaluation metrics
    file_path                  TEXT NOT NULL,
    created_at                   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (model_name, version)
);

-- ----------------------------------------------------------------------------
-- Indexes
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_advertisements_user ON advertisements(user_id);
CREATE INDEX IF NOT EXISTS idx_credibility_ad ON credibility_analysis(advertisement_id);
CREATE INDEX IF NOT EXISTS idx_predictions_ad ON predictions(advertisement_id);
CREATE INDEX IF NOT EXISTS idx_explanations_prediction ON explanations(prediction_id);
