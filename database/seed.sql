-- ============================================================================
-- Seed data for local development / demo only.
-- password_hash values below correspond to the plaintext password "password123"
-- hashed with werkzeug.security.generate_password_hash (pbkdf2:sha256).
-- Replace before any shared or deployed use.
-- ============================================================================

INSERT INTO users (name, email, password_hash, role)
VALUES
    ('Demo Admin', 'admin@example.com',
     'pbkdf2:sha256:600000$placeholderSalt$0000000000000000000000000000000000000000000000000000000000000',
     'admin'),
    ('Demo User', 'user@example.com',
     'pbkdf2:sha256:600000$placeholderSalt$0000000000000000000000000000000000000000000000000000000000000',
     'user');

-- NOTE: The placeholder hashes above are NOT valid working hashes.
-- Run `python backend/utils/seed_users.py` after setup to create real,
-- correctly-hashed demo accounts instead of relying on this file for auth.

INSERT INTO model_versions (model_name, version, training_dataset, metrics_json, file_path)
VALUES
    ('baseline_tfidf_logreg', 'v0', 'unset', '{}', 'models/baseline/tfidf_logreg.joblib');
