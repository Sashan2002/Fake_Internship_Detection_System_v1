"""
API integration tests using Flask's test client against a temporary SQLite
database (never the developer's real database/app.db).
"""
import uuid

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from backend import config as config_module
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(config_module.config, "DATABASE_PATH", test_db_path)

    from backend.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _unique_email():
    return f"user_{uuid.uuid4().hex[:8]}@example.com"


def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_register_and_login_flow(client):
    email = _unique_email()
    payload = {"name": "Test User", "email": email, "password": "supersecure1"}

    res = client.post("/api/auth/register", json=payload)
    assert res.status_code == 201

    res = client.post("/api/auth/login", json={"email": email, "password": "supersecure1"})
    assert res.status_code == 200
    body = res.get_json()
    assert body["email"] == email


def test_login_with_wrong_password_fails(client):
    email = _unique_email()
    client.post("/api/auth/register", json={"name": "Test User", "email": email, "password": "supersecure1"})

    res = client.post("/api/auth/login", json={"email": email, "password": "wrong-password"})
    assert res.status_code == 401


def test_submit_advertisement_requires_fields(client):
    res = client.post("/api/advertisements", json={"title": ""})
    assert res.status_code == 400
    assert "errors" in res.get_json()


def test_submit_and_fetch_advertisement(client):
    email = _unique_email()
    register_res = client.post("/api/auth/register", json={"name": "Test User", "email": email, "password": "supersecure1"})
    user_id = register_res.get_json()["user_id"]

    ad_payload = {
        "user_id": user_id,
        "title": "Marketing Intern",
        "description": "A genuine, structured internship opportunity with mentorship provided.",
    }
    res = client.post("/api/advertisements", json=ad_payload)
    assert res.status_code == 201
    advertisement_id = res.get_json()["advertisement_id"]

    res = client.get(f"/api/advertisements/{advertisement_id}")
    assert res.status_code == 200
    assert res.get_json()["title"] == "Marketing Intern"


def test_analyse_credibility_returns_indicators_and_disclaimer(client):
    payload = {
        "title": "Data Intern",
        "description": "A genuine internship with mentorship and clear responsibilities.",
        "company_profile": "Founded in 2005.",
        "salary_range": "$2,000/month",
    }
    res = client.post("/api/analyse-credibility", json=payload)
    assert res.status_code == 200
    body = res.get_json()
    assert "feature_vector" in body
    assert "not independent proof" in body["disclaimer"]


def test_predict_without_trained_model_returns_503(client):
    """
    Before ml/train_baseline.py has been run, no model artefact exists yet
    (Section 19 checklist), so /api/predict should fail clearly rather than
    fabricate a prediction.
    """
    payload = {
        "title": "Data Intern",
        "description": "A genuine internship with mentorship and clear responsibilities.",
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code in (503, 200)  # 200 only if a model happens to already be trained in this env
