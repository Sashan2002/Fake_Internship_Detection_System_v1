"""
Frontend smoke tests: confirm each page route renders (Section 14, Table
"Page | Main content") and that the templates reference the expected
result-screen fields (Section 14: prediction, fraud probability,
confidence, uncertainty/defer status, credibility indicators, influential
XAI factors, limitation notice, model version and analysis timestamp).
"""
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


@pytest.mark.parametrize("path", ["/", "/dashboard", "/analyse", "/history", "/about"])
def test_pages_render(client, path):
    res = client.get(path)
    assert res.status_code == 200


def test_result_page_renders_with_prediction_id(client):
    res = client.get("/result/1")
    assert res.status_code == 200


def test_result_template_contains_required_result_fields():
    html = open("frontend/templates/result.html").read()
    for required in [
        "verdict-label", "value-probability", "value-confidence",
        "value-uncertainty", "credibility-grid", "evidence-list",
        "prediction-meta",
    ]:
        assert required in html, f"result.html is missing required element id: {required}"


def test_analyse_template_covers_advertisement_and_employer_fields():
    html = open("frontend/templates/analyse.html").read()
    for required in ["description", "requirements", "benefits", "employer_name", "company_profile"]:
        assert required in html
