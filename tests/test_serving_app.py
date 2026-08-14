"""
Tests for the FastAPI serving layer. The triage dependency is overridden
with a fake function so these tests exercise routing, validation, and
response shaping without needing Ollama, the vector index, or the MLflow
registry running.
"""

from fastapi.testclient import TestClient

from src.serving.app import app, get_triage_fn


def fake_triage(text: str) -> dict:
    return {
        "ticket_text": text,
        "predicted_category": "Technical Support",
        "predicted_priority": "Medium",
        "draft_reply": "We're looking into this for you.",
        "cited_ticket_ids": ["123"],
        "link_redacted": False,
        "completed_action_claimed": False,
        "needs_human_escalation": False,
    }


app.dependency_overrides[get_triage_fn] = lambda: fake_triage
client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_triage_endpoint_returns_expected_shape():
    response = client.post("/triage", json={"text": "the app keeps crashing"})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_category"] == "Technical Support"
    assert body["predicted_priority"] == "Medium"
    assert body["cited_ticket_ids"] == ["123"]
    assert body["link_redacted"] is False
    assert body["completed_action_claimed"] is False
    assert body["needs_human_escalation"] is False


def test_triage_endpoint_passes_request_text_to_triage_fn():
    response = client.post("/triage", json={"text": "my order never arrived"})

    assert response.json()["ticket_text"] == "my order never arrived"


def test_triage_endpoint_rejects_empty_text():
    response = client.post("/triage", json={"text": ""})

    assert response.status_code == 422


def test_triage_endpoint_rejects_missing_text_field():
    response = client.post("/triage", json={})

    assert response.status_code == 422
