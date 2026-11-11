from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from judgment_engine.app import create_app
from judgment_engine.config import Limits, Settings

AUTH = {"Authorization": "Bearer test-key"}


def make_client(**overrides) -> TestClient:
    settings = Settings(
        backend="lexical",
        allow_insecure=False,
        data_dir=overrides.pop("data_dir", Path("var")),
        limits=Limits(**overrides.pop("limits", {})),
        **overrides,
    )
    app = create_app(settings, api_key="test-key")
    return TestClient(app)


@pytest.fixture()
def client(tmp_path):
    with make_client(data_dir=tmp_path) as test_client:
        test_client.headers.update(AUTH)
        yield test_client


def three_types_payload():
    return {
        "state": {"ticket": "The export button crashes the settings page in Safari."},
        "questions": {
            "needs_human": {"type": "noul", "instructions": "Does the customer want a person?"},
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"returns": "Exchanges and wrong items", "billing": "Charges and invoices"},
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is it?",
                "criteria": ["Cosmetic", "Workaround exists", "Blocking"],
            },
        },
    }


def test_three_question_types_in_one_request(client):
    response = client.post("/v1/decisions", json=three_types_payload())
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body["answers"]) == {"needs_human", "department", "severity"}
    assert body["answers"]["needs_human"]["type"] == "noul"
    assert 0.0 <= body["answers"]["needs_human"]["noul"] <= 1.0

    choice = body["answers"]["department"]
    assert choice["type"] == "choice"
    assert choice["choice"] in {"returns", "billing"}
    assert set(choice["probabilities"]) == {"returns", "billing"}
    assert sum(choice["probabilities"].values()) == pytest.approx(1.0)

    score = body["answers"]["severity"]
    assert score["type"] == "score"
    assert 0.0 <= score["score"] <= 2.0
    assert set(score["probabilities"]) == {"0", "1", "2"}
    assert score["legend"]["2"] == "Blocking"

    for routing in body["routing"].values():
        assert routing["route"] in {"local", "teacher", "human"}
        assert 0.0 <= routing["confidence"] <= 1.0
    assert body["usage"]["pairs"] == 2 + 2 + 3  # noul は yes/no の 2 組


def test_jev_compatible_endpoint_alias(client):
    assert client.post("/v1/systemone", json=three_types_payload()).status_code == 200


def test_same_input_same_output(client):
    first = client.post("/v1/decisions", json=three_types_payload()).json()["answers"]
    second = client.post("/v1/decisions", json=three_types_payload()).json()["answers"]
    assert first == second


def test_choice_bounds_are_enforced(client):
    payload = {
        "state": "x",
        "questions": {
            "pick": {"type": "choice", "instructions": "pick", "criteria": {"only": "one"}},
        },
    }
    response = client.post("/v1/decisions", json=payload)
    assert response.status_code == 422
    assert "2" in response.json()["error"]["message"]


def test_unknown_type_is_rejected(client):
    payload = {"state": "x", "questions": {"q": {"type": "box", "instructions": "where"}}}
    response = client.post("/v1/decisions", json=payload)
    assert response.status_code == 422


def test_score_levels_are_limited(client):
    payload = {
        "state": "x",
        "questions": {"q": {"type": "score", "instructions": "rate", "criteria": [str(i) for i in range(11)]}},
    }
    response = client.post("/v1/decisions", json=payload)
    assert response.status_code == 422


def test_unknown_top_level_field_is_rejected(client):
    payload = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}, "temperature": 0.1}
    assert client.post("/v1/decisions", json=payload).status_code == 422


def test_policy_risk_routes_to_human(client):
    payload = three_types_payload()
    payload["policy"] = {"risk": "high"}
    body = client.post("/v1/decisions", json=payload).json()
    assert {routing["route"] for routing in body["routing"].values()} == {"human"}


def test_policy_can_disable_teacher(client):
    payload = three_types_payload()
    payload["policy"] = {"abstain_below": 1.0, "allow_teacher": False}
    body = client.post("/v1/decisions", json=payload).json()
    assert {routing["route"] for routing in body["routing"].values()} == {"human"}


def test_missing_bearer_is_401(client):
    response = client.post("/v1/decisions", json=three_types_payload(), headers={"Authorization": ""})
    assert response.status_code == 401


def test_wrong_bearer_is_401(client):
    response = client.post(
        "/v1/decisions", json=three_types_payload(), headers={"Authorization": "Bearer nope"}
    )
    assert response.status_code == 401


def test_health_and_metrics_are_open(client):
    assert client.get("/health", headers={"Authorization": ""}).status_code == 200
    metrics = client.get("/metrics", headers={"Authorization": ""})
    assert metrics.status_code == 200
    assert "judgment_decisions_total" in metrics.text


def test_body_limit_returns_413(client):
    big = "a" * (client.app.state.settings.limits.max_body_bytes + 100)
    response = client.post("/v1/decisions", content=json.dumps({"state": big, "questions": {}}))
    assert response.status_code == 413


def test_rate_limit_returns_429(tmp_path):
    with make_client(data_dir=tmp_path, rate_limit_per_min=2) as test_client:
        test_client.headers.update(AUTH)
        payload = three_types_payload()
        assert test_client.post("/v1/decisions", json=payload).status_code == 200
        assert test_client.post("/v1/decisions", json=payload).status_code == 200
        assert test_client.post("/v1/decisions", json=payload).status_code == 429


def test_models_endpoint(client):
    body = client.get("/v1/models").json()
    assert body["models"][0]["types"] == ["choice", "score", "noul"]


def test_feedback_is_stored_masked(client, tmp_path):
    response = client.post(
        "/v1/feedback",
        json={
            "request_id": "req-1",
            "question_id": "department",
            "outcome": "incorrect",
            "label": "billing",
            "note": "contact me at a@example.com or /home/takuya/secret.txt",
        },
    )
    assert response.status_code == 202
    rows = [json.loads(line) for line in (tmp_path / "feedback.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert "a@example.com" not in rows[0]["note"]
    assert "/home/takuya" not in rows[0]["note"]
    assert rows[0]["label"] == "billing"


def test_state_over_limit_is_rejected(client):
    payload = {
        "state": "x" * (client.app.state.settings.limits.max_state_chars + 1),
        "questions": {"q": {"type": "noul", "instructions": "y"}},
    }
    response = client.post("/v1/decisions", json=payload)
    assert response.status_code == 422
    assert "state" in response.json()["error"]["message"]
