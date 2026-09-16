from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_payload():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert "db" in body
    assert "dataset" in body
    assert set(body["dataset"]["splits"]) == {"train", "val", "test"}
