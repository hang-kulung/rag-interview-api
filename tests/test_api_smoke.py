from fastapi.testclient import TestClient

from app.main import create_app


def test_health_endpoint() -> None:
    # No `with` block: lifespan (which needs Redis/Qdrant/DB) is intentionally not started.
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_exposes_both_apis() -> None:
    client = TestClient(create_app())
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/documents" in paths
    assert "/api/v1/chat" in paths
