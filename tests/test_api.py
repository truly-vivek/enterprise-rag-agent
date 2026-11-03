from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_validates_question_length():
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 422


def test_query_validates_top_k():
    response = client.post(
        "/query",
        json={"question": "What is the policy?", "top_k": 0},
    )
    assert response.status_code == 422


def test_upload_rejects_unsupported_extension():
    response = client.post(
        "/documents",
        files={"file": ("archive.zip", b"content", "application/zip")},
    )
    assert response.status_code == 400
