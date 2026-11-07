import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        llm_provider="fake",
        memory_window=6,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def make_user(client: TestClient, name: str) -> dict:
    r = client.post("/api/users", json={"username": name})
    assert r.status_code in (200, 201), r.text
    return {"X-User-Id": r.json()["id"]}


def make_session(client: TestClient, headers: dict, persona: str = "assistant") -> str:
    r = client.post("/api/sessions", json={"persona": persona}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def say(client: TestClient, headers: dict, session_id: str, text: str) -> str:
    r = client.post("/api/chat", json={"session_id": session_id, "message": text}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["reply"]
