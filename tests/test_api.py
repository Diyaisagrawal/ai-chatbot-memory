import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import make_session, make_user, say


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "provider": "fake"}


def test_user_signin_is_idempotent(client):
    a = client.post("/api/users", json={"username": "diya"})
    b = client.post("/api/users", json={"username": "diya"})
    assert a.status_code == 201 and b.status_code == 200
    assert a.json()["id"] == b.json()["id"]


def test_requires_known_user(client):
    assert client.get("/api/sessions").status_code == 422  # header missing
    assert client.get("/api/sessions", headers={"X-User-Id": "nope"}).status_code == 401


def test_multi_turn_memory(client):
    h = make_user(client, "diya")
    sid = make_session(client, h)

    say(client, h, sid, "Hi, my name is Diya")
    reply = say(client, h, sid, "What's my name?")

    assert "1 earlier message" in reply
    assert "Your name is Diya" in reply


def test_history_persisted_and_session_auto_titled(client):
    h = make_user(client, "diya")
    sid = make_session(client, h)
    say(client, h, sid, "Explain transformers")

    msgs = client.get(f"/api/sessions/{sid}/messages", headers=h).json()
    assert [m["role"] for m in msgs] == ["human", "ai"]
    assert msgs[0]["content"] == "Explain transformers"

    sessions = client.get("/api/sessions", headers=h).json()
    assert sessions[0]["title"] == "Explain transformers"


def test_sessions_are_isolated(client):
    h = make_user(client, "diya")
    s1, s2 = make_session(client, h), make_session(client, h)
    say(client, h, s1, "my name is Diya")
    reply = say(client, h, s2, "what's my name?")
    assert "0 earlier message" in reply
    assert "Your name is" not in reply


def test_users_cannot_access_each_others_sessions(client):
    alice, bob = make_user(client, "alice"), make_user(client, "bob")
    sid = make_session(client, alice)
    say(client, alice, sid, "secret")

    assert client.get(f"/api/sessions/{sid}/messages", headers=bob).status_code == 404
    r = client.post("/api/chat", json={"session_id": sid, "message": "hi"}, headers=bob)
    assert r.status_code == 404
    assert client.delete(f"/api/sessions/{sid}", headers=bob).status_code == 404
    assert client.get("/api/sessions", headers=bob).json() == []


def test_memory_window_trims_prompt_but_keeps_storage(client):
    # memory_window=6 -> at most 6 history messages (3 turns) reach the model
    h = make_user(client, "diya")
    sid = make_session(client, h)
    for i in range(6):
        say(client, h, sid, f"message {i}")
    reply = say(client, h, sid, "latest")

    assert "3 earlier message(s)" in reply
    assert '"message 3"' in reply  # oldest message still in the window
    assert len(client.get(f"/api/sessions/{sid}/messages", headers=h).json()) == 14


def test_memory_survives_restart(settings):
    with TestClient(create_app(settings)) as c:
        h = make_user(c, "diya")
        sid = make_session(c, h)
        say(c, h, sid, "my name is Diya")

    with TestClient(create_app(settings)) as c:  # fresh process, same DB
        assert "Your name is Diya" in say(c, h, sid, "who am I?")


def test_streaming_endpoint_saves_turn(client):
    h = make_user(client, "diya")
    sid = make_session(client, h)
    with client.stream(
        "POST", "/api/chat/stream", json={"session_id": sid, "message": "stream me"}, headers=h
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]

    tokens = [e["token"] for e in events if "token" in e]
    assert len(tokens) > 1 and events[-1] == {"done": True}
    text = "".join(tokens)
    assert text.startswith('You said: "stream me"')

    msgs = client.get(f"/api/sessions/{sid}/messages", headers=h).json()
    assert msgs[-1] == {**msgs[-1], "role": "ai", "content": text}


def test_clear_rename_delete(client):
    h = make_user(client, "diya")
    sid = make_session(client, h, persona="tutor")
    say(client, h, sid, "hello")

    r = client.patch(f"/api/sessions/{sid}", json={"title": "Renamed", "persona": "coder"}, headers=h)
    assert r.json()["title"] == "Renamed" and r.json()["persona"] == "coder"

    assert client.delete(f"/api/sessions/{sid}/messages", headers=h).status_code == 204
    assert client.get(f"/api/sessions/{sid}/messages", headers=h).json() == []

    assert client.delete(f"/api/sessions/{sid}", headers=h).status_code == 204
    assert client.get("/api/sessions", headers=h).json() == []


def test_invalid_persona_rejected(client):
    h = make_user(client, "diya")
    assert client.post("/api/sessions", json={"persona": "pirate"}, headers=h).status_code == 422


def test_prompt_template_renders_persona_and_user():
    from app.prompts import PERSONAS, chat_prompt

    msgs = chat_prompt.invoke(
        {
            "persona_instructions": PERSONAS["tutor"],
            "user_name": "Diya",
            "today": "2026-10-08",
            "history": [],
            "input": "hi",
        }
    ).to_messages()
    assert msgs[0].type == "system" and "patient tutor" in msgs[0].content and "Diya" in msgs[0].content
    assert msgs[-1].type == "human" and msgs[-1].content == "hi"


def test_hf_endpoint_requires_token(tmp_path):
    import pytest

    s = Settings(database_url=f"sqlite:///{tmp_path / 'x.db'}", llm_provider="hf_endpoint", hf_token=None)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        with TestClient(create_app(s)):
            pass


def test_failed_llm_call_does_not_corrupt_memory(client):
    class Boom:
        async def ainvoke(self, *_a, **_k):
            raise RuntimeError("upstream down")

    h = make_user(client, "diya")
    sid = make_session(client, h)
    say(client, h, sid, "first")

    service = client.app.state.chat_service
    real_chain, service.chain = service.chain, Boom()
    try:
        r = client.post("/api/chat", json={"session_id": sid, "message": "lost"}, headers=h)
        assert r.status_code == 502 and "upstream down" in r.json()["detail"]
    finally:
        service.chain = real_chain

    # Only the successful turn is stored -- no orphaned human message.
    assert len(client.get(f"/api/sessions/{sid}/messages", headers=h).json()) == 2
