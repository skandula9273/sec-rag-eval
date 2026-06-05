"""Auth guard on /query (SEC_RAG_API_KEY)."""

import importlib
import os

from fastapi.testclient import TestClient


def _client(monkeypatch_env: dict):
    # Reload app module so the guard reads the env we set for each case.
    for k, v in monkeypatch_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    import sec_rag.api.app as app_mod
    importlib.reload(app_mod)
    return TestClient(app_mod.app)


def test_health_open_regardless_of_key():
    c = _client({"SEC_RAG_API_KEY": "secret123"})
    assert c.get("/health").status_code == 200


def test_query_requires_key_when_set():
    c = _client({"SEC_RAG_API_KEY": "secret123"})
    # no header -> 401 (guard fires before engine/validation)
    assert c.post("/query", json={"query": "x"}).status_code == 401
    # wrong header -> 401
    assert c.post("/query", json={"query": "x"}, headers={"X-API-Key": "nope"}).status_code == 401


def test_query_guard_disabled_when_unset():
    c = _client({"SEC_RAG_API_KEY": None})
    # guard off: passes auth, then hits engine-not-ready (503) — NOT 401
    r = c.post("/query", json={"query": "x"}, headers={})
    assert r.status_code != 401
