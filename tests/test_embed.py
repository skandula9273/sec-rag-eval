"""Embedder retry/backoff on transient OpenAI errors (mocked client, no sleeps)."""

import httpx
import pytest
from openai import RateLimitError

from sec_rag.ingest import embed as embmod
from sec_rag.ingest.embed import Embedder


class _Emb:
    def __init__(self, index, vec):
        self.index = index
        self.embedding = vec


class _Resp:
    def __init__(self, data):
        self.data = data


def _rate_limit_error():
    req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    return RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)


def _quota_error():
    req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    return RateLimitError(
        "Error code: 429 - insufficient_quota: exceeded your current quota",
        response=httpx.Response(429, request=req), body=None,
    )


def _embedder():
    e = Embedder.__new__(Embedder)  # bypass __init__ (no real OpenAI client / key)
    e.model, e.dim, e.batch_size = "m", 3, 2
    return e


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(embmod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Embeddings:
        def create(self, model, input):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _rate_limit_error()
            return _Resp([_Emb(i, [0.0, 0.1, 0.2]) for i, _ in enumerate(input)])

    e = _embedder()
    e.client = type("C", (), {"embeddings": Embeddings()})()
    out = e.embed(["a", "b"])
    assert calls["n"] == 2  # one retry
    assert len(out) == 2 and len(out[0]) == 3


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(embmod.time, "sleep", lambda s: None)

    class Embeddings:
        def create(self, model, input):
            raise _rate_limit_error()

    e = _embedder()
    e.client = type("C", (), {"embeddings": Embeddings()})()
    with pytest.raises(RateLimitError):
        e.embed(["a"])


def test_fails_fast_on_insufficient_quota(monkeypatch):
    monkeypatch.setattr(embmod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Embeddings:
        def create(self, model, input):
            calls["n"] += 1
            raise _quota_error()

    e = _embedder()
    e.client = type("C", (), {"embeddings": Embeddings()})()
    with pytest.raises(RateLimitError):
        e.embed(["a"])
    assert calls["n"] == 1  # billing error: no retries
