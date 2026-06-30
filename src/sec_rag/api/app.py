"""FastAPI service.

  GET  /health  -> liveness, never touches external services.
  POST /query   -> {query, top_k?} -> QueryResponse (see schemas.py).

The QueryEngine (embedder + DB connection) is built once at startup via a
lifespan handler. If required secrets are missing, the engine is left unset and
/query returns 503 with a clear message, so a misconfigured deploy fails
readably instead of at the first embedding call. Run locally:
``uvicorn sec_rag.api.app:app --reload``.
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from sec_rag.config import Secrets, load_config
from sec_rag.pipeline import QueryEngine
from sec_rag.api.schemas import LiveQueryRequest, QueryRequest, QueryResponse

_state: dict = {"engine": None, "live": None, "error": None}


def _prewarm(live, tickers: list[str]) -> None:
    """Index a few filings in the background so common demo queries are instant."""
    for t in tickers:
        try:
            from sec_rag.edgar.client import latest_filing
            live._index(latest_filing(t.strip(), "10-K"))
        except Exception:
            pass  # best-effort warming; never crash startup


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Optional shared-key guard for /query.

    If SEC_RAG_API_KEY is set in the environment, every /query request must send a
    matching ``X-API-Key`` header, so a public deploy is not an open faucet on the
    OpenAI/Anthropic keys. If the env var is unset (local dev, tests, CI) the guard
    is disabled and behaviour is unchanged. /health is never guarded — Cloud Run
    needs it for liveness checks. Compared with hmac.compare_digest (constant time).
    """
    expected = os.environ.get("SEC_RAG_API_KEY")
    if not expected:
        return  # guard disabled
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("SEC_RAG_CONFIG", "configs/v0.yaml")
    try:
        cfg = load_config(config_path)
        _state["engine"] = QueryEngine(cfg)
        # Live EDGAR path shares the same config (embed model, chunking, top_k).
        from sec_rag.edgar.live_engine import LiveEngine
        _state["live"] = LiveEngine(cfg)
        # Optional: warm popular tickers in the background so a demo's likely first
        # queries are instant. Off unless SEC_RAG_PREWARM is set (it costs a few
        # cents per cold start); the persistent cache keeps them warm afterwards.
        if os.environ.get("SEC_RAG_PREWARM"):
            tickers = os.environ.get("SEC_RAG_PREWARM_TICKERS", "AAPL,MSFT,NVDA,GOOGL,AMZN,TSLA").split(",")
            threading.Thread(target=_prewarm, args=(_state["live"], tickers), daemon=True).start()
    except Exception as exc:  # surfaced via /query as 503
        _state["error"] = str(exc)
    try:
        yield
    finally:
        if _state["engine"] is not None:
            _state["engine"].close()


app = FastAPI(title="sec-filings-rag", version="0.0.0", lifespan=lifespan)

# The static GitHub Pages frontend (web/) calls /query and /query/stream from the
# browser, so the API must send CORS headers or the browser blocks the response.
# Origins are unauthenticated-safe to open: /query is still gated by the X-API-Key
# header (require_api_key), which the browser sends and CORS must allow. No cookies
# are used, so allow_credentials stays False and "*" origins are fine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-OpenAI-Key", "X-Anthropic-Key"],
)


# Lightweight per-IP rate limit so a public link can't be hammered into a big bill.
# In-memory + per-instance (resets on cold start) — a guard, not airtight; the cache
# also makes repeat queries free. Tune via SEC_RAG_RATELIMIT="N/seconds".
_RL_N, _RL_WINDOW = (int(x) for x in os.environ.get("SEC_RAG_RATELIMIT", "40/600").split("/"))
_RL_HITS: dict[str, deque] = defaultdict(deque)


def _rate_limited(request: Request) -> bool:
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "?")
    now = time.time()
    q = _RL_HITS[ip]
    while q and q[0] < now - _RL_WINDOW:
        q.popleft()
    if len(q) >= _RL_N:
        return True
    q.append(now)
    return False


def _byok_secrets(openai_key: str | None, anthropic_key: str | None) -> Secrets | None:
    """Per-request keys (BYOK): if the caller supplies BOTH their OpenAI and
    Anthropic keys, queries run on THEIR accounts. Missing -> None -> the engine's
    server keys are used (local dev / owner). DATABASE_URL still comes from env.

    If SEC_RAG_REQUIRE_KEYS is set (cost-safe public mode), a request WITHOUT both
    keys is rejected (401) instead of falling back to the owner's keys."""
    if openai_key and anthropic_key:
        return Secrets(openai_api_key=openai_key, anthropic_api_key=anthropic_key)
    if os.environ.get("SEC_RAG_REQUIRE_KEYS"):
        raise HTTPException(
            status_code=401,
            detail="This demo requires your own API keys — add your OpenAI + Anthropic keys (⚙).",
        )
    return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine_ready": _state["engine"] is not None}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest) -> QueryResponse:
    engine: QueryEngine | None = _state["engine"]
    if engine is None:
        raise HTTPException(status_code=503, detail=f"engine not ready: {_state['error']}")
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    return engine.run(
        req.query, top_k=req.top_k, with_faithfulness=req.with_faithfulness
    ).response


@app.post("/query/stream", dependencies=[Depends(require_api_key)])
def query_stream(
    req: QueryRequest,
    request: Request,
    x_openai_key: str | None = Header(default=None),
    x_anthropic_key: str | None = Header(default=None),
) -> StreamingResponse:
    """Server-Sent Events stream for low time-to-first-token.

    Emits `data: {"type":"token","text":...}` per answer delta, then
    `data: {"type":"done","response": <QueryResponse>}`, then `data: [DONE]`.
    No faithfulness judge on this path (it can't stream). Same retrieval + answer
    as /query. BYOK: X-OpenAI-Key + X-Anthropic-Key run it on the caller's keys.
    """
    engine: QueryEngine | None = _state["engine"]
    if engine is None:
        raise HTTPException(status_code=503, detail=f"engine not ready: {_state['error']}")
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if _rate_limited(request):
        raise HTTPException(status_code=429, detail="Rate limit reached — please wait a minute.")
    secrets = _byok_secrets(x_openai_key, x_anthropic_key)

    def sse():
        try:
            for ev in engine.stream(req.query, top_k=req.top_k, secrets=secrets):
                if ev["type"] == "token":
                    yield f"data: {json.dumps({'type': 'token', 'text': ev['text']})}\n\n"
                else:
                    payload = {"type": "done", "response": ev["response"].model_dump()}
                    yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:  # surface mid-stream failures to the client
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/query/live/stream", dependencies=[Depends(require_api_key)])
def query_live_stream(
    req: LiveQueryRequest,
    request: Request,
    x_openai_key: str | None = Header(default=None),
    x_anthropic_key: str | None = Header(default=None),
) -> StreamingResponse:
    """Live EDGAR path: fetch ``ticker``'s most recent ``form`` and answer over it.

    SSE: a `status` event (which filing was pulled), then `token` deltas, then
    `done` with citations + metrics, then `[DONE]`. Any company in EDGAR; the
    filing is fetched + indexed on demand (cached by accession). BYOK:
    X-OpenAI-Key + X-Anthropic-Key run it on the caller's keys.
    """
    live = _state["live"]
    if live is None:
        raise HTTPException(status_code=503, detail=f"engine not ready: {_state['error']}")
    if not req.query.strip() or not req.ticker.strip():
        raise HTTPException(status_code=422, detail="ticker and query are required")
    if _rate_limited(request):
        raise HTTPException(status_code=429, detail="Rate limit reached — please wait a minute.")
    secrets = _byok_secrets(x_openai_key, x_anthropic_key)

    def sse():
        try:
            for ev in live.stream(req.ticker, req.query, form=req.form,
                                  top_k=req.top_k, secrets=secrets):
                if ev["type"] == "done":
                    payload = {"type": "done", "response": ev["response"].model_dump()}
                    yield f"data: {json.dumps(payload)}\n\n"
                else:  # status | token
                    yield f"data: {json.dumps(ev)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
