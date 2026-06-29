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
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from sec_rag.config import load_config
from sec_rag.pipeline import QueryEngine
from sec_rag.api.schemas import QueryRequest, QueryResponse

_state: dict = {"engine": None, "error": None}


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
    allow_headers=["Content-Type", "X-API-Key"],
)


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
def query_stream(req: QueryRequest) -> StreamingResponse:
    """Server-Sent Events stream for low time-to-first-token.

    Emits `data: {"type":"token","text":...}` per answer delta, then
    `data: {"type":"done","response": <QueryResponse>}`, then `data: [DONE]`.
    No faithfulness judge on this path (it can't stream). Same retrieval + answer
    as /query.
    """
    engine: QueryEngine | None = _state["engine"]
    if engine is None:
        raise HTTPException(status_code=503, detail=f"engine not ready: {_state['error']}")
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    def sse():
        try:
            for ev in engine.stream(req.query, top_k=req.top_k):
                if ev["type"] == "token":
                    yield f"data: {json.dumps({'type': 'token', 'text': ev['text']})}\n\n"
                else:
                    payload = {"type": "done", "response": ev["response"].model_dump()}
                    yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:  # surface mid-stream failures to the client
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
