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
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException

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
