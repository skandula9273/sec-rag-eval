"""Response schemas. This is the contract from the design doc; every visual
surface (Streamlit now, Next.js maybe in V2) renders this shape.

Fields that belong to later phases (rerank_score, faithfulness, and their
timings) are present but Optional and default to None in V0, so the schema is
stable across versions and the UI does not need to change when they turn on.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None  # falls back to config retrieval.top_k


class Citation(BaseModel):
    source_index: int  # 1-based position in the sources panel
    cited: bool  # True if the answer referenced this source -> colored badge
    doc_name: str
    ticker: str | None = None
    filing_type: str | None = None
    filing_date: str | None = None
    page: int | None = None
    section: str | None = None
    excerpt: str
    retrieval_score: float
    rerank_score: float | None = None  # V1+ (reranker)


class Metrics(BaseModel):
    faithfulness: float | None = None  # V1+ (RAGAS inline)
    latency_ms: int
    retrieval_ms: int
    rerank_ms: int | None = None  # V1+
    generation_ms: int
    faithfulness_ms: int | None = None  # V1+
    cost_usd: float
    cost_is_estimate: bool = True  # pricing not yet confirmed; see generate/answer.py
    tokens_in: int
    tokens_out: int
    chunks_retrieved: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    metrics: Metrics
    trace_id: str
    model: str
