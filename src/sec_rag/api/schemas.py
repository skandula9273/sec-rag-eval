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
    # Bounded so bad values fail at the contract boundary (422) instead of deep
    # in the stack: top_k<=0 hit "LIMIT must not be negative" in pgvector and a
    # silent falsy-zero fallback; an unbounded top_k stuffed thousands of chunks
    # into the generation prompt (Anthropic 413 RequestTooLargeError). 1..50
    # comfortably covers recall@5/@10 and any sane ablation. None -> config default.
    top_k: int | None = Field(default=None, ge=1, le=50)
    # The faithfulness judge is a second LLM call (~29% of request latency). Off by
    # default so /query returns the answer fast; set true for the live badge.
    with_faithfulness: bool = False


class LiveQueryRequest(BaseModel):
    """Live EDGAR path: answer ``query`` over ``ticker``'s most recent ``form``."""
    ticker: str
    query: str
    form: str = "auto"  # "10-K" | "10-Q" | "8-K" | "auto" (pick from the question)
    top_k: int | None = Field(default=None, ge=1, le=50)


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
    cost_is_estimate: bool = True  # True unless the model has a confirmed rate in generate/answer.py PRICING
    tokens_in: int
    tokens_out: int
    chunks_retrieved: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    metrics: Metrics
    trace_id: str
    model: str
