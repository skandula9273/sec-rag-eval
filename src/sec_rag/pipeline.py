"""Query pipeline: embed -> dense retrieve -> generate -> assemble response.

One code path, used by both the API (api/app.py) and the eval runner
(eval/run_financebench.py), so the numbers in eval come from exactly the same
pipeline a user hits. Timings are measured with perf_counter and reported in the
breakdown the design doc specifies.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from sec_rag.config import Config, Secrets
from sec_rag.db.pool import new_connection
from sec_rag.generate.answer import generate_answer
from sec_rag.ingest.embed import Embedder
from sec_rag.retrieve.dense import RetrievedChunk, dense_search
from sec_rag.api.schemas import Citation, Metrics, QueryResponse


@dataclass
class PipelineResult:
    response: QueryResponse
    retrieved: list[RetrievedChunk]  # raw, for eval recall@k against evidence


class QueryEngine:
    """Holds the embedder + a long-lived DB connection across queries."""

    def __init__(self, cfg: Config, secrets: Secrets | None = None):
        self.cfg = cfg
        self.secrets = secrets or Secrets()
        self.embedder = Embedder(cfg.embedding, self.secrets)
        # autocommit: this connection is long-lived and read-only. Without it,
        # psycopg leaves an open transaction after each query and an idle engine
        # gets killed by Neon's idle-in-transaction timeout (see db/pool.py).
        self.conn = new_connection(self.secrets, autocommit=True)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "QueryEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run(self, query: str, top_k: int | None = None) -> PipelineResult:
        k = top_k or self.cfg.retrieval.top_k
        trace_id = uuid.uuid4().hex

        t0 = time.perf_counter()
        qvec = self.embedder.embed_one(query)
        retrieved = dense_search(self.conn, qvec, k)
        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        gen = generate_answer(query, retrieved, self.cfg.generation, self.secrets)
        generation_ms = int((time.perf_counter() - t1) * 1000)

        citations = [
            Citation(
                source_index=i,
                cited=(i in gen.cited_indices),
                doc_name=c.doc_name,
                ticker=c.ticker,
                filing_type=c.filing_type,
                filing_date=c.filing_date,
                page=c.page,
                section=c.section,
                excerpt=c.content,
                retrieval_score=c.retrieval_score,
            )
            for i, c in enumerate(retrieved, start=1)
        ]

        metrics = Metrics(
            faithfulness=None,  # V0: off (eval.faithfulness=false). V1 turns RAGAS on.
            latency_ms=retrieval_ms + generation_ms,
            retrieval_ms=retrieval_ms,
            rerank_ms=None,
            generation_ms=generation_ms,
            faithfulness_ms=None,
            cost_usd=gen.cost_usd,
            cost_is_estimate=True,
            tokens_in=gen.tokens_in,
            tokens_out=gen.tokens_out,
            chunks_retrieved=len(retrieved),
        )

        response = QueryResponse(
            answer=gen.text,
            citations=citations,
            metrics=metrics,
            trace_id=trace_id,
            model=gen.model,
        )
        return PipelineResult(response=response, retrieved=retrieved)
