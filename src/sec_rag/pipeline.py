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

import psycopg

from sec_rag.config import Config, Secrets
from sec_rag.db.pool import new_connection
from sec_rag.generate.answer import generate_answer
from sec_rag.generate.faithfulness import score_faithfulness
from sec_rag.ingest.embed import Embedder
from sec_rag.retrieve.dense import RetrievedChunk, dense_search
from sec_rag.retrieve.hybrid import hybrid_search
from sec_rag.retrieve.lexical import lexical_search
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

    def _search(self, query: str, qvec: list[float], k: int) -> list[RetrievedChunk]:
        """Run the configured retriever: dense (V0) or hybrid (V1)."""
        r = self.cfg.retrieval
        if r.method == "hybrid":
            return hybrid_search(
                self.conn, qvec, query, k,
                candidates=r.candidates, k_rrf=r.k_rrf, dense_weight=r.dense_weight,
            )
        if r.method == "lexical":
            return lexical_search(self.conn, query, k)
        if r.method == "dense":
            return dense_search(self.conn, qvec, k)
        raise ValueError(f"unknown retrieval.method: {r.method!r}")

    def _retrieve(self, query: str, qvec: list[float], k: int) -> list[RetrievedChunk]:
        """Retrieve with one reconnect on a dead connection.

        The engine holds one long-lived connection. autocommit (see __init__)
        stops idle-in-transaction kills, but a hard network drop or a
        server-side recycle can still leave a dead socket — every later query
        would then 500. Detect the broken connection, rebuild it once, and
        retry. Read-only, so the retry is safe.
        """
        try:
            return self._search(query, qvec, k)
        except psycopg.OperationalError:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = new_connection(self.secrets, autocommit=True)
            return self._search(query, qvec, k)

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[RetrievedChunk], int]:
        """Embed the query and retrieve top-k chunks — no generation.

        The retrieval half of run(), exposed so the eval harness can measure
        recall@k / MRR (pure retrieval metrics) without the Anthropic generation
        + faithfulness calls. The path is identical to the one run() and the API
        use, so the recall it measures is the recall a user's query gets — it
        just stops before generation.
        """
        k = top_k or self.cfg.retrieval.top_k
        t0 = time.perf_counter()
        qvec = self.embedder.embed_one(query)
        retrieved = self._retrieve(query, qvec, k)
        retrieval_ms = int((time.perf_counter() - t0) * 1000)
        return retrieved, retrieval_ms

    def run(self, query: str, top_k: int | None = None) -> PipelineResult:
        trace_id = uuid.uuid4().hex

        retrieved, retrieval_ms = self.retrieve(query, top_k)

        t1 = time.perf_counter()
        gen = generate_answer(query, retrieved, self.cfg.generation, self.secrets)
        generation_ms = int((time.perf_counter() - t1) * 1000)

        # Faithfulness badge: one judge call grading how grounded the answer is in
        # the retrieved sources. Off by default (eval.faithfulness=false) so it
        # stays off the critical path unless enabled; see generate/faithfulness.py.
        faithfulness = None
        faithfulness_ms = None
        if self.cfg.eval.faithfulness:
            t2 = time.perf_counter()
            fr = score_faithfulness(gen.text, retrieved, self.cfg.generation, self.secrets)
            faithfulness = fr.score
            faithfulness_ms = int((time.perf_counter() - t2) * 1000)

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
            faithfulness=faithfulness,  # None unless eval.faithfulness enabled
            latency_ms=retrieval_ms + generation_ms + (faithfulness_ms or 0),
            retrieval_ms=retrieval_ms,
            rerank_ms=None,
            generation_ms=generation_ms,
            faithfulness_ms=faithfulness_ms,
            cost_usd=gen.cost_usd,
            cost_is_estimate=gen.cost_is_estimate,
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
