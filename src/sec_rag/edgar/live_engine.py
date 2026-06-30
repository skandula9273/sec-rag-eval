"""Live on-demand RAG over a freshly fetched EDGAR filing.

Mirrors QueryEngine, but the corpus is ONE filing pulled live from EDGAR rather
than the pre-indexed Neon corpus. Retrieval is an in-memory exact-cosine search
over that filing's chunks (no Neon -> no 512 MB cap), and the embedded filing is
cached by accession number so repeat questions about the same filing are instant.

Reuses the shared pipeline pieces: chunk_document, Embedder, generate_answer*.
Emits the same stream events as QueryEngine.stream(), so the API/frontend treat
the live path identically.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from sec_rag.config import Config, Secrets
from sec_rag.edgar.client import Filing, fetch_filing_text, latest_filing
from sec_rag.generate.answer import generate_answer_stream
from sec_rag.ingest.chunk import chunk_document, tiktoken_encoder
from sec_rag.ingest.embed import Embedder
from sec_rag.retrieve.dense import RetrievedChunk
from sec_rag.api.schemas import Metrics, QueryResponse
from sec_rag.pipeline import _build_citations


def _normalize(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


@dataclass
class IndexedFiling:
    filing: Filing
    contents: list[str]
    vecs: np.ndarray  # (n_chunks, dim), L2-normalized


class LiveEngine:
    """On-demand RAG over EDGAR. One embedder; an accession-keyed filing cache."""

    def __init__(self, cfg: Config, secrets: Secrets | None = None):
        self.cfg = cfg
        self.secrets = secrets or Secrets()
        self.embedder = Embedder(cfg.embedding, self.secrets)
        self.encoder = tiktoken_encoder(cfg.chunking.encoder)
        self._cache: dict[str, IndexedFiling] = {}

    def _index(self, filing: Filing) -> IndexedFiling:
        """Fetch + chunk + embed a filing (cached by accession)."""
        if filing.accession in self._cache:
            return self._cache[filing.accession]
        text = fetch_filing_text(filing)
        chunks = chunk_document(
            text, self.encoder,
            max_tokens=self.cfg.chunking.max_tokens,
            overlap_tokens=self.cfg.chunking.overlap_tokens,
            strategy="token",
        )
        contents = [c.content for c in chunks]
        vecs = _normalize(np.asarray(self.embedder.embed(contents), dtype=np.float32))
        idx = IndexedFiling(filing=filing, contents=contents, vecs=vecs)
        self._cache[filing.accession] = idx
        return idx

    def _retrieve(self, idx: IndexedFiling, question: str, k: int) -> list[RetrievedChunk]:
        qv = _normalize(np.asarray([self.embedder.embed_one(question)], dtype=np.float32))[0]
        sims = idx.vecs @ qv
        top = np.argsort(-sims)[:k]
        f = idx.filing
        doc_name = f"{f.company} {f.form} {f.filing_date}"
        return [
            RetrievedChunk(
                chunk_id=int(i), doc_name=doc_name, ticker=None,
                filing_type=f.form, filing_date=f.filing_date, page=None,
                section=None, content=idx.contents[int(i)], retrieval_score=float(sims[int(i)]),
            )
            for i in top
        ]

    def stream(self, ticker: str, question: str, *, form: str = "10-K",
               top_k: int | None = None) -> Iterator[dict]:
        """Resolve -> fetch -> retrieve -> stream the grounded answer.

        Yields a 'status' event (which filing was pulled), then 'token' deltas,
        then a final 'done' with citations + metrics. Same shape as
        QueryEngine.stream() plus the status line for the live path.
        """
        k = top_k or self.cfg.retrieval.top_k
        trace_id = uuid.uuid4().hex

        t0 = time.perf_counter()
        filing = latest_filing(ticker, form)
        cached = filing.accession in self._cache
        yield {"type": "status",
               "text": f"{filing.company} — {filing.form} filed {filing.filing_date}"
                       + ("" if cached else " (fetching + indexing…)")}
        idx = self._index(filing)
        retrieved = self._retrieve(idx, question, k)
        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        gen = None
        for item in generate_answer_stream(question, retrieved, self.cfg.generation, self.secrets):
            if isinstance(item, str):
                yield {"type": "token", "text": item}
            else:
                gen = item
        generation_ms = int((time.perf_counter() - t1) * 1000)

        metrics = Metrics(
            faithfulness=None,
            latency_ms=retrieval_ms + generation_ms,
            retrieval_ms=retrieval_ms, rerank_ms=None, generation_ms=generation_ms,
            faithfulness_ms=None, cost_usd=gen.cost_usd, cost_is_estimate=gen.cost_is_estimate,
            tokens_in=gen.tokens_in, tokens_out=gen.tokens_out, chunks_retrieved=len(retrieved),
        )
        response = QueryResponse(
            answer=gen.text, citations=_build_citations(retrieved, gen.cited_indices),
            metrics=metrics, trace_id=trace_id, model=gen.model,
        )
        yield {"type": "done", "response": response}
