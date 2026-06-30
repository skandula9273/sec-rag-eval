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

import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from sec_rag.config import Config, Secrets
from sec_rag.edgar.client import Filing, fetch_filing_text, recent_filings
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


_FORM_8K = re.compile(r"\b(8-?k|press release|announce|acquisition|merger|event|material)\b", re.I)
_FORM_10Q = re.compile(r"\b(quarter|quarterly|q[1-4]\b|10-?q|three months|most recent quarter)\b", re.I)


def detect_form(question: str) -> str:
    """Pick a filing type from the question. Default annual (10-K)."""
    if _FORM_8K.search(question):
        return "8-K"
    if _FORM_10Q.search(question):
        return "10-Q"
    return "10-K"


_MULTI = re.compile(
    r"\b(compare|comparison|compared|versus|vs\.?|year[- ]over[- ]year|yoy|trend|"
    r"grow(th|n|ing)?|change (from|since|over)|over the (last|past)|each year|"
    r"both years|prior year|previous year|year[- ]on[- ]year)\b", re.I)
_THREE = re.compile(r"\b(three|3)[- ]?year|last (three|3)|past (three|3)|trend\b", re.I)


def detect_multi(question: str) -> int:
    """How many filings to pull: 1 (single) or 2-3 (comparison/trend)."""
    if not _MULTI.search(question):
        return 1
    return 3 if _THREE.search(question) else 2


@dataclass
class IndexedFiling:
    filing: Filing
    contents: list[str]
    vecs: np.ndarray  # (n_chunks, dim), L2-normalized


class LiveEngine:
    """On-demand RAG over EDGAR. One embedder; an accession-keyed filing cache."""

    # Keep the newest N embedded filings in Neon (bounded so the free tier holds).
    _CACHE_CAP = 40

    def __init__(self, cfg: Config, secrets: Secrets | None = None):
        self.cfg = cfg
        self.secrets = secrets or Secrets()
        self.embedder = Embedder(cfg.embedding, self.secrets)
        self.encoder = tiktoken_encoder(cfg.chunking.encoder)
        self._cache: dict[str, IndexedFiling] = {}  # in-process (warm instance)
        self._conn = None
        self._init_cache_db()

    def _init_cache_db(self):
        """Best-effort: a Neon-backed cache so a cold start reuses embedded filings.

        Persisting embeddings across instances/cold starts makes repeat questions
        about the same filing instant (no re-fetch, no re-embed). All failures are
        swallowed — the cache is an optimization, never required for a query.
        """
        try:
            from sec_rag.db.pool import new_connection
            self._conn = new_connection(self.secrets, autocommit=True)
            dim = self.cfg.embedding.dim
            with self._conn.cursor() as cur:
                cur.execute("CREATE TABLE IF NOT EXISTS live_filings ("
                            "accession TEXT PRIMARY KEY, cached_at TIMESTAMPTZ NOT NULL DEFAULT now())")
                cur.execute(f"CREATE TABLE IF NOT EXISTS live_chunks ("
                            "accession TEXT REFERENCES live_filings(accession) ON DELETE CASCADE, "
                            "chunk_index INT, content TEXT, "
                            f"embedding vector({dim}), PRIMARY KEY (accession, chunk_index))")
        except Exception:
            self._conn = None  # caching disabled; queries still work

    def _load_cached(self, accession: str) -> IndexedFiling | None:
        if self._conn is None:
            return None
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT content, embedding FROM live_chunks "
                            "WHERE accession = %s ORDER BY chunk_index", (accession,))
                rows = cur.fetchall()
            if not rows:
                return None
            contents = [r[0] for r in rows]
            vecs = np.asarray([r[1] for r in rows], dtype=np.float32)  # stored normalized
            return contents, vecs
        except Exception:
            return None

    def _save_cached(self, filing: Filing, contents: list[str], vecs: np.ndarray):
        if self._conn is None:
            return
        try:
            from pgvector import Vector
            with self._conn.cursor() as cur:
                cur.execute("INSERT INTO live_filings (accession) VALUES (%s) "
                            "ON CONFLICT (accession) DO UPDATE SET cached_at = now()",
                            (filing.accession,))
                cur.execute("DELETE FROM live_chunks WHERE accession = %s", (filing.accession,))
                cur.executemany(
                    "INSERT INTO live_chunks (accession, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s)",
                    [(filing.accession, i, contents[i], Vector(vecs[i])) for i in range(len(contents))],
                )
                # Evict beyond the cap (oldest first); cascade drops their chunks.
                cur.execute("DELETE FROM live_filings WHERE accession IN ("
                            "SELECT accession FROM live_filings ORDER BY cached_at DESC OFFSET %s)",
                            (self._CACHE_CAP,))
        except Exception:
            pass  # cache write failed; not fatal

    def _index(self, filing: Filing, embedder: Embedder | None = None) -> IndexedFiling:
        """Fetch + chunk + embed a filing. Cached in-process and in Neon by accession.

        Embeddings are model-specific, not key-specific, so a filing indexed with
        one caller's key is safely reused for any caller on the same model.
        """
        if filing.accession in self._cache:
            return self._cache[filing.accession]
        cached = self._load_cached(filing.accession)
        if cached is not None:
            contents, vecs = cached
            idx = IndexedFiling(filing=filing, contents=contents, vecs=vecs)
            self._cache[filing.accession] = idx
            return idx
        emb = embedder or self.embedder
        text = fetch_filing_text(filing)
        chunks = chunk_document(
            text, self.encoder,
            max_tokens=self.cfg.chunking.max_tokens,
            overlap_tokens=self.cfg.chunking.overlap_tokens,
            strategy="token",
        )
        contents = [c.content for c in chunks]
        vecs = _normalize(np.asarray(emb.embed(contents), dtype=np.float32))
        idx = IndexedFiling(filing=filing, contents=contents, vecs=vecs)
        self._cache[filing.accession] = idx
        self._save_cached(filing, contents, vecs)  # persist for cold starts
        return idx

    def _retrieve(self, idx: IndexedFiling, question: str, k: int,
                  embedder: Embedder | None = None) -> list[RetrievedChunk]:
        emb = embedder or self.embedder
        qv = _normalize(np.asarray([emb.embed_one(question)], dtype=np.float32))[0]
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

    def stream(self, ticker: str, question: str, *, form: str = "auto",
               top_k: int | None = None, secrets: Secrets | None = None) -> Iterator[dict]:
        """Resolve -> fetch -> retrieve -> stream the grounded answer.

        ``form`` is "10-K" | "10-Q" | "8-K" | "auto" (pick from the question).
        ``secrets`` overrides the engine's keys (BYOK): the filing + query embed and
        generation run on the caller's OpenAI/Anthropic keys. Yields a 'status'
        event (which filing was pulled), then 'token' deltas, then a final 'done'
        with citations + metrics.
        """
        from sec_rag.ingest.embed import Embedder

        k = top_k or self.cfg.retrieval.top_k
        trace_id = uuid.uuid4().hex
        if not form or form.lower() == "auto":
            form = detect_form(question)
        # Comparison/trend questions pull the latest N filings; 8-K events don't.
        n = 1 if form == "8-K" else detect_multi(question)
        gen_secrets = secrets or self.secrets
        embedder = Embedder(self.cfg.embedding, secrets) if secrets else self.embedder

        t0 = time.perf_counter()
        filings = recent_filings(ticker, form, n)
        label = f"{filings[0].company} — " + ", ".join(f"{f.form} {f.filing_date}" for f in filings)
        any_cold = any(f.accession not in self._cache for f in filings)
        yield {"type": "status", "text": label + (" (fetching + indexing…)" if any_cold else "")}

        # Retrieve per filing so each period is represented (cap total context).
        per_k = k if n == 1 else max(2, 12 // n)
        retrieved = []
        for f in filings:
            idx = self._index(f, embedder)
            retrieved += self._retrieve(idx, question, per_k, embedder)
        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        gen = None
        for item in generate_answer_stream(question, retrieved, self.cfg.generation, gen_secrets):
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
