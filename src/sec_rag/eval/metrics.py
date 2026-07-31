"""Retrieval metrics + pluggable evidence matchers.

FinanceBench gives gold ``evidence`` SPANS per question, not chunk ids, so the
headline recall@k is an evidence-hit rate: for each question, the rank of the first
retrieved chunk that "contains" the gold evidence, then recall@k = fraction whose
first hit is at rank <= k. MRR uses the same first-hit rank.

But "contains" is a JUDGEMENT CALL — how leniently a chunk counts as carrying the
evidence — and the headline depends heavily on it. So the matcher is a PLUGGABLE
component, and its parameters live in ``configs/matchers.yaml``, not here:

  * ``strict``   — the normalized gold span is an exact substring of the chunk.
  * ``overlap``  — >= ``threshold`` of the gold span's unique tokens appear in the
                   chunk (the original fuzzy(0.5) metric; kept BYTE-IDENTICAL so
                   committed artifacts stay reproducible — do not change it).
  * ``semantic`` — max cosine(chunk sentence, gold span) >= ``threshold``, embeddings.

``evidence_match_rank`` keeps its original ``mode=``/``threshold=`` signature
(substring -> strict, fuzzy -> overlap) so existing callers and committed numbers
are unchanged; pass ``matcher=`` to score under any pluggable matcher instead.
"""

from __future__ import annotations

import re
from pathlib import Path
from statistics import mean

_WS = re.compile(r"\s+")
# Split a chunk into candidate spans for semantic matching: sentence enders and
# newlines (SEC tables are newline-delimited, not sentences).
_SENT = re.compile(r"(?<=[.!?;])\s+|\n+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


class EvidenceMatcher:
    """Does a retrieved chunk carry a question's gold evidence? name is reported."""

    name: str = "base"

    def matches(self, chunk_content: str, evidence_texts: list[str]) -> bool:
        raise NotImplementedError

    # Optional: batch-embed everything a run will touch so per-chunk matching is
    # cache-only (no network in the hot loop). No-op unless a matcher needs it.
    def warm(self, texts: list[str]) -> None:  # noqa: B027 - intentional no-op default
        return None


class StrictMatcher(EvidenceMatcher):
    """Exact substring: the normalized gold span appears verbatim in the chunk."""

    name = "strict"

    def matches(self, chunk_content: str, evidence_texts: list[str]) -> bool:
        nc = _normalize(chunk_content)
        return any(_normalize(e) in nc for e in evidence_texts if e and e.strip())


class OverlapMatcher(EvidenceMatcher):
    """>= ``threshold`` of the gold span's unique tokens appear in the chunk.

    This is the original ``mode="fuzzy"`` logic, extracted verbatim: same
    normalization, same set intersection over unique tokens, same >= comparison.
    Kept identical so every committed fuzzy(0.5) number reproduces exactly.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "overlap"

    def matches(self, chunk_content: str, evidence_texts: list[str]) -> bool:
        ctoks = set(_normalize(chunk_content).split())
        for e in evidence_texts:
            if not (e and e.strip()):
                continue
            etoks = set(_normalize(e).split())
            if etoks and len(etoks & ctoks) / len(etoks) >= self.threshold:
                return True
        return False


class SemanticMatcher(EvidenceMatcher):
    """Max cosine(chunk sentence, gold span) >= ``threshold``, via an embedder.

    Meaning over tokens: a paraphrase with low token overlap can match, and a chunk
    with high token overlap but a different claim need not. ``embedder`` is any
    object exposing ``embed(list[str]) -> list[list[float]]`` (the project Embedder,
    or a stub in tests). Embeddings are cached and L2-normalized so cosine is a dot
    product. The threshold is stated in config, NOT tuned.
    """

    def __init__(self, embedder, threshold: float = 0.62, max_spans: int = 40):
        self.embedder = embedder
        self.threshold = threshold
        self.max_spans = max_spans
        self.name = "semantic"
        self._cache: dict[str, list[float]] = {}

    @staticmethod
    def _spans(text: str) -> list[str]:
        spans = [s.strip() for s in _SENT.split(text or "") if len(s.strip()) >= 20]
        return spans[:40]

    def _vecs(self, texts: list[str]):
        import numpy as np

        missing = [t for t in texts if t not in self._cache]
        if missing:
            for t, v in zip(missing, self.embedder.embed(missing), strict=False):
                self._cache[t] = v
        arr = np.asarray([self._cache[t] for t in texts], dtype=np.float32)
        n = np.linalg.norm(arr, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return arr / n

    def warm(self, texts: list[str]) -> None:
        """Pre-embed all spans a run will see (dedup) so matches() is cache-only."""
        spans = {s for t in texts for s in self._spans(t)}
        spans |= set(t for t in texts if 0 < len(t) < 400)  # short evidence spans as-is
        todo = [s for s in spans if s not in self._cache]
        for i in range(0, len(todo), 256):
            batch = todo[i : i + 256]
            for t, v in zip(batch, self.embedder.embed(batch), strict=False):
                self._cache[t] = v

    def matches(self, chunk_content: str, evidence_texts: list[str]) -> bool:
        ev = [e for e in evidence_texts if e and e.strip()]
        spans = self._spans(chunk_content)
        if not ev or not spans:
            return False
        S = self._vecs(spans)
        E = self._vecs([_normalize(e)[:400] if len(e) > 400 else e for e in ev])
        return float((S @ E.T).max()) >= self.threshold


def build_matcher(spec: dict, secrets=None) -> EvidenceMatcher:
    """One matcher from a config spec ({'type': ..., params...})."""
    t = spec["type"]
    if t == "strict":
        return StrictMatcher()
    if t == "overlap":
        return OverlapMatcher(threshold=float(spec.get("threshold", 0.5)))
    if t == "semantic":
        from sec_rag.config import EmbeddingConfig, Secrets
        from sec_rag.ingest.embed import Embedder

        emb = Embedder(EmbeddingConfig(**spec["embedding"]), secrets or Secrets())
        return SemanticMatcher(emb, threshold=float(spec["threshold"]))
    raise ValueError(f"unknown matcher type {t!r}")


def load_matchers(
    path: str | Path = "configs/matchers.yaml", secrets=None
) -> dict[str, EvidenceMatcher]:
    """All matchers defined in ``configs/matchers.yaml`` -> {name: matcher}."""
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    return {name: build_matcher(spec, secrets) for name, spec in data["matchers"].items()}


def evidence_match_rank(
    retrieved_contents: list[str],
    evidence_texts: list[str],
    *,
    mode: str = "substring",
    threshold: float = 0.5,
    matcher: EvidenceMatcher | None = None,
) -> int | None:
    """1-based rank of the first retrieved chunk matching any evidence span.

    Backward-compatible: ``mode="substring"`` -> strict, ``mode="fuzzy"`` -> overlap
    (the historical defaults, unchanged). Pass ``matcher=`` to score under a
    pluggable matcher instead. Returns None on a miss (no evidence, or no chunk).
    """
    if not any(e and e.strip() for e in evidence_texts):
        return None
    if matcher is None:
        if mode == "substring":
            matcher = StrictMatcher()
        elif mode == "fuzzy":
            matcher = OverlapMatcher(threshold)
        else:
            raise ValueError(f"unknown match mode: {mode!r}")
    for i, content in enumerate(retrieved_contents, start=1):
        if matcher.matches(content, evidence_texts):
            return i
    return None


def hit_rate_at_k(ranks: list[int | None], k: int) -> float:
    """Mean over queries of (first-hit rank exists and <= k). == recall@k here."""
    if not ranks:
        return 0.0
    return mean(1.0 if (r is not None and r <= k) else 0.0 for r in ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return mean((1.0 / r) if r else 0.0 for r in ranks)


def set_recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Generic recall@k when relevant ids are known: |topk ∩ relevant| / |relevant|."""
    if not relevant_ids:
        return 0.0
    topk = set(retrieved_ids[:k])
    return len(topk & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0
