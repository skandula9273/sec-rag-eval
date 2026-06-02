"""Retrieval metrics.

FinanceBench gives gold ``evidence`` spans per question, not chunk ids, so the
headline recall@k is an evidence-hit rate: for each question, find the rank of
the first retrieved chunk that contains the gold evidence (substring or fuzzy
token-overlap match), then recall@k = fraction of questions whose first hit is
at rank <= k. MRR uses the same first-hit rank.

Set-based recall and reciprocal rank are also provided for cases where relevant
chunk ids are known directly. All functions are pure and unit-tested.
"""

from __future__ import annotations

import re
from statistics import mean

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def evidence_match_rank(
    retrieved_contents: list[str],
    evidence_texts: list[str],
    *,
    mode: str = "substring",
    threshold: float = 0.5,
) -> int | None:
    """1-based rank of the first retrieved chunk matching any evidence span.

    mode="substring": normalized evidence is a substring of the chunk.
    mode="fuzzy":     >= ``threshold`` of evidence tokens appear in the chunk.
    Returns None on a miss (no evidence, or no chunk matches).
    """
    norm_ev = [_normalize(e) for e in evidence_texts if e and e.strip()]
    if not norm_ev:
        return None

    def hit(content: str) -> bool:
        nc = _normalize(content)
        if mode == "substring":
            return any(e in nc for e in norm_ev)
        if mode == "fuzzy":
            ctoks = set(nc.split())
            for e in norm_ev:
                etoks = set(e.split())
                if etoks and len(etoks & ctoks) / len(etoks) >= threshold:
                    return True
            return False
        raise ValueError(f"unknown match mode: {mode!r}")

    for i, content in enumerate(retrieved_contents, start=1):
        if hit(content):
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
