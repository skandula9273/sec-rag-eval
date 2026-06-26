"""Hybrid retrieval: dense + lexical, fused by Reciprocal Rank Fusion (V1).

V0 was dense-only. The V0 eval diagnosed that dense retrieval fails on
table/number questions (recall@5 0.32) far worse than prose (0.66), because
cosine similarity is semantic, not lexical. Hybrid adds a keyword signal
(retrieve/lexical.py) so exact financial terms are caught, then fuses the two
ranked lists.

Why Reciprocal Rank Fusion (RRF): dense scores are cosine in [0,1] and lexical
scores are ts_rank_cd on an unbounded different scale — they cannot be added
directly. RRF fuses by RANK position, not score, so no normalization is needed:

    rrf(chunk) = Σ_lists 1 / (k_rrf + rank_in_list)

A chunk ranked high in either list scores well; a chunk ranked high in BOTH
scores best. k_rrf (default 60, the standard value) damps the contribution of
low ranks. This is the standard, parameter-light hybrid-fusion method.

Everything here is an ablation knob in configs/v1.yaml: candidate depth per
retriever, k_rrf, and (future) per-list weights.
"""

from __future__ import annotations

import psycopg

from sec_rag.retrieve.dense import RetrievedChunk, dense_search
from sec_rag.retrieve.lexical import lexical_search


def _rrf_fuse(
    dense: list[RetrievedChunk],
    lexical: list[RetrievedChunk],
    k_rrf: int,
    top_k: int,
    dense_weight: float = 0.5,
) -> list[RetrievedChunk]:
    """Fuse two ranked lists by (weighted) Reciprocal Rank Fusion, top_k by score.

    Chunks are identified by chunk_id (the same chunk can appear in both lists).
    The returned RetrievedChunk keeps its original retrieval_score (the dense
    cosine if present, else the lexical score) for display, but ORDER is RRF.

    ``dense_weight`` in [0, 1] scales the dense list's contribution by
    dense_weight and the lexical list's by (1 - dense_weight). 0.5 reproduces the
    original equal-weight ordering (uniform scaling doesn't reorder); 1.0 is
    dense-only; 0.0 is lexical-only. This is the fusion-weight ablation lever.
    """
    w_d = dense_weight
    w_l = 1.0 - dense_weight
    scores: dict[int, float] = {}
    obj: dict[int, RetrievedChunk] = {}
    prefer_dense: dict[int, RetrievedChunk] = {}

    for rank, ch in enumerate(dense, start=1):
        scores[ch.chunk_id] = scores.get(ch.chunk_id, 0.0) + w_d * (1.0 / (k_rrf + rank))
        obj.setdefault(ch.chunk_id, ch)
        prefer_dense[ch.chunk_id] = ch  # dense cosine is the nicer display score
    for rank, ch in enumerate(lexical, start=1):
        scores[ch.chunk_id] = scores.get(ch.chunk_id, 0.0) + w_l * (1.0 / (k_rrf + rank))
        obj.setdefault(ch.chunk_id, ch)

    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
    # Prefer the dense-scored object for display when a chunk was in both lists.
    return [prefer_dense.get(cid, obj[cid]) for cid in ordered]


def hybrid_search(
    conn: psycopg.Connection,
    query_vector: list[float],
    question: str,
    top_k: int,
    *,
    candidates: int = 20,
    k_rrf: int = 60,
    dense_weight: float = 0.5,
) -> list[RetrievedChunk]:
    """Dense + lexical retrieval fused by (weighted) RRF.

    ``candidates`` is how deep each retriever goes before fusion (wider than
    top_k so a chunk ranked, say, 12th by dense but 2nd by lexical can surface).
    ``query_vector`` is the embedded question (for dense); ``question`` is the
    raw text (for lexical). ``dense_weight`` tunes the dense/lexical balance
    (see ``_rrf_fuse``).
    """
    dense = dense_search(conn, query_vector, candidates)
    lexical = lexical_search(conn, question, candidates)
    return _rrf_fuse(dense, lexical, k_rrf=k_rrf, top_k=top_k, dense_weight=dense_weight)
