"""Reranker reorder logic (pure, no model/torch)."""

from sec_rag.retrieve.dense import RetrievedChunk
from sec_rag.retrieve.rerank import _apply_scores


def _c(cid: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_name="D", ticker=None, filing_type=None,
        filing_date=None, page=1, section=None, content=f"chunk {cid}",
        retrieval_score=0.5,
    )


def test_reorders_by_score_desc():
    out = _apply_scores([_c(1), _c(2), _c(3)], [0.1, 0.9, 0.5], top_k=3)
    assert [c.chunk_id for c in out] == [2, 3, 1]


def test_truncates_to_top_k():
    out = _apply_scores([_c(i) for i in range(5)], [0.1, 0.2, 0.3, 0.4, 0.5], top_k=2)
    assert [c.chunk_id for c in out] == [4, 3]


def test_sets_rerank_score_and_preserves_retrieval_score():
    out = _apply_scores([_c(1), _c(2)], [0.7, 0.2], top_k=2)
    assert out[0].rerank_score == 0.7 and out[1].rerank_score == 0.2
    assert out[0].retrieval_score == 0.5  # original score untouched


def test_empty_in_empty_out():
    assert _apply_scores([], [], top_k=5) == []
