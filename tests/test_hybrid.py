"""RRF fusion tests (pure logic, no DB)."""

from sec_rag.retrieve.dense import RetrievedChunk
from sec_rag.retrieve.hybrid import _rrf_fuse


def _chunk(cid: int, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_name="D", ticker=None, filing_type=None,
        filing_date=None, page=1, section=None, content="x", retrieval_score=score,
    )


def test_chunk_in_both_lists_outranks_chunk_in_one():
    # chunk 1 is rank-2 in both lists; chunk 9 is rank-1 in dense only.
    dense = [_chunk(9), _chunk(1)]
    lexical = [_chunk(5), _chunk(1)]
    out = _rrf_fuse(dense, lexical, k_rrf=60, top_k=3)
    ids = [c.chunk_id for c in out]
    # 1 appears in both (2/(60+2)) > 9 in one (1/(60+1)) -> 1 ranks first
    assert ids[0] == 1


def test_dedupes_by_chunk_id():
    dense = [_chunk(1), _chunk(2)]
    lexical = [_chunk(1), _chunk(2)]
    out = _rrf_fuse(dense, lexical, k_rrf=60, top_k=10)
    assert sorted(c.chunk_id for c in out) == [1, 2]  # no duplicates


def test_respects_top_k():
    dense = [_chunk(i) for i in range(1, 11)]
    lexical = [_chunk(i) for i in range(11, 21)]
    out = _rrf_fuse(dense, lexical, k_rrf=60, top_k=5)
    assert len(out) == 5


def test_rank_order_matches_rrf_score():
    # dense rank1=a, rank2=b; lexical rank1=b -> b gets two contributions.
    a, b = _chunk(100), _chunk(200)
    out = _rrf_fuse([a, b], [b], k_rrf=60, top_k=2)
    assert [c.chunk_id for c in out] == [200, 100]


def test_empty_lexical_falls_back_to_dense_order():
    dense = [_chunk(1), _chunk(2), _chunk(3)]
    out = _rrf_fuse(dense, [], k_rrf=60, top_k=3)
    assert [c.chunk_id for c in out] == [1, 2, 3]


def test_prefers_dense_object_for_display_score():
    # same chunk_id in both; dense object has the cosine display score we want.
    dense_obj = _chunk(1, score=0.87)
    lexical_obj = _chunk(1, score=4.2)
    out = _rrf_fuse([dense_obj], [lexical_obj], k_rrf=60, top_k=1)
    assert out[0].retrieval_score == 0.87
