"""Eval metric tests with hand-checked expected values."""

from sec_rag.eval.metrics import (
    evidence_match_rank,
    hit_rate_at_k,
    mean_reciprocal_rank,
    reciprocal_rank,
    set_recall_at_k,
)


def test_evidence_substring_rank():
    retrieved = ["foo bar", "the revenue grew 5 percent in fy2023", "baz"]
    assert evidence_match_rank(retrieved, ["revenue grew 5 percent"]) == 2


def test_evidence_substring_normalizes_whitespace_and_case():
    retrieved = ["Revenue   GREW  5\nPercent"]
    assert evidence_match_rank(retrieved, ["revenue grew 5 percent"]) == 1


def test_evidence_miss_returns_none():
    assert evidence_match_rank(["nothing relevant"], ["needle text"]) is None


def test_evidence_empty_returns_none():
    assert evidence_match_rank(["anything"], []) is None
    assert evidence_match_rank(["anything"], ["   "]) is None


def test_evidence_fuzzy_threshold():
    # 3 of 4 evidence tokens present -> 0.75 >= 0.5 -> hit at rank 1
    retrieved = ["alpha beta gamma omega extra words"]
    assert evidence_match_rank(
        retrieved, ["alpha beta gamma delta"], mode="fuzzy", threshold=0.5
    ) == 1
    # raise threshold above the overlap -> miss
    assert evidence_match_rank(
        retrieved, ["alpha beta gamma delta"], mode="fuzzy", threshold=0.8
    ) is None


def test_hit_rate_at_k():
    ranks = [1, 3, None, 2]
    assert hit_rate_at_k(ranks, 2) == 0.5   # ranks 1 and 2 are <= 2
    assert hit_rate_at_k(ranks, 5) == 0.75  # 1,3,2 hit; None misses
    assert hit_rate_at_k([], 5) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1, 2, None]) == (1.0 + 0.5 + 0.0) / 3


def test_set_recall_at_k():
    assert set_recall_at_k([1, 2, 3, 4], {3, 5}, 3) == 0.5
    assert set_recall_at_k([1, 2, 3], set(), 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([9, 3, 7], {3}) == 0.5
    assert reciprocal_rank([9, 8, 7], {3}) == 0.0
