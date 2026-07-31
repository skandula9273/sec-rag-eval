"""Pluggable evidence matchers — each on pairs with a known right answer.

Includes the two cases designed to SEPARATE the matchers:
  * high token overlap but wrong meaning  -> overlap says match, semantic says no
  * correct meaning but low token overlap -> overlap says no,    semantic says yes
The strict matcher (exact substring) says no to both, being the most conservative.
"""

import numpy as np

from sec_rag.eval.metrics import (
    OverlapMatcher,
    SemanticMatcher,
    StrictMatcher,
    evidence_match_rank,
)


class StubEmbedder:
    """Deterministic offline embedder for tests. Each text gets a one-hot vector by
    the first meaning-tag keyword it contains; same tag -> cosine 1, different -> 0,
    unknown -> zero vector (no match). Lets us assert semantic behaviour with no
    network and no real embeddings."""

    def __init__(self, tags: list[tuple[str, int]]):
        self.tags = tags
        self.dim = max(i for _, i in tags) + 1

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        v = [0.0] * self.dim
        for kw, idx in self.tags:
            if kw in t:
                v[idx] = 1.0
                return v
        return v  # unknown -> zeros

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


# income/earnings are synonyms (same tag 0); liabilities(1) and assets(2) are distinct.
_TAGS = [("income", 0), ("earnings", 0), ("liabilities", 1), ("assets", 2)]


def test_strict_exact_substring_only():
    m = StrictMatcher()
    assert m.matches("... revenue grew 25% to $100M ...", ["revenue grew 25%"]) is True
    # a one-token difference is no longer a substring.
    assert m.matches("revenue grew 5% to $100M", ["revenue grew 25%"]) is False
    assert m.name == "strict"


def test_overlap_threshold_and_its_blind_spot():
    m = OverlapMatcher(threshold=0.5)
    # exact-ish -> match.
    assert m.matches("in 2022 revenue grew 25 percent", ["revenue grew 25 percent"]) is True
    # BLIND SPOT: number swapped (25 -> 5) but 3/4 tokens still overlap -> false match.
    assert m.matches("in 2022 revenue grew 5 percent", ["revenue grew 25 percent"]) is True
    # genuinely different, low overlap -> no match.
    assert m.matches("the board approved a new dividend", ["revenue grew 25 percent"]) is False


def test_semantic_paraphrase_low_overlap_matches():
    # correct meaning, low token overlap: overlap misses it, semantic catches it.
    gold = ["net income increased sharply"]
    chunk = "the firm reported that net earnings rose"
    assert OverlapMatcher(0.5).matches(chunk, gold) is False        # only "net" overlaps
    assert StrictMatcher().matches(chunk, gold) is False            # not a substring
    sem = SemanticMatcher(StubEmbedder(_TAGS), threshold=0.62)
    assert sem.matches(chunk, gold) is True                        # income ~ earnings


def test_semantic_high_overlap_wrong_meaning_rejected():
    # high token overlap, different claim (liabilities vs assets): overlap false-matches,
    # semantic correctly rejects, strict rejects.
    gold = ["total current liabilities of 2811 million"]
    chunk = "total current assets of 1554 million were reported"
    assert OverlapMatcher(0.5).matches(chunk, gold) is True         # total/current/of/million
    assert StrictMatcher().matches(chunk, gold) is False
    sem = SemanticMatcher(StubEmbedder(_TAGS), threshold=0.62)
    assert sem.matches(chunk, gold) is False                       # liabilities != assets


def test_semantic_threshold_is_respected():
    # spans must clear the 20-char minimum, so use realistic phrasings.
    gold = ["net income for the fiscal year"]
    chunk = "the company reported net earnings for the fiscal year"  # cosine 1 under the stub
    assert SemanticMatcher(StubEmbedder(_TAGS), threshold=0.99).matches(chunk, gold) is True
    # an unrelated topic -> zero vector -> cosine 0 -> below any positive threshold.
    assert SemanticMatcher(StubEmbedder(_TAGS), threshold=0.1).matches(
        "the weather forecast for tomorrow looks clear", gold
    ) is False


def test_evidence_match_rank_with_pluggable_matcher():
    # the first chunk high-overlap-wrong-meaning, the second the real paraphrase:
    # overlap ranks the wrong chunk first; semantic ranks the right one.
    chunks = [
        "total current assets of 1554 million were reported",  # overlap false-positive
        "the firm reported that net earnings rose",            # true semantic hit
    ]
    gold = ["net income increased sharply"]
    # overlap finds no chunk (gold shares no dense tokens with either); semantic finds #2.
    assert evidence_match_rank(chunks, gold, matcher=OverlapMatcher(0.5)) is None
    sem = SemanticMatcher(StubEmbedder(_TAGS), 0.62)
    assert evidence_match_rank(chunks, gold, matcher=sem) == 2
    # no evidence -> None regardless of matcher.
    assert evidence_match_rank(chunks, [], matcher=StrictMatcher()) is None


def test_semantic_vectors_are_normalized():
    # cosine via normalized dot product: identical-tag texts give 1.0, not >1.
    sem = SemanticMatcher(StubEmbedder(_TAGS), threshold=0.62)
    v = sem._vecs(["net income here", "net earnings there"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.isclose(float(v[0] @ v[1]), 1.0)
