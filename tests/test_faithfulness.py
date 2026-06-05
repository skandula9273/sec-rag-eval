"""Faithfulness judge — pure parsing/scoring logic (no API call).

We unit-test the JSON parsing and score arithmetic by exercising the regex +
score formula directly; the live judge call is covered by manual verification
(grounded answer -> 1.0, hallucinated -> 0.0).
"""

import json

from sec_rag.generate.faithfulness import _JSON_RE


def _score_from_judge_text(text: str):
    """Mirror score_faithfulness's parse+score, isolated from the API call."""
    m = _JSON_RE.search(text)
    if not m:
        return 0.0
    try:
        d = json.loads(m.group(0))
        claims = int(d.get("claims", 0))
        supported = int(d.get("supported", 0))
    except (ValueError, TypeError):
        return 0.0
    supported = max(0, min(supported, claims))
    return 1.0 if claims == 0 else round(supported / claims, 4)


def test_all_claims_supported():
    assert _score_from_judge_text('{"claims": 4, "supported": 4}') == 1.0


def test_partial_support():
    assert _score_from_judge_text('{"claims": 4, "supported": 1}') == 0.25


def test_no_claims_is_faithful():
    # A refusal asserts nothing unsupported -> 1.0 (reward grounded refusals).
    assert _score_from_judge_text('{"claims": 0, "supported": 0}') == 1.0


def test_none_supported():
    assert _score_from_judge_text('{"claims": 3, "supported": 0}') == 0.0


def test_judge_overcount_is_clamped():
    # supported can't exceed claims even if the judge miscounts.
    assert _score_from_judge_text('{"claims": 2, "supported": 5}') == 1.0


def test_garbage_judge_output_scores_zero():
    assert _score_from_judge_text("the judge rambled with no json") == 0.0


def test_json_embedded_in_prose():
    assert _score_from_judge_text('Here is my grade: {"claims": 2, "supported": 1} done') == 0.5
