"""FinanceBench evidence-extraction tests.

Pure-logic only: ``_extract_evidence`` parses one already-loaded row dict, so it
needs no network or the ``datasets`` library.
"""

from sec_rag.ingest.financebench import _extract_evidence


def test_evidence_page_zero_is_retained():
    """FinanceBench evidence_page_num is 0-based; page 0 (a cover page) is a real
    value. A truthy ``or`` chain would drop it — this guards against that."""
    texts, pages = _extract_evidence(
        {"evidence": [{"evidence_text": "cover page text", "evidence_page_num": 0}]}
    )
    assert texts == ["cover page text"]
    assert pages == [0]


def test_evidence_normal_page():
    _, pages = _extract_evidence(
        {"evidence": [{"evidence_text": "x", "evidence_page_num": 59}]}
    )
    assert pages == [59]


def test_evidence_missing_page_yields_no_page():
    texts, pages = _extract_evidence({"evidence": [{"evidence_text": "no page here"}]})
    assert texts == ["no page here"]
    assert pages == []


def test_evidence_multiple_spans_keep_order():
    texts, pages = _extract_evidence(
        {
            "evidence": [
                {"evidence_text": "first", "evidence_page_num": 0},
                {"evidence_text": "second", "evidence_page_num": 12},
            ]
        }
    )
    assert texts == ["first", "second"]
    assert pages == [0, 12]


def test_evidence_string_form():
    """Some rows may carry evidence as a bare string rather than a list of dicts."""
    texts, pages = _extract_evidence({"evidence": "plain string evidence"})
    assert texts == ["plain string evidence"]
    assert pages == []


def test_evidence_empty():
    assert _extract_evidence({}) == ([], [])
    assert _extract_evidence({"evidence": []}) == ([], [])
