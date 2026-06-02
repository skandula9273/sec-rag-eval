"""Chunker tests. Uses a character-level fake encoder (1 char = 1 token) so the
windowing is exact and deterministic without pulling in tiktoken.
"""

import pytest

from sec_rag.ingest.chunk import chunk_document, chunk_tokens, split_sections


class CharEncoder:
    """1 token per character; round-trips exactly, including spaces."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


ENC = CharEncoder()


def test_windows_with_overlap():
    # 10 chars, window 4, overlap 1 -> step 3 -> starts 0,3,6
    assert chunk_tokens("abcdefghij", ENC, 4, 1) == ["abcd", "defg", "ghij"]


def test_overlap_actually_overlaps():
    out = chunk_tokens("abcdefghij", ENC, 4, 1)
    assert out[0][-1] == out[1][0]  # 'd' shared between window 0 and 1


def test_short_text_single_chunk():
    assert chunk_tokens("abc", ENC, 4, 1) == ["abc"]
    assert chunk_tokens("abcd", ENC, 4, 1) == ["abcd"]


def test_empty_text():
    assert chunk_tokens("", ENC, 4, 1) == []


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_tokens("abcdef", ENC, 4, 4)  # overlap == max
    with pytest.raises(ValueError):
        chunk_tokens("abcdef", ENC, 4, -1)
    with pytest.raises(ValueError):
        chunk_tokens("abcdef", ENC, 0, 0)  # max_tokens must be > 0


def test_chunk_document_pages_and_indices():
    pages = ["aaaa", "", "bbbbbb"]  # page 2 empty -> skipped
    chunks = chunk_document(pages, ENC, max_tokens=4, overlap_tokens=0, strategy="token")
    # page1 -> "aaaa" (1 chunk); page3 -> "bbbb","bb" (2 chunks)
    assert [c.page for c in chunks] == [1, 3, 3]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert [c.token_count for c in chunks] == [4, 4, 2]


def test_split_sections_detects_items():
    text = "preamble text\nItem 1A. Risk Factors\nrisks here\nItem 7. MD&A\nmgmt text"
    sections = split_sections(text)
    labels = [s[0] for s in sections]
    assert labels[0] is None  # preamble
    assert "Item 1A. Risk Factors" in labels
    assert "Item 7. MD&A" in labels


def test_split_sections_no_headers():
    assert split_sections("just some text") == [(None, "just some text")]


def test_section_then_token_tags_sections():
    text = "Item 1A. Risk Factors\nrisk body\nItem 7. MD&A\nmgmt body"
    chunks = chunk_document(
        text, ENC, max_tokens=1000, overlap_tokens=0, strategy="section_then_token"
    )
    sections = {c.section for c in chunks}
    assert "Item 1A. Risk Factors" in sections
    assert "Item 7. MD&A" in sections


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        chunk_document("text", ENC, strategy="nope")
