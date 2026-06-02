"""Chunking.

Two strategies, both ablation knobs in ``configs/*.yaml``:

* ``token``              : fixed-size token windows with overlap.
* ``section_then_token`` : split on SEC item headers first, then token-window
                           within each section so a chunk never straddles two
                           Items and each chunk carries its section label.

The token encoder is injected (``Encoder`` protocol) so this module is testable
without tiktoken and so the encoder itself becomes a swappable variable. The
default encoder is tiktoken's ``cl100k_base``, which is what text-embedding-3-*
tokenizes with, so token budgets here match what the embedding API sees.

Known V0 limitation (candidate ablation, not hidden): pages are chunked
independently, so content that spans a page boundary is not merged. Cross-page
merging is a V1 option.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


def tiktoken_encoder(name: str = "cl100k_base") -> Encoder:
    """Default production encoder. Imported lazily so tests need no tiktoken."""
    import tiktoken

    return tiktoken.get_encoding(name)


@dataclass
class Chunk:
    content: str
    token_count: int
    chunk_index: int
    page: int | None = None
    section: str | None = None


# SEC item headers, e.g. "Item 1A. Risk Factors", "ITEM 7. MANAGEMENT'S ...".
_ITEM_RE = re.compile(r"(?im)^[ \t]*(item\s+\d+[a-z]?\b\.?[^\n]*)$")


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split text at SEC item headers.

    Returns a list of (section_label, section_text). Text before the first
    header (or all text, if no headers) is returned with section_label=None.
    """
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        return [(None, text)]

    out: list[tuple[str | None, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        out.append((None, preamble))
    for i, m in enumerate(matches):
        label = " ".join(m.group(1).split())  # normalize whitespace in header
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start() : end].strip()
        if body:
            out.append((label, body))
    return out


def chunk_tokens(
    text: str, encoder: Encoder, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Sliding token windows of ``max_tokens`` with ``overlap_tokens`` overlap."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("require 0 <= overlap_tokens < max_tokens")

    tokens = encoder.encode(text)
    if not tokens:
        return []
    if len(tokens) <= max_tokens:
        return [text]

    step = max_tokens - overlap_tokens
    out: list[str] = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        out.append(encoder.decode(window))
        if start + max_tokens >= len(tokens):
            break
        start += step
    return out


def chunk_document(
    pages: str | list[str],
    encoder: Encoder,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    strategy: str = "token",
) -> list[Chunk]:
    """Chunk a document into ordered ``Chunk`` objects.

    ``pages`` is either a single string or a list of page texts (1-indexed page
    numbers are attached for citations). ``strategy`` is "token" or
    "section_then_token".
    """
    if isinstance(pages, str):
        numbered = [(None, pages)]
    else:
        numbered = [(i + 1, p) for i, p in enumerate(pages)]

    chunks: list[Chunk] = []
    idx = 0
    for page_no, page_text in numbered:
        if not page_text or not page_text.strip():
            continue
        if strategy == "section_then_token":
            segments = split_sections(page_text)
        elif strategy == "token":
            segments = [(None, page_text)]
        else:
            raise ValueError(f"unknown chunking strategy: {strategy!r}")

        for section, seg_text in segments:
            for piece in chunk_tokens(seg_text, encoder, max_tokens, overlap_tokens):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        content=piece,
                        token_count=len(encoder.encode(piece)),
                        chunk_index=idx,
                        page=page_no,
                        section=section,
                    )
                )
                idx += 1
    return chunks
