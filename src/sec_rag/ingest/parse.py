"""PDF -> text.

V0 uses pypdf for text extraction, page by page, so each chunk can carry a page
number for citations. pypdf does not recover table structure; table extraction
is an explicit V2 ablation (unstructured / llama-parse), not done here.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def _clean(text: str) -> str:
    """Strip NUL bytes from extracted text.

    pypdf occasionally emits stray NUL (0x00) bytes from malformed PDF text
    streams. They are extraction noise, not content — but PostgreSQL ``text``
    columns reject NUL, so an unfiltered page kills the load (psycopg DataError).
    Strip them here, at the parse boundary, so every downstream stage (chunk
    token counts, the embedding call, the DB insert) sees the same clean text.
    """
    return text.replace("\x00", "")


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Return one text string per page (empty string for pages with no text).

    Page order is preserved. Downstream (chunk_document) numbers pages 1-based,
    which is deliberate: a citation's page must match what a human sees in a PDF
    viewer and the printed page footer (both 1-based). Do NOT "align" this to
    FinanceBench's ``evidence_page_num``, which is a raw 0-based array index used
    only for its internal bookkeeping, not for display.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    return [_clean(page.extract_text() or "") for page in reader.pages]
