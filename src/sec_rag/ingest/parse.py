"""PDF -> text.

V0 uses pypdf for text extraction, page by page, so each chunk can carry a page
number for citations. pypdf does not recover table structure; table extraction
is an explicit V2 ablation (unstructured / llama-parse), not done here.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Return one text string per page (empty string for pages with no text)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "") for page in reader.pages]
