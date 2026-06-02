"""FinanceBench loader.

Loads the FinanceBench rows from Hugging Face and normalizes them into a small
``Question`` shape the rest of the pipeline uses. Each row carries the gold
``answer`` and the ``evidence`` spans that recall@k is scored against.

Field names follow the FinanceBench dataset card. This loader does not trust
them blindly: if an expected field is absent it raises a ValueError that prints
the columns actually present, so a schema change on the dataset surfaces loudly
instead of silently mislabeling data. Confirm against the card if this fires.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Question:
    id: str
    question: str
    answer: str
    doc_name: str
    evidence_texts: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    question_type: str | None = None


# Candidate column names per logical field (first present wins).
_FIELD_CANDIDATES = {
    "id": ["financebench_id", "id"],
    "question": ["question"],
    "answer": ["answer"],
    "doc_name": ["doc_name", "document_name"],
    "question_type": ["question_type", "category"],
}


def _first_present(row: dict, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in row and row[name] is not None:
            return name
    return None


def _extract_evidence(row: dict) -> tuple[list[str], list[int]]:
    """Pull evidence text + page numbers from the (nested) evidence field."""
    ev = row.get("evidence") or row.get("evidence_text") or []
    texts: list[str] = []
    pages: list[int] = []
    if isinstance(ev, str):
        texts.append(ev)
    else:
        for item in ev:
            if isinstance(item, dict):
                txt = item.get("evidence_text") or item.get("text")
                if txt:
                    texts.append(txt)
                pg = item.get("evidence_page_num") or item.get("page_number") or item.get("page")
                if isinstance(pg, int):
                    pages.append(pg)
            elif isinstance(item, str):
                texts.append(item)
    return texts, pages


def load_questions(dataset_name: str = "PatronusAI/financebench") -> list[Question]:
    """Load and normalize FinanceBench rows. Requires the ``datasets`` library."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    # FinanceBench ships a single split; take the first available one.
    split = next(iter(ds.keys()))
    rows = ds[split]

    out: list[Question] = []
    for row in rows:
        resolved = {}
        for logical, candidates in _FIELD_CANDIDATES.items():
            key = _first_present(row, candidates)
            if key is None and logical in ("question", "answer", "doc_name"):
                raise ValueError(
                    f"FinanceBench row missing required field '{logical}'. "
                    f"Columns present: {sorted(row.keys())}. "
                    "Update _FIELD_CANDIDATES to match the dataset card."
                )
            resolved[logical] = row[key] if key else None

        texts, pages = _extract_evidence(row)
        out.append(
            Question(
                id=str(resolved["id"]) if resolved["id"] is not None else str(len(out)),
                question=resolved["question"],
                answer=resolved["answer"],
                doc_name=resolved["doc_name"],
                evidence_texts=texts,
                pages=pages,
                question_type=resolved["question_type"],
            )
        )
    return out


def locate_pdf(doc_name: str, data_dir: str | Path) -> Path | None:
    """Find ``{doc_name}.pdf`` under data_dir (case-insensitive). None if absent."""
    data_dir = Path(data_dir)
    target = f"{doc_name}.pdf".lower()
    for p in data_dir.rglob("*.pdf"):
        if p.name.lower() == target:
            return p
    return None


def _download(out_dir: str) -> int:
    """`make data` entry point.

    The FinanceBench PDFs are company-sourced and not redistributed on HF for
    licensing reasons; they are fetched via the official repo's downloader. This
    command does not invent a download API — it points at the source and checks
    whether PDFs are already present, so the dependency is explicit.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.rglob("*.pdf"))
    if existing:
        print(f"Found {len(existing)} PDFs under {out}/ — nothing to do.")
        return 0
    print(
        "No PDFs found. Download the FinanceBench PDFs from the official repo and "
        f"place them under {out}/:\n"
        "  https://github.com/patronus-ai/financebench\n"
        "(see its pdfs/ directory and download script). Re-run `make ingest` after."
    )
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="FinanceBench data helper")
    ap.add_argument("--download", action="store_true", help="check/instruct PDF download")
    ap.add_argument("--out", default="data/", help="data directory")
    args = ap.parse_args()
    if args.download:
        sys.exit(_download(args.out))
    qs = load_questions()
    print(f"Loaded {len(qs)} FinanceBench questions.")


if __name__ == "__main__":
    main()
