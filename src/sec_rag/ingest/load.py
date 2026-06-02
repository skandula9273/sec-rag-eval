"""Ingestion orchestrator: parse -> chunk -> embed -> load to pgvector.

Run: ``python -m sec_rag.ingest.load --config configs/v0.yaml``

Idempotent per document: re-ingesting a doc_name replaces its chunks rather than
duplicating them, so reruns during development stay clean. Documents are
deduped across questions (one PDF backs many FinanceBench questions).
"""

from __future__ import annotations

import argparse
import re

from sec_rag.config import Config, Secrets, load_config
from sec_rag.db.pool import connect
from sec_rag.ingest.chunk import Chunk, chunk_document, tiktoken_encoder
from sec_rag.ingest.embed import Embedder
from sec_rag.ingest.financebench import load_questions, locate_pdf
from sec_rag.ingest.parse import extract_pages

_FILING_RE = re.compile(r"(10K|10Q|8K)", re.IGNORECASE)
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _doc_metadata(doc_name: str) -> dict:
    """Best-effort metadata from a FinanceBench doc_name like APPLE_2023_10K.

    Heuristic only (used for V1 metadata filters); never trusted for eval.
    """
    filing = _FILING_RE.search(doc_name)
    year = _YEAR_RE.search(doc_name)
    company = doc_name.split("_")[0] if "_" in doc_name else None
    return {
        "company": company,
        "filing_type": filing.group(1).upper() if filing else None,
        "filing_date": f"{year.group(0)}-01-01" if year else None,
    }


def _upsert_document(cur, doc_name: str, source_path: str) -> int:
    meta = _doc_metadata(doc_name)
    cur.execute(
        """
        INSERT INTO documents (doc_name, company, filing_type, filing_date, source_path)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (doc_name) DO UPDATE
          SET source_path = EXCLUDED.source_path,
              company = EXCLUDED.company,
              filing_type = EXCLUDED.filing_type,
              filing_date = EXCLUDED.filing_date
        RETURNING id
        """,
        (doc_name, meta["company"], meta["filing_type"], meta["filing_date"], source_path),
    )
    return cur.fetchone()[0]


def _insert_chunks(cur, doc_id: int, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
    cur.executemany(
        """
        INSERT INTO chunks (doc_id, chunk_index, page, section, content, token_count, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        [
            (doc_id, c.chunk_index, c.page, c.section, c.content, c.token_count, emb)
            for c, emb in zip(chunks, embeddings)
        ],
    )


def ingest(cfg: Config, data_dir: str = "data/", secrets: Secrets | None = None) -> dict:
    secrets = secrets or Secrets()
    secrets.require("openai_api_key", "database_url")

    encoder = tiktoken_encoder(cfg.chunking.encoder)
    embedder = Embedder(cfg.embedding, secrets)

    doc_names = sorted({q.doc_name for q in load_questions(cfg.eval.dataset)})
    stats = {"documents": 0, "chunks": 0, "skipped": []}

    with connect(secrets) as conn:
        for doc_name in doc_names:
            pdf = locate_pdf(doc_name, data_dir)
            if pdf is None:
                stats["skipped"].append(doc_name)
                continue
            pages = extract_pages(pdf)
            chunks = chunk_document(
                pages,
                encoder,
                max_tokens=cfg.chunking.max_tokens,
                overlap_tokens=cfg.chunking.overlap_tokens,
                strategy=cfg.chunking.strategy,
            )
            if not chunks:
                stats["skipped"].append(doc_name)
                continue
            embeddings = embedder.embed([c.content for c in chunks])
            with conn.cursor() as cur:
                doc_id = _upsert_document(cur, doc_name, str(pdf))
                _insert_chunks(cur, doc_id, chunks, embeddings)
            conn.commit()
            stats["documents"] += 1
            stats["chunks"] += len(chunks)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest FinanceBench PDFs into pgvector")
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--data", default="data/")
    args = ap.parse_args()
    cfg = load_config(args.config)
    stats = ingest(cfg, data_dir=args.data)
    print(
        f"Ingested {stats['documents']} docs / {stats['chunks']} chunks. "
        f"Skipped (no PDF / no text): {len(stats['skipped'])}."
    )
    if stats["skipped"]:
        print("  missing:", ", ".join(stats["skipped"][:10]),
              "..." if len(stats["skipped"]) > 10 else "")


if __name__ == "__main__":
    main()
