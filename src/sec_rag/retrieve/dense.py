"""Dense retrieval over pgvector (V0).

Cosine distance via the pgvector ``<=>`` operator, which is matched by the
``vector_cosine_ops`` HNSW index in db/schema.sql. ``<=>`` returns a distance in
[0, 2]; retrieval_score is reported as ``1 - distance`` so higher is better and
the number reads as a cosine similarity.

V0 is dense only. Hybrid (BM25 + dense) and reranking are deliberately not here;
they are V1/V2 and will live in sibling modules (retrieve/hybrid.py,
retrieve/rerank.py) so the dense path stays a clean ablation baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_name: str
    ticker: str | None
    filing_type: str | None
    filing_date: str | None
    page: int | None
    section: str | None
    content: str
    retrieval_score: float


_SQL = """
SELECT c.id, d.doc_name, d.ticker, d.filing_type, d.filing_date,
       c.page, c.section, c.content,
       1 - (c.embedding <=> %(qvec)s) AS score
FROM chunks c
JOIN documents d ON d.id = c.doc_id
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> %(qvec)s
LIMIT %(k)s
"""


def dense_search(
    conn: psycopg.Connection, query_vector: list[float], top_k: int
) -> list[RetrievedChunk]:
    with conn.cursor() as cur:
        cur.execute(_SQL, {"qvec": query_vector, "k": top_k})
        rows = cur.fetchall()
    return [
        RetrievedChunk(
            chunk_id=r[0],
            doc_name=r[1],
            ticker=r[2],
            filing_type=r[3],
            filing_date=str(r[4]) if r[4] is not None else None,
            page=r[5],
            section=r[6],
            content=r[7],
            retrieval_score=float(r[8]),
        )
        for r in rows
    ]
