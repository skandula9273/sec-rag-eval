"""Lexical (keyword) retrieval over Postgres full-text search (V1).

The lexical half of hybrid retrieval. Returns the same ``RetrievedChunk`` shape
as dense.py so hybrid.py can fuse the two ranked lists.

Why core Postgres FTS and not BM25: pg_search (ParadeDB's Okapi BM25) is
deprecated on Neon and cannot be enabled, so this uses ``ts_rank_cd`` over a GIN
``to_tsvector`` index (schema.sql: chunks_content_fts). ts_rank_cd is a
term-frequency lexical signal — not true BM25, but a defensible keyword ranker
that catches exact financial terms dense embeddings blur ("net sales", line-item
names), which is the V0-diagnosed weakness on table/number questions.

Query construction (verified empirically — see docs/v1-plan.md): the raw question
must NOT go to plainto_tsquery/websearch_to_tsquery, which AND the terms and
return zero hits for a normal question. Instead, run the question through
plainto_tsquery to strip stopwords + stem (it keeps meaningful short tokens like
"3m"), then OR the resulting lexemes so a chunk matching ANY significant term is
ranked, weighted by ts_rank_cd. retrieval_score is the raw ts_rank_cd value
(NOT comparable to dense cosine — hybrid fuses by RANK, not score; see hybrid.py).
"""

from __future__ import annotations

import psycopg

from sec_rag.retrieve.dense import RetrievedChunk

# Build an OR tsquery from the cleaned/stemmed lexemes of the question. We let
# Postgres do stopword removal + stemming via plainto_tsquery, then turn its
# AND (&) form into OR (|) so any-term matches rank instead of all-term.
_LEXQUERY_SQL = """
WITH q AS (
    SELECT replace(plainto_tsquery('english', %(question)s)::text, ' & ', ' | ') AS oq
)
SELECT c.id, d.doc_name, d.ticker, d.filing_type, d.filing_date,
       c.page, c.section, c.content,
       ts_rank_cd(to_tsvector('english', c.content),
                  to_tsquery('english', q.oq)) AS score
FROM chunks c
JOIN documents d ON d.id = c.doc_id, q
WHERE q.oq <> ''
  AND to_tsvector('english', c.content) @@ to_tsquery('english', q.oq)
ORDER BY score DESC
LIMIT %(k)s
"""


def lexical_search(
    conn: psycopg.Connection, question: str, top_k: int
) -> list[RetrievedChunk]:
    """Top-k chunks by full-text relevance to ``question``.

    Returns [] if the question reduces to no searchable terms (all stopwords).
    The score field carries ts_rank_cd; it is on a different scale than dense
    cosine, so callers must fuse by rank, not by mixing raw scores.
    """
    with conn.cursor() as cur:
        cur.execute(_LEXQUERY_SQL, {"question": question, "k": top_k})
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
