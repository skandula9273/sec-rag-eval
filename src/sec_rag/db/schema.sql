-- pgvector schema for the FinanceBench corpus (V0).
-- Apply with:  psql "$DATABASE_URL" -f src/sec_rag/db/schema.sql
--
-- Embedding dimension is 1536 to match text-embedding-3-small. If you ablate the
-- embedding model to a different dimension, the column type and index must change
-- with it (that is why dim is also pinned in configs/*.yaml).

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source document (a FinanceBench PDF in V0).
CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    doc_name     TEXT NOT NULL UNIQUE,   -- FinanceBench doc_name, e.g. APPLE_2023_10K
    ticker       TEXT,
    company      TEXT,
    filing_type  TEXT,                   -- 10K | 10Q | 8K
    filing_date  DATE,
    source_path  TEXT,                   -- local path under data/
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per chunk. embedding nullable so parse/chunk can land before embed.
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    doc_id       BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,           -- order within the document
    page         INT,                    -- source page (for citations)
    section      TEXT,                   -- e.g. "Item 1A. Risk Factors", if detected
    content      TEXT NOT NULL,
    token_count  INT,
    embedding    vector(1536),
    UNIQUE (doc_id, chunk_index)
);

-- HNSW index for cosine distance. Matches retrieval.distance: cosine and the
-- pgvector <=> operator used in retrieve/dense.py.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Metadata filters (ticker / filing_type / date) used by retrieval in V1.
CREATE INDEX IF NOT EXISTS documents_ticker_idx ON documents (ticker);
CREATE INDEX IF NOT EXISTS documents_filing_idx ON documents (filing_type, filing_date);

-- Full-text (lexical) index for V1 hybrid retrieval. GIN over the english
-- tsvector of chunk content; matched by ts_rank_cd in retrieve/lexical.py.
-- (pg_search/BM25 is deprecated on Neon, so hybrid's lexical half uses core FTS.)
CREATE INDEX IF NOT EXISTS chunks_content_fts
    ON chunks USING gin (to_tsvector('english', content));
