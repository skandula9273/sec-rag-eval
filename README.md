# sec-filings-rag

Retrieval-augmented question answering over US public-company SEC filings, plus a
measurement harness that scores retrieval, faithfulness, latency, and cost across
design ablations. The design doc is the spec: `../design-doc.md`.

The point of the project is not one pipeline. It is showing which design choices
(chunking, embedding model, retrieval method, reranking) move which metric, and at
what cost, measured against a public benchmark (FinanceBench).

## Status

V0, build phase. The skeleton is standing; retrieval is not yet measured. V0 done
means: a deployed FastAPI service that ingests the FinanceBench PDFs, runs dense
retrieval via pgvector, generates answers with Claude Haiku, and reports a
recall@5 baseline on the FinanceBench 150 questions. Target: June 14, 2026.

V0 is dense retrieval only. Hybrid retrieval and reranking are V1/V2 and are not
in this tree yet, by design.

## Layout

```
sec-filings-rag/
  configs/v0.yaml            ablation knobs (chunk size, embed model, top_k, ...)
  src/sec_rag/
    config.py                load .env + yaml into a typed Settings object
    ingest/
      financebench.py        load the FinanceBench dataset rows + locate PDFs
      parse.py               PDF -> page text (pypdf)
      chunk.py               token-based, section-aware chunker (implemented)
      embed.py               OpenAI embeddings wrapper (text-embedding-3-small)
      load.py                parse -> chunk -> embed -> write to pgvector
    db/
      schema.sql             documents + chunks tables, HNSW cosine index
      pool.py                psycopg connection + pgvector registration
    retrieve/dense.py        dense top-k over pgvector cosine distance
    generate/answer.py       Claude Haiku call, returns answer + citations
    api/
      app.py                 FastAPI: GET /health, POST /query
      schemas.py             response models matching the design-doc JSON
    eval/
      metrics.py             recall@k, MRR (implemented + tested)
      run_financebench.py    eval runner -> timestamped JSON in eval_results/
  demo/streamlit_app.py      V0 demo: answer + citations + faithfulness/latency badges
  tests/                     unit tests for chunk + metrics
  eval_results/              committed JSON, one file per run
  data/                      FinanceBench PDFs (gitignored, not redistributed)
```

## Setup

Requires Python 3.11, a Neon Postgres database with the `vector` extension
available, and OpenAI + Anthropic API keys.

```bash
cp .env.example .env          # fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, DATABASE_URL
make install                  # editable install + dev/demo extras
make lock                     # freeze exact versions -> requirements.lock (commit it)
make db-init                  # apply db/schema.sql to $DATABASE_URL
make data                     # download FinanceBench PDFs into data/
make ingest                   # parse -> chunk -> embed -> load to pgvector
make eval                     # FinanceBench recall@5/@10 -> eval_results/<timestamp>.json
make demo                     # Streamlit demo
```

## Implemented vs. stubbed

Implemented and tested now (deterministic, no network): the token/section-aware
chunker (`ingest/chunk.py`) and the eval metrics (`eval/metrics.py`).

Wired with real API call patterns but exercised only against live services
(keys + DB required): embeddings, generation, pgvector load/retrieve, the FastAPI
endpoints, and the eval runner. These have correct signatures against the pinned
library versions; functions that need a network call raise a clear error if keys
or `DATABASE_URL` are missing rather than failing deep in a call stack.

`# TODO(W2)` / `# TODO(W3)` / `# TODO(W4)` markers point at the work each
remaining V0 week fills in, per the week plan in
`../session-summary-2026-05-21.md`.

## Reproducibility

Fixed seed in `configs/v0.yaml` (`eval.seed`) for any sampling step.
`temperature: 0.0` for generation. `make lock` writes exact installed versions to
`requirements.lock`, which is committed alongside the constraint floor in
`pyproject.toml`. Eval output is written to `eval_results/<timestamp>.json` and
committed per run, so a number in the writeup traces to a specific file.

## License note

FinanceBench is CC-BY-NC-4.0. This is non-commercial portfolio work; the PDFs are
not redistributed in this repo (`data/` is gitignored). `make data` pulls them
from the source.
