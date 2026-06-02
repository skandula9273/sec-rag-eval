# Session summary — May 29, 2026
## Flagship SEC Filings RAG — Task #3 done (repo scaffolded)

**Status:** V0 skeleton standing at `OUTPUTS/flagship-sec-rag/sec-filings-rag/`.
Dense-retrieval only, per locked scope. Repo name: `sec-filings-rag`.
**V0 deadline:** June 14, 2026.
**Schedule note:** today is May 29. The May 21 week plan put Task #3 (scaffold) +
Task #4 (Neon schema) in W1 (May 22–24). Scaffold landed May 29 → ~1 week behind.
W2 (ingestion) work is written but not yet run against live services.

---

## What got built this session

Scaffolded the full V0 tree. Dense retrieval only — no BM25, no reranker, no
agent calls, no corpus beyond FinanceBench (drift watch held).

- **Metadata:** `pyproject.toml` (constraint floors + lockfile plan), `Makefile`
  (install/lock/db-init/data/ingest/eval/demo/test), `README.md`, `.gitignore`,
  `.env.example`, `configs/v0.yaml` (every knob is an ablation lever).
- **Config:** `src/sec_rag/config.py` — secrets (env) split from ablation config (yaml).
- **DB:** `db/schema.sql` (documents + chunks, `vector(1536)`, HNSW cosine index),
  `db/pool.py` (psycopg3 + `register_vector`).
- **Ingestion:** `ingest/{financebench,parse,chunk,embed,load}.py`.
- **Retrieval:** `retrieve/dense.py` (pgvector `<=>` cosine, score = 1 − distance).
- **Generation:** `generate/answer.py` (Claude Haiku, parses `[n]` citations back out).
- **API:** `api/app.py` (`/health`, `/query`, lifespan-built engine), `api/schemas.py`
  (the design-doc JSON contract; rerank/faithfulness fields present but Optional).
- **Pipeline:** `pipeline.py` — one path shared by API and eval.
- **Eval:** `eval/metrics.py` (recall@k, MRR, evidence-match), `eval/run_financebench.py`
  (timestamped JSON to `eval_results/`).
- **Demo:** `demo/streamlit_app.py` (answer + cited-vs-retrieved source badges + metrics).
- **Tests:** `tests/test_chunk.py`, `tests/test_metrics.py`.

## Verified

- `python -m py_compile` clean across all 23 modules.
- Chunker + metrics logic: 32/32 assertions pass (run via standalone harness; the
  sandbox had no PyPI access to install pytest — run `make test` locally to get
  the same checks through the committed pytest suite).

What is **not** yet verified (needs keys + Neon + the PDFs, i.e. live services):
embeddings, generation, pgvector load/retrieve, `/query`, the eval runner. These
are written with real API-call patterns against the pinned libs and raise clear
errors when keys/DATABASE_URL are missing — but no end-to-end run has happened.

---

## Open decision (needs your call — possible design-doc amendment)

**Faithfulness badge in V0.** The locked design doc's demo spec says V0 shows a
RAGAS faithfulness badge on every answer. The V0 success criterion + week plan
only schedule recall@5, and RAGAS inline adds a judge call (~500–700 ms, ~$0.001)
to the V0 critical path. I scaffolded with `eval.faithfulness: false` (badge slot
renders "—") to protect V0. Two ways to resolve, both need a dated doc amendment:

- **A — wire RAGAS inline now:** honors the demo spec; adds the `ragas` dep + the
  judge call to V0.
- **B — defer to V1:** amend the doc to move the live faithfulness badge to V1;
  V0 demo shows recall-focused metrics only.

**Pricing placeholder.** `generate/answer.py` `PRICING` is set to 0.0 and flagged
as estimate (`cost_is_estimate: true` flows through the schema and eval JSON).
Set real Haiku rates from the pricing page before any cost number goes in the writeup.

---

## Next actions (Task #4 → #6)

1. **Task #4:** create the Neon DB, then `make db-init` to apply `schema.sql`.
   Confirm the `vector` extension is enabled on the Neon instance.
2. Fill `.env` (OPENAI_API_KEY, ANTHROPIC_API_KEY, DATABASE_URL), `make install`,
   `make lock`, commit `requirements.lock`.
3. Confirm the FinanceBench HF column names against the dataset card. `financebench.py`
   raises with the real column list if they differ — adjust `_FIELD_CANDIDATES`.
4. **Task #5:** `make data` (PDFs into `data/`), then `make ingest`. End-of-week
   check from the W2 plan: top-5 retrieval on ~10 sample questions returns the
   evidence chunk.
5. **Task #6:** `make eval` for the baseline recall@5/@10 → commit the JSON.
6. Decide on the open faithfulness question above and amend the doc.

## Drift held this session

No hybrid retrieval, no reranker, no agent/tool calls, no corpus expansion, no UI
polish ahead of retrieval, no new papers, eval dataset not reopened.
