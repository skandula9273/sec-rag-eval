# sec-filings-rag

Read this file fully before touching code. It is the working contract for this repo.

## What this is

A retrieval-augmented QA service over US SEC filings (10-K, 10-Q, 8-K) plus an
eval harness that scores retrieval, latency, and cost against a public benchmark
(FinanceBench). The point of the project is the **eval rigor and the engineering
underneath**, not the chatbot. Numbers must be reproducible and honest.

Owner: Sai (Santosh Kandula). Audience for the work: engineering leads who will
read the code and scrutinize the numbers. Treat every output that way.

## Where we are now (read before planning) — updated 2026-06-30

- Phase: **Shipped — benchmarked V2 engine + a live EDGAR product on top.** Two
  surfaces share one RAG engine. The retrieval ablation is complete; the product
  has been built out and deployed.
- **Live EDGAR product (built after V2):** a static frontend on GitHub Pages
  (`skandula9273.github.io/sec-rag-eval`, code in `web/`) calling the deployed API.
  Enter a ticker → fetch that company's newest **10-K/10-Q/8-K** live from EDGAR
  (auto-detected; multi-filing compare for year-over-year), index on demand
  (`src/sec_rag/edgar/`, in-memory + Neon cache), stream a grounded, **section-cited**
  answer. **BYOK** (own keys via headers), per-IP **rate limit**, optional pre-warm.
  Endpoints: `/query`, `/query/stream`, `/query/live/stream`. The original
  FinanceBench corpus path is unchanged and still the measured benchmark.
  Version history + decisions: `docs/versions.md`, `docs/decisions-and-steps.md`.
- Phase (engine): **V2 retrieval config productionized.** The ablation program
  found the recall lever (the embedding model) and it is the LIVE config.
- **Live config = `configs/v2.yaml`: dense + text-embedding-3-large @1536-d +
  1024-token chunks.** Corpus re-ingested into Neon: **84 docs, 15,192 chunks,
  274 MB** (fits the free tier). ⚠️ The deployed Cloud Run API must set
  `SEC_RAG_CONFIG=configs/v2.yaml` — the query and corpus embedding model must
  match or retrieval is incoherent. See `DEPLOY.md`.
- **Current baseline** — `eval_results/financebench_20260629T160938Z.json`
  (v2, 150 q, dense, retrieval-only, fuzzy(0.5)), vs the committed v0 baseline:

  | Metric | v0 (3-small/512) | **v2 (3-large@1536/1024)** | V2 target |
  |---|---|---|---|
  | recall@5 | 0.44 | **0.64** | 0.75 |
  | recall@10 | 0.54 | **0.74** | — |
  | MRR | 0.317 | **0.492** | — |
  | tables@5 | 0.32 | **0.70** | — |
  | faithfulness | 0.941 | **0.929** ✓ | 0.80 |
  | cost / query | $0.0063 | **$0.009** (eval; API ~$0.005–6) | <$0.005 |
  | p95 latency (e2e) | ~15.6 s | **~15.3 s** (eval w/ judge; API faster) | <2.5 s ✗ |

  Full v2 baseline: `eval_results/financebench_20260629T193049Z.json` (150 q,
  full pipeline, 0 errors). Cost/latency above are EVAL numbers (top_k=10 + judge
  on); the **live API** runs top_k=5 + judge off (faithfulness opt-in), so it is
  cheaper + faster. recall@5 0.44 → **0.64** and tables 0.32 → **0.70** came from
  ONE lever — the
  embedding model (3-large, Matryoshka-truncated to 1536-d so it fits `vector(1536)`
  and the free tier) + larger (1024) chunks. recall@10 0.747 is essentially at the
  0.75 target. Five other levers (hybrid, reranker ×2, table-extraction, smaller
  chunks) were measured and rejected — full ablation table in `docs/depth-round.md`.

- **Remaining gaps:** (1) **latency** — p95 ~15.6 s vs <2.5 s, untouched; the free
  engineering track (connection pool + faithfulness judge off the request path).
  (2) The **full v2 eval** (recall + faithfulness + cost through generation) is
  pending Anthropic credits; only retrieval-only recall is measured so far.
- Authoritative current state: this section + `docs/depth-round.md` (the ablation
  record). The dated session summaries are historical.

### V1.1 status — hybrid tested and RETIRED (dense is the ceiling)

The clean A/B is done — via the retrieval-only eval mode (`--no-generate`, 150 q,
fuzzy, 0 errors). recall@k / MRR need no Anthropic (only OpenAI embeddings + the
DB), and dense reproduced the committed 0.44 *exactly*, validating the harness.

**Result:** hybrid (dense + Postgres FTS, RRF) **regressed** — recall@5 0.44 →
**0.347**, every category down, including tables (0.32 → 0.26). The fusion-weight
sweep (`eval_results/ablation_fusion_20260615T214346Z.json`) shows **no blend
beats dense**: lexical-only recall@5 **0.04**, tables@5 **0.00**.

**Conclusion (decided):** pure dense is the recall ceiling on this corpus; hybrid
is **retired as a recall lever** (dense stays the default; the `dense_weight` knob
is kept, favouring dense). **The deeper finding:** lexical's 0.00 on tables means
the exact line-item terms aren't in the chunk text — **pypdf flattened the tables
at parse time** — so no retriever can surface evidence parsing already destroyed.
The tables gap is a **parsing problem, not a retrieval-method one** → **table
extraction** (unstructured.io / llama-parse) is the next real lever, ahead of the
reranker. Full reasoning + depth-round write-up: `docs/depth-round.md`. Committed
A/B JSONs: `financebench_20260615T210625Z.json` (dense), `...212005Z.json`
(hybrid), `ablation_fusion_20260615T214346Z.json` (sweep).

## Conceptual model

Three pipelines share one query path. The API and the eval harness call the same
`QueryEngine` — never build a second path. The retriever is selected by
`cfg.retrieval.method` (`dense` | `hybrid`); both paths return the same shape.

```mermaid
flowchart TD
    subgraph INGEST["Ingestion (offline, make ingest)"]
      P[FinanceBench PDFs] --> PA[parse.py — pypdf, page by page]
      PA --> CH[chunk.py — section-aware + token windows]
      CH --> EM[embed.py — text-embedding-3-small, 1536-dim]
      EM --> LD[load.py — idempotent upsert]
    end
    LD --> DB[("Neon Postgres + pgvector<br/>documents + chunks<br/>HNSW cosine + GIN tsvector")]

    subgraph QUERY["Query (FastAPI /query)"]
      Q[user question] --> QE[embed query]
      QE --> RT{"retrieval.method"}
      RT -- dense --> DS[dense.py — pgvector &lt;=&gt; cosine, top_k]
      RT -- hybrid --> HY["hybrid.py — dense + lexical.py (FTS), RRF fuse"]
      DS --> GEN["answer.py — Claude Haiku, grounded, numbered citations"]
      HY --> GEN
      GEN --> FA["faithfulness.py — Haiku judge (badge)"]
      FA --> RESP[QueryResponse: answer + citations + metrics]
    end
    DB --> DS
    DB --> HY

    subgraph EVAL["Eval (make eval)"]
      FB[FinanceBench questions] --> RUN[run through QueryEngine]
      RUN --> SC[evidence_match_rank → recall@k, MRR]
      SC --> JSON[timestamped JSON committed to eval_results/]
    end
    RESP -. same engine .-> RUN
```

## Scope — phase boundaries

The design doc (`docs/design-doc.md`) describes the full V0→V2 system and is
**locked**. Any deviation requires a **dated amendment in that doc with a
rationale** — say so and propose the amendment, do not silently implement. (Five
amendments exist already; read the `## Amendments` section before proposing one.)

**V1 is executed as separate, individually-measured increments on the same
FinanceBench corpus** (per the 2026-06-06 amendment), to keep one variable
changing at a time against the committed 0.44 baseline:

- **V1.1 — Hybrid retrieval** (dense + Postgres FTS + RRF). **DONE — retired.**
  Tested: no fusion blend beats dense; the tables gap is upstream (parsing). See
  the V1.1 status above.
- **V1.2 — Cross-encoder reranker** (BGE base) over the top dense candidates.
  Still planned, but reframed by the V1.1b finding: a reranker only re-orders what
  retrieval already surfaced, so it **cannot recover table evidence pypdf
  destroyed** — expect a prose-side precision (MRR) win, not a tables fix.
- **V1.3 — Corpus expansion** (S&P 100 via EDGAR + Finnhub news) + 100 hand-built
  labeled custom queries.
- **V1.4 — Full three-layer eval** (FinanceBench + custom 100 + faithfulness
  judge); per-category ablation table is the deliverable.

**Open resequencing (needs a dated design-doc amendment before acting):** the
V1.1b ablation is evidence that **table extraction** — parked in V2 below — is the
real lever on the 0.32 tables gap and should likely come *before* the
reranker/corpus work. Propose the amendment; do not silently reorder.

**Still out of scope until V2** — do not pull forward without a told-to-do-it:
time-decay scoring, table-extraction ablation, embedding-model comparison,
observability dashboards (OTel/Grafana/LangFuse), agentic tool calls.

**Lexical backend is fixed: core Postgres FTS** (`tsvector` + `ts_rank_cd`, GIN
index). `pg_search`/ParadeDB BM25 is **deprecated on Neon** and cannot be enabled
(verified against the live DB). Do not reach for it again.

## Engineering rules (non-negotiable)

1. **No fake APIs, ever.** Before calling a library function, confirm it exists in
   the pinned version. If unsure, say so — never invent a clean-looking call.
2. **Never fake or cherry-pick numbers.** Honest metrics even when they hurt.
   (recall@5 0.44 is below floor and stays reported as-is. Primary recall metric
   is **fuzzy(0.5)**; substring is published alongside as a strict lower bound.)
3. **No code I can't read line by line.** Explain *why* (why this loss, this index,
   this chunk size), not just what. Pair every choice with its reason.
4. **Reproducible by default.** Fixed seeds (13), pinned versions
   (`requirements.lock`), no hardcoded local paths, temperature 0.0. If it can't
   be rerun, it doesn't count.
5. **Eval on real + edge cases**, not the easy split. Validate where it should
   break (dense confusing "grew 5%" vs "grew 25%" — the tables/numbers failure
   mode now measured at recall@5 0.32).
6. **Ablation-friendly structure.** Every knob lives in `configs/*.yaml` so one
   variable changes at a time. `v0.yaml` is frozen so the baseline stays
   reproducible; new work goes in `v1.yaml`. Strong experimental design > one
   impressive number.
7. **Simpler method first.** Dense baseline before anything heavier. No reaching
   for a bigger model when a smaller one answers the question.
8. **Secrets never committed.** `.env` is gitignored. Keys come from env only.
   The deployed API is guarded by an `X-API-Key` header (see `api/app.py`).
9. **FinanceBench is CC-BY-NC-4.0.** Non-commercial portfolio use. Do not
   redistribute the PDFs.

"Done" = working code + clear metrics + a short writeup of decisions and tradeoffs.
Match that shape without being asked.

## Known engineering debt (on record, address before claiming production-grade)

- **Single shared DB connection serializes queries.** Safe (verified 6/6
  concurrent) but a throughput ceiling and a latency contributor. A connection
  pool is the fix.
- **p95 latency ~15.6 s is over the <5 s floor.** Generation + the inline
  faithfulness judge (a second Haiku call) dominate the critical path. Levers:
  connection pool, move the judge off the request path, batch.
- **Eval runner swallows infra failures.** The per-question resilience (commit
  `3b3880b`) let a billing outage masquerade as 77 question failures while still
  emitting aggregate recall. Harden it to treat billing/auth errors as fatal (or
  suppress aggregate metrics when `n_scored << n_questions`) so a partial run
  never looks like a result.
- Retrieval score is `1 - cosine_distance` (theoretically [-1,1]); clamp-for-
  display is optional. Lockfile pins the project itself, so re-run `pip install
  -e .` after `pip install -r requirements.lock`.

## Commands

```
make install                 # install deps into the env
make lock                    # freeze exact versions -> requirements.lock (commit it)
make db-init                 # apply db/schema.sql to Neon (needs DATABASE_URL)
make data                    # fetch FinanceBench PDFs into data/
make ingest                  # parse -> chunk -> embed -> load into pgvector
make eval                    # run FinanceBench eval (uses configs/v0.yaml)
make eval CONFIG=configs/v1.yaml   # eval the hybrid config -> timestamped JSON
make demo                    # launch Streamlit demo (start the API first)
make test                    # pytest suite (chunk, metrics, fusion, schemas, auth...)
```

Deploy is documented in `DEPLOY.md` (Cloud Run; `Dockerfile`, `Dockerfile.demo`).
Run `make test` and the relevant live `make` target after any change. Don't
report a task done on logic-only checks if it touches a live service — and don't
cite an eval JSON whose `n_scored < n_questions` or `n_errors > 0`.

## Project layout

```
src/sec_rag/
  config.py            # Secrets (env) split from Config (yaml); .require() fails loud
  pipeline.py          # QueryEngine — the one shared path (API + eval); picks retriever
  db/{schema.sql,pool.py}      # documents+chunks, vector(1536) HNSW + tsvector GIN; psycopg3
  ingest/{financebench,parse,chunk,embed,load}.py
  retrieve/dense.py    # pgvector <=> cosine, score = 1 - distance
  retrieve/lexical.py  # Postgres FTS (tsvector + ts_rank_cd), OR-ranked tsquery
  retrieve/hybrid.py   # dense + lexical, Reciprocal Rank Fusion (k≈60)
  generate/answer.py   # Claude Haiku, grounded prompt, parses [n] back out; PRICING
  generate/faithfulness.py     # self-contained Haiku judge (RAGAS definition); badge
  api/{app.py,schemas.py}      # FastAPI /health /query; X-API-Key guard; schema = contract
  eval/{metrics.py,run_financebench.py}    # recall@k/MRR/faithfulness/cost; --sleep throttle
configs/v0.yaml        # frozen V0 baseline knobs
configs/v1.yaml        # hybrid: retrieval.method, fusion params, candidates, k_rrf
demo/streamlit_app.py  # cited vs retrieved badges; faithfulness badge
Dockerfile, Dockerfile.demo, DEPLOY.md   # Cloud Run deploy
tests/                 # chunk, metrics, financebench, pricing, faithfulness, schemas, auth, hybrid
eval_results/          # committed JSON, one file per run (only complete runs)
data/                  # FinanceBench PDFs (gitignored)
```

## Live-services map (what each thing actually unblocks)

- **Neon Postgres** → the vector store + lexical store. `make db-init` applies
  `db/schema.sql` (HNSW cosine + GIN tsvector). `vector` extension is enabled.
- **OPENAI_API_KEY** → `embed.py`. Embeds chunks (ingest) and the query (every
  `/query`). Dim stays 1536 to match the schema.
- **ANTHROPIC_API_KEY** → `answer.py` (generation) **and** `faithfulness.py` (the
  judge). A depleted balance breaks `/query` generation and any eval run — top up
  before a full `make eval`.
- **DATABASE_URL** → `db/pool.py`. Long-lived autocommit conn for the engine,
  context-managed for ingest.
- **FinanceBench PDFs** → `data/`. `parse.py` reads page-by-page so citations
  carry 1-based page numbers (page-0 falsy-drop bug fixed).

Fill `.env` from `.env.example`, then `make install && make lock` and commit the lock.

## Reference docs

All live under `docs/` so Claude Code can read and `@`-mention them:

- `docs/design-doc.md` — locked V0→V2 design + the `## Amendments` log. Source of
  truth for scope, eval design, and success criteria. Read the amendments.
- `docs/v1-plan.md` — current V1 sequencing, the hybrid design, the lexical-backend
  findings, and the pre-registered V1.1 success criterion.
- `docs/session-summary-2026-05-29.md`, `-2026-05-21.md` — historical (V0 scaffold
  era). Do not treat as current state.

When in doubt about scope, the design doc (+ amendments) wins. When in doubt about
current state, this section + `v1-plan.md` win. When something contradicts these
rules, stop and ask.
