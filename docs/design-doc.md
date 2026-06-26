# Flagship — SEC Filings RAG + Eval Platform

**Author:** Sai (Santosh Kandula)
**Date:** May 21, 2026
**Status:** Design doc v1.0 — locked May 21, 2026. Scope, eval, and visual surface all closed. Build starting.

---

## Project

A retrieval-augmented question-answering service over US public-company SEC filings (10-K, 10-Q, 8-K) and market news, plus a measurement system that benchmarks retrieval, faithfulness, latency, and cost across a suite of design ablations.

## Problem

Off-the-shelf LLMs cannot answer questions about specific company filings, because (a) they lack the source documents in context and (b) generating answers without retrieval produces unverifiable claims. Existing RAG demos in the financial domain typically ship one pipeline and call it done. They do not measure which design choices — chunking strategy, embedding model, retrieval method, reranker, time-decay scoring, table extraction — actually improve which metric, and at what cost.

This project closes that gap: a working financial RAG service plus the ablation infrastructure that proves which design decisions move which numbers, scored against a public benchmark.

## Success criteria

| Metric | V0 floor | V2 target | Source |
|---|---|---|---|
| Recall@5 (FinanceBench) | 0.55 | 0.75 | FinanceBench 150 |
| RAGAS faithfulness | 0.65 | 0.80 | FinanceBench + 100 hand-built |
| p95 end-to-end latency | < 5000 ms | < 2500 ms | Production traces |
| Cost per query | < $0.01 | < $0.005 | Token + API usage logs |

The point isn't the absolute numbers — it's measurable improvement across ablations with the design choices documented.

## Architecture

```
INGESTION (cron, daily)
  EDGAR EFTS API ─► filing fetcher ─► HTML/XBRL parser ─┐
                                      (Items + tables)  │
  Finnhub news ─► news fetcher ─► article cleaner ──────┤
                                                        ▼
                                            ┌─────────────────────┐
                                            │  chunker             │
                                            │  (section-aware,     │
                                            │   512 tokens)        │
                                            └──────────┬──────────┘
                                                       ▼
                                            ┌─────────────────────┐
                                            │  embedder            │
                                            │  (text-emb-3-small)  │
                                            └──────────┬──────────┘
                                                       ▼
                                            ┌─────────────────────┐
                                            │  pgvector (Neon)     │
                                            │  docs+chunks+meta    │
                                            └─────────────────────┘

QUERY (FastAPI on Cloud Run)
  user query ─► /query ─► query embedder ─► hybrid retrieval
                                            (BM25 + dense + metadata
                                             filters + time-decay)
                                                       │
                                                       ▼
                                            reranker (BGE cross-encoder)
                                                       │
                                                       ▼
                                            top-5 chunks + prompt
                                                       │
                                                       ▼
                                            LLM (Claude Haiku)
                                                       │
                                                       ▼
                                            response + citations
                                                       │
                                                       └─► OTel trace ─► Grafana
                                                       └─► LangFuse trace

EVAL
  FinanceBench 150 ┐
                   ├─► run /query ─► log retrieval evidence +
  Custom 100 ──────┘                  generated answer
                                                       │
                                                       ▼
                                     compute recall@k, MRR, RAGAS faithfulness,
                                     p95 latency, $/query
                                                       │
                                                       ▼
                                            Grafana dashboards + JSON
                                            results committed to repo
```

## Scope

**In scope.** S&P 100 companies. 10-K + 10-Q + 8-K filings, 2019–present. Finnhub company news. Hybrid retrieval (BM25 + dense). Cross-encoder reranking. Three-layer eval (FinanceBench + custom + RAGAS). OpenTelemetry + Grafana + LangFuse observability. GCP Cloud Run deployment.

**Out of scope (V2+ optional).** DEF 14A, Form 4, S-1. Earnings call transcripts. Agentic tool calls (SQL / function calling over structured data). Fine-tuning of embedding or generation models. Multi-turn conversational eval. Non-English documents.

**V0 — May 25 → Jun 14.** FinanceBench corpus only (~360 PDFs from `github.com/patronus-ai/financebench`). Dense retrieval only. `/query` endpoint. Cloud Run deploy. Baseline recall@5 measurement against FinanceBench.

**V1 — Jun 15 → Jul 12.** Expand corpus to S&P 100 via EDGAR + Finnhub news. Hybrid retrieval. Cross-encoder reranker. Full three-layer eval running. Custom 100 queries built and labeled.

**V2 — Jul 13 → Aug 9.** Time-decay scoring. Table extraction ablation (Llama-parse or `unstructured.io`). Embedding model comparison (3-small vs. 3-large vs. Voyage finance-2 vs. open BGE). Observability dashboards. 1-page technical report.

**Polish — Aug 10 → Sep 7.** README + architecture diagram + demo GIF + writeup. Resume bullets locked. Applications submitted.

## Tech stack

| Component | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | Standard for this stack |
| Service | FastAPI | Async, typed, deployable |
| Vector store | pgvector on Neon free tier | Postgres + HNSW; reads more rigorous than Pinecone on a resume |
| Embeddings (V0) | text-embedding-3-small | Cheap, fast, defensible |
| LLM (V0) | Claude Haiku | Cheap, fast, lane-coherent |
| Reranker | BGE cross-encoder base | Open, self-host, free |
| Eval | FinanceBench (HF) + custom + RAGAS | Public benchmark + domain queries + LLM judge |
| Observability | OpenTelemetry → Grafana Cloud free + LangFuse OSS | Free tiers |
| Deploy | GCP Cloud Run | Free tier, scales to zero |
| News | Finnhub free tier | Ticker-keyed company news, 60 req/min |

## Eval design

Three layers, all run through the same query interface.

**L1 — FinanceBench public benchmark.** 150 questions, each with `evidence` text + gold `answer` + linked PDF. Recall@5 and recall@10 computed by substring/fuzzy match against evidence text. Faithfulness scored against gold answer. Publishable numbers vs. published baselines.

**L2 — Hand-built custom queries (target n=100).** Categorized for ablation reads:
- 30 single-doc factual
- 25 multi-doc reasoning
- 20 freshness-sensitive (answer depends on most recent filing)
- 15 table-reasoning (tests table extraction)
- 10 entity-disambiguation (ticker collisions, name-shared companies)

**L3 — RAGAS (LLM-as-judge).** Faithfulness, answer relevance, context precision, context recall. Manual spot-check of 20 judgments per evaluation run to surface LLM-judge bias; findings disclosed in the writeup.

## User-facing output and demo interface

### API response schema

Every `/query` call returns the same structured shape. The schema is the contract; every visual surface is a rendering of it.

```json
{
  "answer": "Apple's three largest stated risk factors in FY2023 were ...",
  "citations": [
    {
      "doc_name": "APPLE_2023_10K",
      "ticker": "AAPL",
      "filing_type": "10K",
      "filing_date": "2023-11-02",
      "page": 12,
      "section": "Item 1A. Risk Factors",
      "excerpt": "...",
      "retrieval_score": 0.87,
      "rerank_score": 0.94
    }
  ],
  "metrics": {
    "faithfulness": 0.84,
    "latency_ms": 1842,
    "retrieval_ms": 312,
    "rerank_ms": 88,
    "generation_ms": 1442,
    "faithfulness_ms": 612,
    "cost_usd": 0.0042,
    "tokens_in": 4210,
    "tokens_out": 287
  },
  "trace_id": "01HZ4Q...",
  "model": "claude-haiku-4-5"
}
```

**Faithfulness in the response.** Computing RAGAS faithfulness inline adds one judge-LLM call per query (~500–700 ms, ~$0.001). Accepted cost in V0/V1 because the score is shown to the user as eval-discipline signal. V2 may move this to async or cached batch if latency targets tighten.

**Model field.** The `model` field stays in the JSON for traceability and dev-side eval swapping. The user-facing UI does **not** render a model selector — model choice stays a developer/eval-harness concern.

### Demo interface progression

| Phase | Interface | Purpose |
|---|---|---|
| V0 | Streamlit demo + `curl` examples in README. No model selector. Faithfulness badge on every answer. | Self-host on Cloud Run, link from resume |
| V1 | Polished Streamlit with query history, source highlighting, latency + faithfulness badges. Model swap behind a dev flag only. | Recruiter video walkthrough |
| V2 (optional) | Next.js frontend with query playground. Same model-hidden behavior. | Only if V0/V1 ship clean and there's budget. Not a goal in itself. |

Streamlit at V0/V1 because the project's value is the eval rigor and engineering underneath, not the frontend. A clean Streamlit + a tight README GIF is enough portfolio signal.

### What a user sees (V0)

- Header: project name only. No model selector — the model in use is a developer concern, not a UI affordance.
- Query input with placeholder: "Ask about a 10-K — e.g., 'What were Apple's biggest risk factors in FY2023?'"
- Answer card: generated response with inline citation markers `[1]`, `[2]`, etc., and a **RAGAS faithfulness badge** (e.g., "Faithfulness 0.84") rendered alongside the answer. The badge tells anyone hitting the demo that the system grades itself in real time — the eval discipline made visible.
- Sources panel: 5 retrieved chunks. Chunks **cited** in the answer carry colored citation badges `[1] [2] [3]`. Chunks **retrieved but not cited** carry neutral badges `4` `5` — same shape, different color. This distinction makes the user/recruiter aware of what was in context vs. what the LLM actually drew from, without overclaiming.
- Metrics row at the bottom: latency (with retrieval / rerank / generation / faithfulness breakdown), cost, tokens in/out, chunks retained after rerank.

### Eval dashboard (separate view, developer-facing)

A Grafana dashboard tracking, per evaluation run:
- Recall@5 over time
- RAGAS faithfulness over time
- p50 / p95 / p99 latency
- Cost per query
- Per-category breakdown (factual / multi-doc / freshness / table / entity-disambiguation)

Screenshots from this dashboard go in the technical writeup, not the user demo.

### Repo artifacts shipped alongside the code

- Architecture SVG (this doc's diagram, drawn cleanly).
- 30-second demo GIF of one query → answer → sources.
- Eval results JSON, timestamped per run, committed to repo.
- `make demo` launches Streamlit locally; `make eval` reproduces headline numbers.

## Known failure modes — surface, don't hide

- **Embedding similarity on numbers.** Cosine similarity treats "revenue grew 5%" and "revenue grew 25%" as near-identical. A real failure case is presented in the writeup as a known limitation of dense retrieval over financial text.
- **Table extraction loss.** Naive HTML/PDF parsing destroys 10-K tables. Tables-on vs. tables-off is an explicit V2 ablation.
- **LLM-judge bias.** RAGAS faithfulness is itself LLM-judged. Spot-check 20 of its judgments per run; report agreement rate honestly.
- **FinanceBench license.** CC-BY-NC-4.0. Project is non-commercial portfolio work; license requirements met.

## Reproducibility commitments

- Pinned dependency versions in `pyproject.toml`.
- Fixed random seeds for any sampling step.
- Eval results committed to repo as JSON, timestamped per run.
- Architecture diagram committed as SVG alongside this doc.
- One-command rerun: `make eval` reproduces the headline numbers from any clean clone.

## Open decisions

None. Scope, eval design, and visual surface all locked May 21, 2026.

Decisions made and closed in v1.0:
- Universe: S&P 100.
- Filing types: 10-K, 10-Q, 8-K. 2019–present.
- Eval primary: FinanceBench (150 rows). Secondary: 100 hand-built. Judge: RAGAS.
- Stack: pgvector on Neon, text-embedding-3-small, Claude Haiku, BGE cross-encoder reranker, FastAPI on Cloud Run.
- Observability: OpenTelemetry → Grafana Cloud, LangFuse OSS.
- Cited vs. retrieved distinction: yes, separate badge styles in the Sources panel.
- Faithfulness on user-facing output: yes, inline RAGAS score per query.
- Model selector in UI: no. Model swap is a dev-flag concern only.

Deviations require an explicit design-doc amendment with date and rationale.

## Amendments

### 2026-06-03 — Eval primary match mode: fuzzy, not substring (clarification)

The eval-design section says recall@k is computed "by substring/fuzzy match
against evidence text" without committing to one. First live retrieval over the
ingested corpus (84 docs, 25,992 chunks) forces the choice. On a seeded
10-question sample, retrieval found the correct **document** 10/10, but
evidence-span matching diverged sharply by mode:

- **substring recall@10: 2/10**
- **fuzzy (token-overlap ≥ 0.5) recall@10: 7/10**

Cause: FinanceBench gold `evidence_text` spans are large multi-line financial
tables, and pypdf re-extracts that text with different whitespace/ordering than
the dataset's own extraction. An exact contiguous substring of one 512-token
chunk almost never survives, so substring **understates** recall — it measures
text-extraction agreement, not retrieval quality.

**Decision:** report **fuzzy(0.5) as the primary recall metric**, with substring
published alongside as a strict lower bound. This is a measurement-honesty fix
(rule 2: never cherry-pick numbers), not a scope change — both modes already
exist in `eval/metrics.py`; this only fixes which one is the headline.

The 3 fuzzy misses were numerical-reasoning questions (computed across table
cells) — the known dense-retrieval-on-numbers failure mode (rule 5). Reported in
the eval, not hidden.

### 2026-06-04 — Haiku pricing confirmed; cost no longer an estimate

The `PRICING` placeholder (`0.0`, `cost_is_estimate: true`) is resolved.
Confirmed against Anthropic's pricing page: **Claude Haiku 4.5 = $1.00 / 1M
input tokens, $5.00 / 1M output tokens**. Set as the per-token rate in
`generate/answer.py`. `cost_is_estimate` is now *derived* (`model not in
PRICING`) rather than hardcoded, so a priced model reports `false` and any
future unpriced model auto-flags `true`. Measured V0 cost: **$0.0063/query**
(under the <$0.01 floor).

### 2026-06-04 — Faithfulness badge: lightweight judge, not RAGAS (deviation)

The locked design names **RAGAS** for faithfulness. RAGAS is not viable in the
current environment: `ragas` is built for LangChain 0.x and does not import
under LangChain 1.x (the June 2026 ecosystem) — it imports module paths
(`ChatVertexAI`) that no longer exist, and pinning LangChain back to 0.x breaks
`langchain-openai`/`langgraph` and risks the `openai`/`anthropic` deps the
pipeline relies on. A working RAGAS would require a fragile, conflict-ridden
dependency tower, violating the reproducibility rule (rule 4).

**Decision:** implement faithfulness with a **self-contained Haiku judge**
(`generate/faithfulness.py`) using RAGAS's *definition* — the fraction of the
answer's claims supported by the retrieved sources — via one judge call
(temperature 0). Same user-facing badge and committed metric, zero added
dependencies, reproducible. A grounded refusal scores 1.0 (asserts nothing
unsupported). Verified discriminating: grounded claim → 1.0, hallucinated
answer → 0.0. Measured V0 mean faithfulness: **0.941** (above the 0.65 floor).

The V0 faithfulness badge is now **on** (`eval.faithfulness: true`), resolving
the open item from the 2026-05-29 session summary in favour of option A (wire it
in V0) — by a different mechanism than the doc named, for the reason above.

### 2026-06-04 — Known V0 limitations (on record, not bugs)

Surfaced during stress-testing; documented rather than over-engineered in V0:

- **Single shared DB connection serializes concurrent queries.** Safe (psycopg
  locks; verified 6/6 concurrent), but a throughput ceiling. A connection pool
  is the V1 scaling fix. The engine now reconnects once on a dropped connection.
- **Retrieval score is `1 - cosine_distance`**, theoretically in [-1, 1]. A
  negative score is possible (didn't occur across the corpus) and would read
  oddly in the UI. Left honest in V0; clamp-for-display is a V1 option.
- **Setup footgun:** `requirements.lock` pins the project itself, so
  `pip install -r requirements.lock` de-edits the editable install. Re-run
  `pip install -e .` after restoring the lockfile. Noted in the README setup.

### 2026-06-06 — V1 sequencing: separate measured increments (deviation)

The locked V1 scope (above) bundles corpus expansion (S&P 100 via EDGAR) +
hybrid retrieval + cross-encoder reranker + 100 custom queries as one phase.

**Amendment:** V1 is executed as separate, individually-measured increments on
the *same* FinanceBench corpus, in order: (1) hybrid retrieval, (2) reranker,
(3) corpus expansion, (4) full three-layer eval. **Rationale:** changing the
corpus and the retrieval method at once makes the recall delta un-attributable —
more documents add distractors, shifting recall independent of the algorithm.
Isolating each change against the committed V0 baseline (recall@5 0.44) is what
rule #6 (ablation-friendly, one variable at a time) requires. Same end state as
the locked scope; the change is ordering + measurement discipline, not content.
Detail in `v1-plan.md`.

Also recorded there: `pg_search` (ParadeDB BM25) is **deprecated on Neon** and
cannot be enabled, so V1 hybrid uses core Postgres FTS (`tsvector` +
`ts_rank_cd`) for the lexical half. Verified against the live DB.

### 2026-06-26 — Hybrid retired; table extraction promoted ahead of the reranker (deviation)

The 2026-06-06 amendment sequenced V1 as (1) hybrid → (2) reranker → (3) corpus →
(4) full eval, and the locked scope parks table extraction in V2. The V1.1 result
forces a change to both.

**Evidence (committed; retrieval-only A/B + fusion-weight sweep, 150 q, fuzzy, 0
errors):**
- Hybrid (dense + Postgres FTS, RRF) **regressed**: recall@5 0.44 → 0.347; every
  category fell, including the metrics-generated (tables) category it targeted
  (0.32 → 0.26).
- Fusion-weight sweep: **no blend beats dense.** Lexical-only recall@5 **0.04**,
  tables@5 **0.00**; recall rises monotonically with dense weight and only *ties*
  dense at ≈0.95–1.0.
- Dense reproduced the committed 0.44 baseline exactly in the same harness, so the
  comparison is sound, not a measurement artifact.
- JSONs: `eval_results/financebench_20260615T210625Z.json` (dense),
  `...212005Z.json` (hybrid), `ablation_fusion_20260615T214346Z.json` (sweep).

**Finding:** lexical's **0.00 on tables** shows the exact line-item terms are not
in the chunk text to match — `pypdf` flattens 10-K tables at parse time. No
retriever (dense or lexical) can surface evidence that parsing already destroyed.
The tables gap is a **parsing problem, not a retrieval-method problem.**

**Amendment:**
1. **Hybrid retrieval is retired as a recall lever.** Dense stays the default; the
   code + `dense_weight` knob remain (favouring dense) for reproducibility and as a
   documented negative result, not as the shipping path.
2. **Table extraction is promoted from V2 to the next V1 increment**, ahead of the
   cross-encoder reranker. New order: **(1) hybrid — done/retired; (2) table
   extraction (unstructured.io / llama-parse), measured against the 0.44 baseline;
   (3) reranker; (4) corpus expansion; (5) full three-layer eval.**
3. The reranker stays in scope but is **reframed**: it only re-orders candidates
   retrieval already surfaced, so it cannot recover destroyed table evidence —
   expect a prose-side precision (MRR) gain, not a tables fix.

**Rationale:** rule #5 (eval on real + edge cases; validate where it breaks) and
rule #6 (one variable at a time) say to attack the *measured* bottleneck next —
and the measured bottleneck is upstream parsing, not retrieval fusion or reranking.
Same V2 end state (table extraction was always in scope); the change is *ordering*
— pulling it forward — plus retiring hybrid. Full reasoning: `docs/depth-round.md`
and `v1-plan.md`.

---

*Living document. Versioned in repo. Updates noted at top with date and rationale.*
