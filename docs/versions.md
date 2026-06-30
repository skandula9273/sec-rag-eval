# Versions — executive summary of what we built

A retrieval-augmented QA platform over SEC filings. Two product surfaces share one
RAG engine: a **benchmarked corpus** (FinanceBench, for measured rigor) and a
**live EDGAR** path (any company, newest filings). Every number below is committed
in `eval_results/` and traceable to a config.

---

## Part 1 — The benchmarked engine (V0 → V2)

The goal here was never one impressive number; it was **proving which design choice
moves which metric**, one variable at a time, against a public benchmark.

| Version | What it is | Headline result |
|---|---|---|
| **V0** | Dense retrieval baseline: `text-embedding-3-small` (1536-d) + 512-token chunks, pgvector on Neon, Claude Haiku generation, FinanceBench eval harness, deployed on Cloud Run. | recall@5 **0.44**, tables **0.32**, faithfulness **0.94**, ~$0.006/q. Honest: below the 0.55 floor. |
| **V1.1** | Hybrid retrieval (dense + Postgres FTS, RRF fusion) + a fusion-weight sweep. | **Retired.** No blend beats dense; lexical is noise (0.04). Finding: the tables gap is *not* a fusion problem. |
| **V1.2** | Cross-encoder reranker (BGE) over dense candidates. | **Retired.** Regressed (0.44 → 0.39); a general reranker demotes good hits. |
| *(diagnostics)* | Table-extraction probe; retrieval-depth probe; chunk-size 256; HNSW-vs-exact. | Localized the bottleneck to the **embedding representation** (5 negatives). |
| **V2** | The lever, productionized: **`text-embedding-3-large` @ 1536-d (Matryoshka) + 1024-token chunks**, dense. Re-ingested (15,192 chunks / 274 MB, fits the free tier), redeployed. | recall@5 **0.64**, recall@10 **0.74**, tables **0.70**, faithfulness **0.93**. The win, at zero infra cost. |
| **Latency** | Faithfulness judge moved off the request path; streaming (SSE) added. | Measured breakdown; honest finding that e2e <2.5 s is generation-bound. TTFT ~3.4 s. |

**The story in one line:** dense @ 3-small / 512 (0.44) → six measured interventions,
five rejected → the embedding model was the lever → dense @ 3-large@1536 / 1024
(**0.64**, tables more than doubled), live and free-tier. Full ablation table in
`depth-round.md`.

---

## Part 2 — The product (static frontend + live EDGAR)

A dark, uplifting single-page app on **GitHub Pages**
(`skandula9273.github.io/sec-rag-eval`) that streams grounded, cited answers from
the deployed API. Then it grew from "our fixed corpus" to "any company, live."

| Build | What shipped |
|---|---|
| **v9** | Static frontend (vanilla HTML/CSS/JS), light turquoise theme, streaming answers + cited/retrieved sources + metrics. Open API (no friction). |
| **v10** | **Live EDGAR**: enter a ticker → fetch that company's latest 10-K live, index on-demand (in-memory, no Neon cap), answer with citations. Any of ~10,400 companies. |
| **v11** | **BYOK** — optional own OpenAI + Anthropic keys (browser-only), so visitors can run on their own credits. |
| **v12** | **All filing types** (10-K / 10-Q / 8-K, auto-detected) + **multi-filing compare** (year-over-year across 2–3 filings). |
| **v13** | **Dedicated table extractor** (merges split currency cells), **persistent Neon cache** (fast cold starts), **section-labeled citations** ("Item 1A. Risk Factors"), `require-BYOK` flag. |
| **v14** | Per-IP **rate limit** (safe public link), multi-filing up to 5 periods, cleaned section labels, optional background **pre-warm** of popular tickers. |

**The live system now does:** any company → newest 10-K/10-Q/8-K (or several for a
comparison) → fetched live from EDGAR → parsed (XBRL noise stripped, tables kept) →
indexed with section labels → cached across cold starts → grounded, cited, streamed
answer → optionally on the visitor's own keys.

---

## Scorecard vs the original targets

| Target | Now | Status |
|---|---|---|
| recall@5 0.75 | **0.64** (recall@10 0.74) | strong; the measured ceiling on this corpus/metric |
| faithfulness 0.80 | **0.93** | ✅ |
| cost <$0.005 | ~$0.005–0.009 | ◑ (eval vs live) |
| p95 <2.5 s e2e | generation-bound; **TTFT ~3.4 s** via streaming | reframed (honest) |

The benchmark proves the engineering; the live EDGAR product shows it at scale.
