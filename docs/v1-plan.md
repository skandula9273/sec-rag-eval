# V1 plan — the recall lift

**Status:** planning. V0 is complete and deployed (recall@5 0.44, faithfulness
0.94). V1's job is to lift retrieval recall toward the 0.75 V2 target, and —
more importantly — to *prove which design choice moves which metric*, one
variable at a time.

**Baseline to beat (committed):**
`eval_results/financebench_20260605T020304Z.json`
- recall@5 **0.44**, recall@10 0.54, MRR 0.317
- per-category recall@5: novel/prose **0.66**, metrics-generated **0.32**,
  domain-relevant **0.34**

The V0 diagnosis *is* the V1 hypothesis: dense retrieval fails on financial
**tables/numbers** (0.32) far worse than on prose (0.66), because (a) cosine
similarity is semantic, not numeric, and (b) pypdf flattens tables. Lexical
(keyword) retrieval should recover exact terms ("FY2022 net sales", line-item
names) that dense embeddings blur.

## Sequencing — one variable at a time (deviation from the locked spec)

The locked design doc (`design-doc.md`) bundles V1 as: corpus expansion (S&P 100
via EDGAR) + hybrid retrieval + reranker + 100 custom queries, all together.

**Proposed change (needs a dated amendment):** do them as *separate, measured
increments on the same FinanceBench corpus*, not all at once. Rationale: changing
the corpus AND the retrieval method simultaneously makes the recall delta
un-attributable — more documents means more distractors, which shifts recall
independent of the algorithm. Rule #6 (ablation-friendly; one variable at a time)
requires isolating each change against the committed baseline.

Order:
1. **V1.1 — Hybrid retrieval** (BM25 + dense), same corpus. Clean A/B vs 0.44.
2. **V1.2 — Cross-encoder reranker** over the hybrid candidates, same corpus.
3. **V1.3 — Corpus expansion** (EDGAR S&P 100) + 100 custom labeled queries.
4. **V1.4 — Full three-layer eval** running on the expanded set.

Each step: one config, one `make eval` run, one committed JSON, one comparison.

## V1.1 — Hybrid retrieval (the headline)

**Goal:** combine lexical (BM25) and dense (cosine) retrieval so exact financial
terms are caught. Expected to move the metrics-generated category most.

**Lexical backend — DECIDED: core Postgres FTS (`tsvector` + `ts_rank_cd`).**

Verified against the live DB before building (rule #1):
- `pg_search` (ParadeDB BM25) appears in `pg_available_extensions` but **Neon has
  deprecated it** — `CREATE EXTENSION pg_search` fails with
  "deprecated and no longer allowed". So it is NOT usable. (Caught in seconds by
  trying to enable it rather than assuming.)
- Core Postgres FTS works and is always available. `ts_rank_cd` is not true Okapi
  BM25 but is a solid term-frequency lexical signal — defensible, zero extra
  extensions, stays in one database.

**Query construction (this was non-obvious — verified empirically):**
- Do NOT pass the raw question to `plainto_tsquery`/`websearch_to_tsquery`: both
  AND the terms, so a normal question ("FY2022 net sales revenue") returns **0
  hits** because the corpus rarely contains every literal token.
- Recipe that works: run the question through `plainto_tsquery` (it drops
  stopwords + stems and keeps meaningful short tokens like "3m"), take its text
  form `'3m' & 'total' & 'revenu'`, and convert `&` → `|` for OR-ranking. Then
  rank with `ts_rank_cd`. Without stopword stripping, junk terms ("what", "was")
  pull the wrong companies to the top; with it, the right company ranks first
  (verified: a 3M question returns 3M docs at ranks 1–2).
- Index: GIN on `to_tsvector('english', content)` (standard, supported on Neon).

**Fusion method:** Reciprocal Rank Fusion (RRF) — `score = Σ 1/(k + rank_i)`
across the dense and lexical ranked lists (k≈60). RRF is rank-based, so it needs
no score normalization between cosine and BM25 (which are on different scales) —
that's why it's the standard hybrid-fusion choice. Alternative: weighted
normalized score blend (an ablation knob).

**New code (scope-isolated, mirrors how dense.py was built):**
- `retrieve/lexical.py` — BM25/FTS top-k over `chunks.content`.
- `retrieve/hybrid.py` — run dense + lexical, fuse with RRF, return top-k.
- `db/schema.sql` — add the lexical index (BM25 index or a `tsvector` GIN index).
- `configs/v1.yaml` — `retrieval.method: hybrid`, fusion params, weights. v0.yaml
  stays untouched so the baseline remains reproducible.
- `pipeline.py` — select retriever by `cfg.retrieval.method` (dense | hybrid).
- tests — RRF fusion is pure logic; unit-test it like the metrics.

**Success criterion (honest, pre-registered):** recall@5 up vs 0.44, AND the
**metrics-generated category** up vs 0.32 specifically. If overall recall rises
but the table category doesn't, the lift came from somewhere else and the
diagnosis was wrong — report that honestly.

## V1.2 — Cross-encoder reranker

**Goal:** re-score the top-N hybrid candidates with a model that reads
query+chunk *together* (cross-encoder), vs the bi-encoder that embeds them
separately. Lifts precision near the top (MRR, recall@5).

- Model: BGE cross-encoder base (open, self-host, free) per the design doc.
- `retrieve/rerank.py`: take top-N (e.g. 20) from hybrid, rerank, return top-k.
- Adds the `rerank_score` field that already exists (Optional) in the schema and
  the `rerank_ms` metric slot — the contract was built for this.
- Ablation: rerank on/off, N candidates, as config knobs.

## V1.3 — Corpus expansion

- EDGAR EFTS API → fetch S&P 100 10-K/10-Q/8-K → same parse/chunk/embed/load.
- Finnhub news (optional, freshness).
- 100 hand-built custom queries, categorized (factual / multi-doc / freshness /
  table / entity-disambiguation) per the design doc.
- Re-run all eval layers on the expanded corpus.

## V1.4 — Full three-layer eval

- L1 FinanceBench (have it), L2 custom 100 (build it), L3 RAGAS/judge (have the
  lightweight judge; revisit real RAGAS if the LangChain ecosystem allows).
- Per-category dashboards; the ablation table is the deliverable.

## What stays out of V1 (still V2)

Time-decay scoring, table-extraction ablation, embedding-model comparison,
observability dashboards. Don't pull these forward.

## First concrete step

Build **V1.1 hybrid retrieval** behind `configs/v1.yaml`, leaving v0 untouched,
then `make eval CONFIG=configs/v1.yaml` and diff the JSON against the 0.44
baseline — especially the metrics-generated category. That single comparison
tells us whether the diagnosis holds.
