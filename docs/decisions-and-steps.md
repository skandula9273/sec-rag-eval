# Decisions & steps — how this was built

A high-level narrative of the path: the key decisions, *why* each was made, and the
concrete steps taken. The detailed evidence (numbers, JSONs) lives in
`depth-round.md`; the locked scope + dated amendments live in `design-doc.md`.

---

## The operating principles (the spine)

Every step obeyed the same contract (`CLAUDE.md`):

1. **One variable at a time**, measured against a committed baseline.
2. **Honest numbers** — report regressions and below-floor results as-is.
3. **No fake APIs** — confirm a library/endpoint works before building on it.
4. **Reproducible** — fixed seed (13), pinned lockfile, temp 0, committed eval JSONs.
5. **Scope changes need a dated design-doc amendment**, not silent drift.
6. **Sole-author git history** — every commit attributed to the owner.

These turned a sequence of experiments into an attributable story.

---

## Phase 1 — Establish an honest baseline (V0)

- **Decision:** dense retrieval first (simplest method), scored on a public
  benchmark (FinanceBench), not a hand-picked split.
- **Steps:** scaffold the shared `QueryEngine` (one path for API *and* eval) →
  ingest 84 filings into pgvector → measure. recall@5 **0.44** — below the 0.55
  floor, reported honestly.
- **Key sub-decisions:** fuzzy(0.5) as the primary recall metric (substring
  under-measures because pypdf re-extracts tables differently); confirmed Haiku
  pricing; a self-contained faithfulness judge instead of RAGAS (which doesn't
  import under LangChain 1.x). Each is a dated amendment.

## Phase 2 — Find the lever (the ablation program)

The V0 per-category split (prose 0.66 vs **tables 0.32**) framed the hypothesis.
The discipline: change one thing, measure, keep or reject.

- **Hybrid (dense + lexical):** tested → **regressed**. A fusion-weight sweep showed
  no blend beats dense; lexical alone is 0.04. → *Not a fusion problem.*
- **Diagnostic before building:** two cheap probes (does table evidence survive
  parsing? where does it rank?) showed the evidence **survives** pypdf and is in the
  candidate set or missing entirely — a **ranking + recall** problem, **not
  parsing**. This *reversed* a hypothesis I'd just committed — caught for ~5 min of
  CPU instead of a wasted re-ingest.
- **Reranker (BGE):** tested → **regressed** (over both 3-small and 3-large). A
  general cross-encoder demotes good hits.
- **Chunk size 256:** **worse**. **HNSW vs exact:** no difference.
- **Embedding model → 3-large:** the **win** — recall@5 0.44 → 0.57, tables 0.32 →
  0.62. Five negatives had localized the bottleneck to the *representation*; the
  model swap delivered.
- **Larger chunks (1024):** further lift → 0.64 / tables 0.72 (prose stays flat,
  proving it's real signal, not metric inflation).

**Lesson made explicit:** a diagnostic that proves an *opportunity* doesn't prove a
*tool* captures it — measure the tool.

## Phase 3 — Make the win deployable, for free

- **Constraint hit:** 3-large is 3072-d (2× storage); the corpus already maxed
  Neon's free tier (468/512 MB).
- **Decision:** test **Matryoshka truncation** — 3-large @ **1536-d** kept the full
  recall (== 3072-d) at the *same* storage and schema. The "needs a paid DB"
  constraint dissolved under one more cheap measurement.
- **Steps:** new `configs/v2.yaml` (3-large@1536 + 1024 chunks) → re-ingest
  (recovering from a DiskFull and a network drop, which produced two hardenings:
  TRUNCATE-before-swap and `--resume`) → redeploy → new committed baseline **0.64**.
- **Latency:** measured the breakdown (generation 52%, judge 29%, retrieval 19%) →
  honest finding that e2e <2.5 s is generation-bound → moved the judge off the
  request path → added **streaming** (TTFT ~3.4 s).

## Phase 4 — The product surface

- **Decision (with the owner):** keep the benchmarked backend, add a static
  frontend matching a chosen design, on **GitHub Pages**. Streamlit can't do that
  design and can't run on Pages → a vanilla static app calling the deployed API
  via the streaming endpoint.
- **Hard-won lesson:** caching. Unversioned assets meant browsers served stale
  code; fixed with `?v=` cache-busting + a visible build marker (debugging made
  *observable* instead of guessed).

## Phase 5 — Live EDGAR (any company, newest filings)

- **Decision (with the owner):** *add* a live path alongside the benchmark (don't
  discard the rigor). Architecture flip: static pre-indexed corpus → **fetch-and-
  index-on-demand** (in-memory exact-cosine index, so no Neon storage cap).
- **Steps (rule #3 throughout — verified EDGAR live first):** EDGAR client
  (ticker→CIK, submissions, document fetch) → on-demand `LiveEngine` (fetch → parse
  → chunk → embed → retrieve → stream) → API endpoint + frontend ticker field.
  Caught a real gap by testing the *deployed* API, not just local: `bs4` wasn't a
  declared dependency.
- **Then, the frontier:** dedicated table extractor; persistent Neon cache (fast
  cold starts); BYOK (own keys); all filing types + auto-detection; multi-filing
  compare (2–5 periods); section-labeled citations; a rate-limit guard for safe
  public sharing; optional pre-warm.

---

## The throughline

The same loop, end to end: **form a hypothesis → change one variable → measure
honestly → keep or kill → write down why.** It produced a measured ablation table
(the portfolio centerpiece), a deployable win at zero infra cost, and a live tool
that scales the same RAG to any public company — with every decision on the record.
