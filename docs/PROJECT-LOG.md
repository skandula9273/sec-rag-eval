# Project log — everything we built, every decision, and the next level up

An exhaustive record of `sec-filings-rag`: what it is, every phase, every decision
and why, every action taken (62 commits), the mistakes and what they taught, and a
deep-dive on where you go next. This is the "read this and you know the whole
project" document. Companion docs: `versions.md` (short), `decisions-and-steps.md`
(narrative), `depth-round.md` (interview prep), `design-doc.md` (locked spec +
amendments).

---

## 0. What this project is

A retrieval-augmented QA platform over US SEC filings, built two ways that share
one engine:

1. **A benchmarked engine** — dense retrieval over the FinanceBench corpus, scored
   by an eval harness (recall@k, MRR, faithfulness, latency, cost). The point was
   *proving which design choice moves which metric, one variable at a time.*
2. **A live product** — a static web app that answers about **any** public company
   from its **newest** EDGAR filings (10-K/10-Q/8-K), fetched and indexed on the
   fly, streamed with citations.

**End state:** recall@5 0.44 → **0.64** (tables 0.32 → **0.70**), faithfulness
**0.93**, live at a cache-proof URL, ~10,400 companies reachable, BYOK, 70 tests,
62 commits, sole-authored.

**Operating contract (`CLAUDE.md`), obeyed throughout:** one variable at a time;
honest numbers even when they hurt; no fake APIs (verify before building); fixed
seed 13 + pinned lockfile + temp 0; scope changes require a dated design-doc
amendment; sole-author git history.

---

## 1. Phase V0 — an honest baseline (commits 650857e → 6534be8)

**Goal:** the simplest thing that works, measured on a public benchmark.

**Built:** the shared `QueryEngine` (one code path for API *and* eval, so numbers
can't drift from production); ingest (pypdf parse → token chunk → OpenAI embed →
pgvector load); dense retrieval (pgvector `<=>` cosine, HNSW); Claude Haiku
generation with parsed `[n]` citations; a self-contained faithfulness judge;
FastAPI `/health` + `/query`; a Streamlit demo; the FinanceBench eval runner
writing timestamped JSON; Cloud Run deploy (Dockerfiles, `X-API-Key` guard).

**Bugs fixed against real data/services (each a commit):** ingest against real
PDFs; sending the query vector as a true `pgvector` type (not a list); a
**falsy-zero bug dropping page-0 evidence** (`if page:` discards page 0);
`/query` 500 after idle (autocommit connection to dodge Neon's idle-in-transaction
kill); `top_k` bounds at the API boundary; per-question eval resilience.

**Key decisions (each a dated design-doc amendment):**
- **Fuzzy(0.5) as the primary recall metric**, substring as a strict lower bound.
  First live eval showed substring 2/10 vs fuzzy 7/10 — because FinanceBench gold
  spans are big tables and pypdf re-extracts whitespace differently, so exact
  substring measures *text-extraction agreement, not retrieval*. Honesty fix.
- **Faithfulness via a self-contained Haiku judge**, not RAGAS (which doesn't
  import under LangChain 1.x). Same definition, zero fragile deps.
- **Confirmed Haiku pricing**; `cost_is_estimate` derived (`model not in PRICING`),
  so an unpriced model auto-flags itself instead of reporting a fake number.

**V0 result (committed, 150 q):** recall@5 **0.44** (below the 0.55 floor —
reported as-is), faithfulness **0.94**, cost **$0.0063**. Per-category: prose
**0.66** vs tables/numbers **0.32**. *That gap is the thesis of everything after.*

---

## 2. Phase V1 — the ablation program (commits 8e7c661 → 63a6f54)

The discipline: form a hypothesis, change one variable, measure honestly against
the committed 0.44 baseline, keep or kill, write down why. This is the substance.

**A key methodology decision (amendment):** execute V1 as *separate, individually-
measured increments on the same corpus*, not a bundle — changing corpus and method
at once makes the delta un-attributable.

**A tooling win that unblocked everything:** a **retrieval-only eval mode**
(`--no-generate`). recall@k/MRR are pure retrieval metrics (query embed + DB, no
Anthropic), so the ablations run for pennies and a billing outage can't masquerade
as a result. Dense reproduced 0.44 *exactly* in this mode, validating the harness.

**The interventions (all measured):**

| # | Lever | Result | Verdict |
|---|---|---|---|
| 1 | **Hybrid** (dense + Postgres FTS, RRF) | recall@5 0.44 → 0.35; fusion-weight sweep: no blend beats dense, lexical-only 0.04 | ✗ retired |
| 2 | **Table-extraction diagnostic** | gold table evidence *survives* pypdf (8/8); pdfplumber no better | reframed the problem |
| 3 | **Retrieval-depth diagnostic** | table evidence: 32% at rank 1–5, 26% at 6–20, **32% miss top-100** | ranking + recall, not parsing |
| 4 | **Reranker** (BGE cross-encoder) | 0.44 → 0.39; demotes good hits | ✗ retired |
| 5 | **Chunk size 256** | 0.28 (worse) | ✗ |
| 6 | **HNSW vs exact search** | identical | — not a lever |
| 7 | **Embedding model → 3-large** | 0.44 → **0.57**, tables 0.32 → **0.62** | ✅ **the lever** |
| 8 | **Larger chunks (1024)** over 3-large | → **0.64**, tables **0.72** (prose flat → real, not metric inflation) | ✅ |
| 9 | **Reranker over 3-large** | hurts *worse* (0.57 → 0.38) | ✗ confirmed retired |
| 10 | **Dimension truncation** (Matryoshka) | 3-large@**1536** == @3072 on recall | ✅ free-tier fit |

**The turning points:**
- **A reversed hypothesis, caught cheap.** I committed a design-doc amendment
  blaming *parsing* for the tables gap, then two ~5-minute diagnostics disproved my
  own hypothesis (evidence survives parsing; it's a ranking/recall problem) — and I
  wrote a *correcting* amendment. Cheap tests before an expensive re-ingest.
- **Five negatives localized the bottleneck to the representation**, and the
  embedding-model swap delivered. "I changed one variable at a time until the data
  pointed at the embedding, then proved it" — a methodical result, not a guess.
- **A dissolved constraint.** 3-large is 3072-d (2× storage) and the corpus already
  maxed Neon's free tier — looked like it needed a paid DB. One more cheap test
  (Matryoshka truncation to 1536-d) kept the full recall at the same storage. The
  constraint evaporated under measurement.

**Hardenings born from failures:** `embed.py` retry with backoff, then **fail-fast
on `insufficient_quota`** (billing ≠ transient); the eval runner treating infra
errors as fatal; `load.py --resume`; **TRUNCATE-before-swap** (a full re-ingest on
a near-full Neon blew the cap via dead tuples).

---

## 3. Phase V2 — productionize the win (commits dc9313d → cfd59db)

**Decision:** `configs/v2.yaml` = dense + **text-embedding-3-large @1536-d** +
**1024-token chunks**. Re-ingested (15,192 chunks / 274 MB — fits the free tier),
redeployed, `SEC_RAG_CONFIG` documented (query and corpus embed model must match).

**New committed baseline (full pipeline, 150 q, 0 errors):** recall@5 **0.64**,
recall@10 **0.74**, tables **0.70**, faithfulness **0.93**, cost ~$0.005–0.009.
Deployed == measured (same engine).

**Latency:** measured the breakdown (generation 52%, judge 29%, retrieval 19%) →
honest finding that e2e <2.5 s is *generation-bound* (target mis-specified) → moved
the faithfulness judge **off the request path** → added **streaming** (SSE), TTFT
~3.4 s. Reframed the latency target as TTFT/retrieval-latency.

---

## 4. Phase — the product surface (commits f980fbf → b0c32c3)

**Decision (with you):** keep the benchmarked backend, add a **static frontend**
matching a chosen design, on GitHub Pages, calling the deployed API via streaming.
(Streamlit can't do that design and can't run on Pages.)

Built: a vanilla HTML/CSS/JS app; iterated the theme (dark → **light turquoise**
per your call); the API-key UX (added → removed → BYOK); a visible build marker;
`?v=` cache-busting. **Lesson:** unversioned assets served stale code — fixed with
cache-busting + a build marker (debugging made *observable*, not guessed).

---

## 5. Phase — live EDGAR (commits f3778cd → 365f657)

**Decision (with you):** *add* a live path alongside the benchmark (don't discard
the rigor). Architecture flip: static pre-indexed corpus → **fetch-and-index-on-
demand** (in-memory exact-cosine index → no storage cap; Neon cache for cold
starts).

**Built (verifying EDGAR live first, rule #3):**
- **EDGAR client** — ticker→CIK (company map), submissions API, document fetch,
  HTML→text. Caught a real gap by testing the *deployed* API: `bs4` wasn't a
  declared dependency (worked locally, 500'd in prod).
- **On-demand `LiveEngine`** — fetch → parse → chunk → embed → retrieve → stream;
  same event shape as the benchmarked path.
- **Frontier build-out:** dedicated **table extractor** (merges split currency
  cells); **persistent Neon cache** (bounded 40, best-effort); **BYOK** (per-request
  keys via headers); **all filing types** + auto-detection; **multi-filing compare**
  (2–5 periods, "vs last year", "5-year trend"); **section-labeled citations**
  ("Item 1A. Risk Factors"); a per-IP **rate limit** + `require-BYOK` flag for safe
  public sharing; optional background **pre-warm** of popular tickers.

---

## 6. Phase — docs, polish, and the caching saga (commits 2324377 → a5948f5)

- **Docs:** `versions.md`, `decisions-and-steps.md`, refreshed README + CLAUDE.md.
- **The bug that looked like caching (the big lesson).** For many rounds, "the old
  page keeps coming up" was treated as a cache/CDN/browser issue — we tried
  hard-refresh, incognito, `?v=` busting, a no-cache Cloud Run URL, even a new
  domain. **It reproduced on someone else's device** (your call), which proved it
  *wasn't* cache. The real cause: a **CSS bug** — `.modal { display: flex }`
  overrode the HTML `hidden` attribute, so the API-keys popup was *permanently open
  on every load, for everyone*. One line fixed it: `[hidden] { display: none
  !important }`. **Lesson: verify the rendered artifact, not just the served files;
  a reproduction on a fresh device is the fastest way to separate "our bug" from
  "your cache."**
- **Cache-proof hosting:** GitHub Pages forces a 10-min HTML cache we can't change
  (and had flaky deploys), so we also serve the site from **Cloud Run with
  `no-store` headers** — always fresh, reliable.

---

## 7. What exists now (inventory)

- **Live app** (two hosts): Cloud Run static (cache-proof, primary) + GitHub Pages.
- **API** (Cloud Run): `/health`, `/query`, `/query/stream`, `/query/live/stream`;
  CORS, BYOK headers, rate limit, pre-warm.
- **Engine:** `pipeline.py` (shared), `edgar/` (live), `ingest/` + `retrieve/` +
  `generate/` (benchmarked), `eval/` (harness + ablation scripts).
- **Data:** Neon pgvector (v2 corpus 274 MB + a bounded live cache).
- **Evidence:** 16 committed eval JSONs; the ablation table in `depth-round.md`.
- **Docs:** design-doc (+8 amendments), depth-round, versions, decisions-and-steps,
  this log. **62 commits, 70 tests, sole-authored.**

**Scorecard vs targets:** recall@5 0.64 (goal 0.75) · faithfulness 0.93 ✓ · cost
~borderline · latency reframed to TTFT ~3.4 s.

---

## 8. Deep dive — the next level up (for you)

You've shipped a real system. The honest gap between "impressive portfolio project"
and "the next level" is in four areas. In rough priority:

### A. Make the results *provably* better, not just bigger
- **The recall@5 0.64 → 0.75 gap is unfinished.** The diagnostic said ~32% of table
  evidence is *not in dense top-100 at all* — that's a **representation** problem an
  embedding swap only half-solved. Next levers, measured: **Voyage finance-2 /
  domain-tuned embeddings**, **query rewriting** (turn "net PP&E" into the terms
  filings actually use), and **table-structure-aware retrieval** (embed table rows
  with their headers). Each is a one-variable ablation you already have the harness
  for.
- **Build the missing eval for the *live* path.** Right now FinanceBench scores the
  benchmark corpus, but the live EDGAR answers are unscored. Level-up move: a small
  labeled set of live questions with known answers (from XBRL facts you can fetch
  programmatically) → an automated correctness score for the live product. *That*
  turns "a cool demo" into "a measured live system."

### B. Show senior-engineer judgment, not just features
- **Observability.** The original design doc scoped OTel → Grafana + LangFuse and it
  was never built. Even lightweight structured logging + a per-query trace
  (retrieval hits, scores, latency breakdown, cost) surfaced in a dashboard is what
  distinguishes "I built a RAG" from "I operate a RAG." This is the single biggest
  signal upgrade for an eng-lead audience.
- **Evaluation as CI.** Wire `make eval` (retrieval-only, cheap) into a GitHub
  Action so every change reports recall deltas automatically. "My eval runs on every
  PR" is a strong, rare signal.
- **Cost/latency SLOs + guardrails** made explicit (per-query budget, a hard cap,
  graceful degradation), not just measured once.

### C. Depth in the domain (financial RAG specifically)
- **Numerical/temporal correctness.** The hardest, most valuable financial-RAG
  problem: getting *numbers* right and *periods* right (FY vs quarter, restated vs
  original). A verifier step (re-check the cited number against the source table) or
  XBRL-grounded answers would be genuinely differentiated.
- **Multi-hop / cross-document reasoning** ("how did margin change after the
  acquisition announced in the 8-K?") — beyond the current single/multi-filing
  retrieval.

### D. The narrative and the audience
- **Write the 1-page technical report / blog post** the design doc always planned.
  `depth-round.md` is the draft; the story ("five measured negatives localized the
  bottleneck to the embedding; here's the ablation table") is more hireable than any
  single number. **This is the highest-ROI hour you can spend** — it makes all the
  work legible.
- **A 30–60s demo GIF** in the README (ask a ticker → streamed cited answer).
- **Resume framing:** lead with the *measurement discipline* and the *live system at
  scale*, not the tech list.

### The one-sentence "next level"
You proved you can **build and measure**; the next level is proving you can
**operate and communicate** — add live-path evaluation + observability (so the
system grades and watches itself), and write the report that makes the rigor
visible. That's the jump from "strong project" to "this person runs production ML."

---

## 9. Honest limitations still on the record

Cold start on a never-seen filing (~15–25 s, inherent to on-demand; cache + pre-warm
mitigate); HTML table parsing good-but-imperfect on complex statements; multi-filing
capped at 5 with simple per-period retrieval; the live path is unscored (see §8A);
GitHub Pages' 10-min HTML cache + flaky deploys (mitigated by the Cloud Run host).
None are hidden — all are documented, which is itself the point of this project.
