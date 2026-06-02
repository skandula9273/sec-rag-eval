# Session summary — May 21, 2026
## Flagship SEC Filings RAG + Eval Platform — design phase complete

**Status at end of session:** Design doc v1.0 locked. Build phase starts at Task #3.
**V0 deadline:** June 14, 2026 (~3.5 weeks out).
**Total V0–V2 budget:** under $25 (embedding + LLM API calls; infra on free tiers).

---

## What happened this session

1. **FinanceBench picked as eval primary** after comparing three Hugging Face candidates:
   - `PatronusAI/financebench` — 150 rows, per-row evidence text + linked PDFs, recall@k directly computable. **Picked.**
   - `TheFinAI/flare-finqa` — gated access; numerical reasoning over pre-extracted tables, not retrieval-shaped. Rejected.
   - `MehdiHosseiniMoghadam/ConvFinQA` — third-party repost; conversational complexity orthogonal to single-shot RAG eval. Rejected.

2. **Domain and architecture locked.** Considered sports media, tech/engineering (GitHub OSS), then settled on SEC filings + market news. Considered broadening to hybrid agent/tool-call architecture; deferred to V2 to protect V0 scope.

3. **Design doc v1.0 written and locked** at `OUTPUTS/flagship-sec-rag/design-doc.md`. Covers project pitch, problem statement, success criteria (4 metrics with V0 floor and V2 target), architecture diagram (ingestion / query / eval paths), scope (in / out / V0–V1–V2 phasing), tech stack, three-layer eval, user-facing output, known failure modes, reproducibility commitments.

4. **Visual surface locked** via mockup review. Three final decisions:
   - Cited chunks vs. retrieved-but-not-cited get **separate badge styles** in the Sources panel.
   - RAGAS faithfulness score shown **inline** on every answer (accepted ~500–700 ms + ~$0.001 latency cost as eval-discipline signal).
   - No model selector in the UI; `model` stays in JSON for traceability only.

---

## Scope locked — don't relitigate

- Universe: S&P 100.
- Filing types: 10-K, 10-Q, 8-K. Range: 2019–present.
- Eval: FinanceBench (150 Qs) primary, 100 hand-built custom secondary, RAGAS as judge layer.
- Stack: pgvector on Neon free tier, `text-embedding-3-small`, Claude Haiku, BGE cross-encoder reranker, FastAPI on Cloud Run.
- News: Finnhub free tier.
- Observability: OpenTelemetry → Grafana Cloud free, LangFuse OSS.
- License note: FinanceBench is CC-BY-NC-4.0. Non-commercial portfolio use is fine.

Deviations from these require a design-doc amendment with date and rationale. Not a conversation.

---

## Task list — current state

| # | Task | Status |
|---|---|---|
| 1 | Verify FinanceBench access + download PDFs | Done |
| 2 | Write 1-page flagship design doc | Done |
| 3 | Scaffold flagship repo | **Next** |
| 4 | Stand up Neon Postgres + pgvector | Pending |
| 5 | Ingest FinanceBench PDFs end-to-end | Pending |
| 6 | Build FinanceBench eval loader + baseline recall@k | Pending |
| 7 | Confirm scope decisions or push back | Done |

---

## V0 plan — May 22 → June 14

**V0 success criterion:** a deployed FastAPI service that ingests the ~360 FinanceBench PDFs, runs dense retrieval via pgvector, generates answers with Claude Haiku, and reports a recall@5 number on the FinanceBench 150 questions. Crappy is fine. Deployed and demoable is the bar.

| Week | Dates | Work |
|---|---|---|
| W1 | May 22–24 | Task #3 (scaffold repo) + Task #4 (Neon + pgvector schema). Get the skeleton standing. |
| W2 | May 25–31 | Task #5 (PDF parsing → section-aware chunking → embedding → load to pgvector). End-of-week check: top-5 retrieval on 10 sample FinanceBench questions visibly returns the evidence chunk. |
| W3 | Jun 1–7 | Dense retrieval implementation, `/query` endpoint, Task #6 (FinanceBench eval loader + baseline recall@5/10 numbers). |
| W4 | Jun 8–14 | Deploy to Cloud Run, run the full eval, capture recall@5 baseline, write README + record demo GIF. |

V1 (Jun 15 – Jul 12) and V2 (Jul 13 – Aug 9) phases follow per the design doc.

---

## How to start the next session

1. Open `OUTPUTS/flagship-sec-rag/design-doc.md` and re-skim — focus on the architecture diagram and scope sections (5 min).
2. Open this file to confirm the task list and week plan are still current.
3. Open Cowork. Say: "starting Task #3, scaffold the flagship repo." That's the trigger.
4. Decisions you'll need at hand for Task #3:
   - GitHub repo name (suggestion: `sec-rag-eval` or similar).
   - Public or private (recommended: public — this is the portfolio artifact).
   - Where to host the repo locally (which folder on your machine).

---

## Drift watch — things to avoid in the next session

- Don't add hybrid retrieval at V0 — that's V1 work.
- Don't add agent / tool calls at V0 — that's V2 work, and it might never happen.
- Don't expand the corpus beyond FinanceBench PDFs at V0 — EDGAR ingestion is V1.
- Don't polish the UI before retrieval works.
- Don't read more RAG papers before shipping V0 code.
- Don't reopen the eval-dataset decision. FinanceBench is locked.

If any of these surface, the answer is the same: "after V0 ships and the FinanceBench recall@5 baseline is measured."

---

## Files in this project folder

- `design-doc.md` — the locked v1.0 spec. The reference for every implementation decision.
- `session-summary-2026-05-21.md` — this file. The handoff for the next session.

---

## Daily check-in shape (per career-roadmap session-context)

When you come back:
1. What I did yesterday.
2. What I'm doing today.
3. Blockers, if any.

Claude's job: stress-test progress, call out drift, hold to kill switches, don't replan unless something material changed.
