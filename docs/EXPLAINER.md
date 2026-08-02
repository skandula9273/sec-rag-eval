# Explainer — the one-page version (rehearse from this)

A cheat-sheet for explaining this project out loud: the pitch, the numbers to have
at your fingertips, the story beats, and the answers to the questions people ask.
Everything here is defensible from the other docs — this is the *compression*, not a
new claim. Full detail: `README.md` (pitch), `PROJECT-LOG.md` (narrative),
`depth-round.md` (defense + commit-by-commit log).

---

## The 20-second version

> A retrieval-augmented QA system over SEC filings — but the **point is the
> evaluation rigor, not the chatbot**. I found the one retrieval lever that worked by
> measuring five that didn't, took recall@5 from **0.44 to 0.64**, deployed it as a
> live tool over any US public company, and — the part that matters — I kept catching
> and publishing my *own* over-claims.

## The 60-second version

> It's RAG over SEC filings (10-K/10-Q/8-K). I started with an honest dense-retrieval
> baseline: recall@5 **0.44**, with a glaring weakness on tables and numbers at
> **0.32**. Then a disciplined ablation hunt — **one variable at a time** against that
> baseline. Hybrid retrieval, a cross-encoder reranker, table extraction, smaller
> chunks: **five levers I measured and rejected**. The one that worked was the
> **embedding model** — swapping to `text-embedding-3-large` took recall to **0.64**
> and tables to **0.70**. I productionized that exact config and built a live EDGAR
> product on top that answers about any of ~10,400 companies from their newest filing
> on demand, streamed with citations. The part I'm proudest of is the honesty: the
> whole platform brackets recall three ways and checks it against human labels — and
> when a model "validated" my shipped metric at κ 0.67, hand-labeling showed that was
> an **artifact** and dropped it to **0.18**. I shipped the correction.

---

## Numbers to know cold

| Number | What it is |
|---|---|
| **0.44 → 0.64** | recall@5 (fuzzy), V0 → V2 — the headline gain |
| **0.32 → 0.70** | tables/numbers recall@5 — the category that was broken, then fixed |
| **[0.093, 0.64, 0.807]** | the same retrieval under strict / fuzzy / semantic matching — "recall is a **bracket**, not a point" |
| **0.93** | faithfulness (grounding), audited by a second model 19/20 |
| **0.50** | answer *correctness* over 150 (separate from faithfulness — grounded ≠ right) |
| **κ 0.67 → 0.18** | the metric-validity false positive: LLM-labeler "validation" vs. my hand labels |
| **~80% / ~20%** | of the V0→V2 gain that's real system improvement vs. chunk-size metric inflation |
| **~10,400** | companies reachable live via EDGAR's ticker→CIK map |
| **84 filings / 15,192 chunks / 274 MB** | the benchmarked FinanceBench corpus (fits Neon's free tier) |
| **118 tests · seed 13 · temp 0** | reproducibility floor (tests run with no network/keys/DB) |

## The 5 story beats (this *is* the project)

1. **The baseline was honest and below target on purpose.** 0.44 overall, tables
   0.32. That gap is the thesis of everything after — I led with the weakness.
2. **Five negatives localized the bottleneck.** Hybrid, reranker (×2), table
   extraction, smaller chunks — all measured, all rejected. It was methodical
   *elimination* until the data pointed at the embedding representation, not a lucky
   guess. Then the model swap proved it.
3. **A constraint dissolved under one more measurement.** 3-large is 3072-dim (2×
   storage) and the corpus already maxed the free tier — looked like it needed a paid
   DB. Matryoshka truncation to 1536-dim kept the *full* recall at the *same* storage.
   The constraint evaporated.
4. **I caught my own mistakes.** The κ 0.67→0.18 false positive; a "clean clone
   reproduces the headline numbers" claim that was false (the corpus lives in a DB, not
   the repo); a design-doc amendment blaming *parsing* for the tables gap that I
   disproved with a 5-minute diagnostic. Each correction is committed. **This is what
   separates it from a résumé project.**
5. **Deployed == measured.** The API and the eval call the *same* `QueryEngine`, so the
   numbers describe the real deployed system — never a lab-only path.

---

## The questions people ask — and the one-liners

- **"Why was your first attempt below target?"** On purpose — V0 is the baseline to
  beat, not the finish line. The interesting part is *why*: dense retrieval fails on
  numbers because cosine similarity is semantic ("grew 5%" and "grew 25%" embed almost
  identically). That 0.32 drove the whole plan.
- **"How do you know the gain is real, not metric gaming?"** I crossed the confound —
  a full model×chunk×matcher grid plus answer accuracy (which *can't* be inflated by
  chunk size). ~80% of the gain is real, ~20% is chunk-size overlap inflation, and I
  say so.
- **"Isn't an LLM judging faithfulness circular?"** Yes — so a *different* model (Opus)
  audited the Haiku judge on 20 verdicts: 19/20 agree, and the one miss was too *harsh*,
  not lenient. And faithfulness is grounding, not correctness — I score correctness
  separately at 0.50.
- **"What would you do with another month?"** Domain-tuned embeddings (Voyage
  finance-2) for the ~32% of table evidence that's not retrieved at all; score the
  *live* path (not just the benchmark); and observability (per-query traces).
- **"Is it a training project?"** No — it's a *retrieval + evaluation* system. No
  training loop or loss function. The depth is in retrieval design, measurement
  honesty, and the engineering underneath. Saying that is the right answer.

## How to get fluent (do this, don't just read)

1. Read `PROJECT-LOG.md` once end-to-end (~20 min) for the shape.
2. Say the **60-second version** out loud 3–4 times until the beats are yours.
3. Memorize the **first two rows** of the numbers table (0.44→0.64, 0.32→0.70) and the
   **κ 0.67→0.18** story. Everything else you can derive from the beats.
4. For a deep drill, keep `depth-round.md` open — its Q&A section is the interview.
