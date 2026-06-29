# Depth round — things I'd be asked about this project

A running notes file for the Flagship depth-round interview (an interviewer picks
this project and drills it for 45 min: *did you actually do the work, and do you
understand the choices you made*). Also the draft of the eventual blog post.

**Format for every entry:** the choice → alternatives considered → the tradeoff →
what I'd do differently. Update it the moment a decision is made, not after.

**Framing note (say this if they treat it like a training project):** this is a
*retrieval + evaluation* system, not a model-training one. There is no training
loop, loss function, or FLOPs/parameter tradeoff. The depth here is in retrieval
design, measurement honesty, and the engineering underneath. Don't manufacture a
training-curve answer — there isn't one, and saying so is the right answer.

---

## The questions they actually ask — and my answers

### "What was your baseline, and why was your first attempt below it?"
V0 target floor was **recall@5 0.55**; my dense-retrieval baseline came in at
**0.44** on FinanceBench (150 q, fuzzy match). Below floor *on purpose* — V0 is
the baseline to beat, not the finish line. The *why* is the interesting part:
the per-category split shows prose/narrative questions at **0.66** but
**metrics-generated (tables/numbers) at 0.32**. Dense retrieval fails on numbers
for two compounding reasons: (a) cosine similarity is *semantic*, so "revenue
grew 5%" and "grew 25%" embed almost identically; (b) pypdf flattens 10-K tables
into whitespace soup before they're ever embedded. That 0.32 is the thesis of the
entire V1/V2 plan.

### "Tell me about a failure mode you didn't expect and how you debugged it."
First live eval, substring recall@10 was **2/10** — looks catastrophic. Instead of
shipping or hiding it, I ran the *same* questions under a fuzzy (token-overlap ≥
0.5) matcher: **7/10**. The gap meant the metric, not the retriever, was broken.
Root cause: FinanceBench's gold `evidence_text` spans are large multi-line tables;
pypdf re-extracts the same text with different whitespace/ordering than the
dataset's own extraction, so an exact contiguous substring almost never survives a
512-token chunk. Substring was measuring *text-extraction agreement*, not
retrieval quality. Fix: report **fuzzy(0.5) as primary, substring as a strict
lower bound** — a measurement-honesty fix, documented as a dated design-doc
amendment, not a quiet metric swap.

### "What's the most expensive mistake you made, and what did it teach you?"
A hybrid-retrieval eval run aborted ~halfway when the Anthropic credit balance hit
zero: 73/150 scored, 77 errors. The runner had been hardened to be *resilient to
per-question failures*, so it swallowed the billing outage as 77 "question
failures" and still emitted an aggregate **recall@5 0.274** — which read like
hybrid had *regressed* vs the 0.44 baseline. It hadn't; the number was computed
over a non-random partial slice and was meaningless. **Lesson:** in an eval
harness whose entire value is honest numbers, infrastructure failures (billing,
auth) must be *fatal* — never folded into the same resilience path as a genuine
per-question miss. A partial run that still prints a headline metric is worse than
a crash. (Fix: classify infra errors as fatal, or suppress aggregates when
`n_scored << n_questions`.)

### "What's your validation strategy, and why not [alternative]?"
- Public benchmark (**FinanceBench 150**) run through the **exact same
  `QueryEngine` as production** — never a second eval-only path, so the numbers
  describe the deployed system.
- **Per-category recall**, not just overall — overall recall hides the
  tables(0.32)/prose(0.66) gap that *is* the finding. "I understand where my
  system fails" beats "here's my average."
- **Fuzzy primary + substring lower bound** (see above), **fixed seed (13)**,
  **temp 0**, **pinned lockfile**, **timestamped JSON committed per run** — one
  command reproduces the headline numbers from a clean clone.
- **Pre-registered success criteria**: e.g. V1.1 hybrid must lift recall@5 vs 0.44
  *and* the tables category vs 0.32 *specifically* — so a lift "from somewhere
  else" can't be spun as confirming the diagnosis.
- Why not RAGAS for faithfulness? See the faithfulness entry — it doesn't import
  under LangChain 1.x; I used its *definition* via a self-contained judge instead.

### "What would you do with another month?"
In priority order, because the data points here: **(1) real table extraction**
(unstructured.io / llama-parse) — the 0.32 category is the dominant recall lever
and pypdf is the bottleneck; (2) the cross-encoder reranker over hybrid
candidates; (3) latency: connection pool + move the faithfulness judge off the
request path (p95 ~15.6 s vs <2.5 s target); (4) the 100-query custom eval set for
multi-doc / freshness / entity-disambiguation coverage; (5) embedding-model
ablation (3-small vs 3-large vs Voyage finance-2 vs BGE).

---

## Decision log (choice / alternatives / tradeoff / would-do-differently)

### Chunking — 512 tokens, 64 overlap, token-window + section-aware (cl100k_base)
- **Alternatives:** 256 (sharper evidence spans, more fragments, more rows);
  1024 (more context per chunk, dilutes the embedding, coarser citations).
- **Tradeoff:** evidence-span precision vs retrieval recall vs cost. 512 is the
  defensible middle and matches the embedding model's sweet spot.
- **Would-do-differently / open:** chunk size is a config lever but **not yet
  ablated** — I can't yet *prove* 512 beats 256 on this corpus. That's a clean
  ablation I'd run. (Honest gap; don't claim 512 is optimal, only defensible.)

### Embedding model — text-embedding-3-small (1536-dim)
- **Alternatives:** 3-large (3072-dim, better but ~6× cost + 2× storage/index
  size), Voyage finance-2 (domain-tuned), open BGE (self-host, free).
- **Tradeoff:** cost/latency/index-size vs retrieval quality. Chose the cheap,
  fast, defensible baseline — simpler-first (rule #7).
- **Would-do-differently:** V2 ablation comparing all four on the *same* corpus &
  queries. The dim must stay 1536 to match the schema unless I re-index.

### Vector store — pgvector on Neon, HNSW, cosine
- **Alternatives:** Pinecone / Weaviate (managed vector DBs), FAISS (in-process).
- **Tradeoff:** one Postgres holds vectors **+** lexical (tsvector) **+** metadata
  + gives transactional upserts and a free tier — and reads as more rigorous than
  a turnkey vector SaaS. HNSW over IVFFlat: better recall/latency at this scale,
  no training step.
- **Score = 1 − cosine_distance**, theoretically [−1, 1]; a negative score is
  possible (didn't occur) and would read oddly in the UI — left honest in V0.

### Retrieval method — dense (V0) → hybrid: dense + Postgres FTS, RRF fusion (V1.1)
- **Alternatives:** pure dense; true BM25 via ParadeDB `pg_search`; weighted
  normalized score blend instead of RRF.
- **Tradeoffs / non-obvious findings:**
  - `pg_search` (ParadeDB BM25) is **deprecated on Neon** — `CREATE EXTENSION`
    fails. Caught in seconds by *trying it*, not assuming. So lexical = core
    Postgres FTS (`tsvector` + `ts_rank_cd`), which is TF-based, **not** true Okapi
    BM25 — a real, stated limitation.
  - **RRF over weighted blend:** RRF is rank-based, so it needs no score
    normalization between cosine and `ts_rank_cd` (different scales). That's
    *why* it's the standard hybrid-fusion choice.
  - **tsquery construction (the empirical gotcha):** feeding a raw question to
    `plainto_tsquery`/`websearch_to_tsquery` ANDs every token → a normal question
    returns **0 hits** because the corpus rarely contains every literal token. Fix:
    let `plainto_tsquery` drop stopwords + stem, then convert `&` → `|` for
    OR-ranking. Without stopword stripping, junk tokens ("what", "was") surface the
    wrong company; with it, a 3M question ranks 3M docs #1–2.
- **Result (2026-06-15, retrieval-only A/B, 150 q, fuzzy, 0 errors):** hybrid
  **regressed across the board** — overall recall@5 0.44 → **0.3467**, recall@10
  0.54 → 0.44, MRR 0.317 → 0.245 — and the pre-registered test **failed**: the
  metrics-generated (tables) category went **0.32 → 0.26**, the opposite of the
  hypothesis. (Dense reproduced 0.44 exactly in the same harness, so the
  comparison is trustworthy.) **Diagnosis (hypothesis, not yet proven):**
  equal-weight RRF fuses a *noisy* Postgres-FTS lexical list into a stronger dense
  list and drags good dense hits down — tellingly, even the prose category (dense's
  strength, 0.66) fell to 0.54, which a genuinely helpful lexical signal would not
  do. **Next:** measure lexical-only recall (is the signal any good on its own?),
  then try **weighted fusion** (down-weight lexical) and tune `k_rrf` — both already
  ablation knobs. **Do not ship hybrid as the default until it beats dense.** The
  regression is reported, not buried (rule #2).
- **Decision (2026-06-15) — V1.1b fusion-weight ablation:** before judging hybrid,
  sweep the fusion weight in **one efficient pass** — retrieve the dense + lexical
  candidate lists *once* per question, then re-fuse in memory at
  `dense_weight ∈ {0.0 … 1.0}` (fusion is cheap; re-retrieving is not). The sweep
  yields three answers at once: `dense_weight = 0.0` is **lexical-only** (the
  standalone quality of the keyword signal), `1.0` is **dense-only** (must
  reproduce 0.44 as a sanity check), and the middle is **weighted RRF**. The one
  question: does *any* blend beat dense's 0.44, or is pure dense the ceiling on
  this corpus? `dense_weight` is added as a config knob so the winning setting is
  reproducible through the normal runner (rule #6).
- **Result (2026-06-15, V1.1b sweep, 150 q, fuzzy, retrieval-only):** **no blend
  beats dense.** lexical-only (dense_weight 0.0) recall@5 = **0.04**, tables@5 =
  **0.00** — the Postgres-FTS signal is ~noise on this corpus, and *zero* on the
  very category it was meant to fix. Recall rises monotonically with dense_weight
  and only *ties* dense at ≈0.95–1.0 (recall@5 0.44, tables 0.32). **Verdict:
  pure dense is the ceiling; hybrid (dense + Postgres FTS) is retired as a recall
  lever.** Dense stays the default; `dense_weight` stays a knob, set to favour
  dense.
- **The deeper finding (this is the real takeaway):** lexical's **0.00 on tables**
  points *upstream*. The exact line-item terms aren't in the chunk text to match
  because **pypdf flattened the tables at parse time** — so no retriever, dense or
  lexical, can surface evidence that parsing already destroyed. **The tables gap
  is a parsing problem, not a retrieval-method problem.** That's direct,
  evidence-backed support for prioritising **table extraction** (unstructured.io /
  llama-parse) over further retrieval/fusion tuning — the hunch from the
  "feature-complete, adapted by findings" call is now a measured result. This is
  the depth-round answer to "a failure mode you didn't expect": I expected hybrid
  to help tables; instead it proved the bottleneck is elsewhere — see the
  correction below for *where*.
- **Correction (2026-06-26, two cheap diagnostics before any re-ingest):** the
  parsing hypothesis was **wrong**, and the diagnostic-first checkpoint caught it
  before a full re-embed. (a) *Parser comparison* (8 table Qs): gold table evidence
  is recoverable in a 512-tok window under **pypdf 8/8**; pdfplumber does **not**
  help (sometimes worse, 0.95 → 0.91) — table evidence survives parsing. (b)
  *Retrieval depth* (50 metrics-generated Qs, dense top-100): evidence ranks 1–5 =
  32%, **6–20 = +26%** (in the candidate set, below top-5), 21–100 = +10%, and
  **misses top-100 for 32%**. So the tables gap is **two problems, not parsing**: a
  **ranking** problem (26% at rank 6–20 → a cross-encoder reranker can promote
  these) and a **recall** problem (32% absent from top-100 → embedding/chunking
  lever; no reranker reaches these). **Decisions:** table extraction dropped;
  **reranker is the next build** (widen candidates 20 → 50 to reach the 21–50
  band); **embedding-model ablation** (3-large / Voyage) promoted to target the
  deep-miss band; recall@5 0.75 needs *both*. **The depth-round lesson:** I
  committed an amendment on the parsing hypothesis, then disproved my own
  hypothesis with a ~5-min diagnostic instead of a wasted corpus re-embed —
  cheap tests before expensive commitments.

### Reranker (V1.2) — built and measured; BGE base does NOT help here
- **Choice:** BGE cross-encoder base over dense top-50 → top-5; the V1.1b
  diagnostic said 26% of table evidence sat at ranks 6–20, promotable by a joint
  query+chunk scorer. Confirmed it installs + discriminates (smoke test 1.0 vs 0.0).
- **Result (2026-06-27, retrieval-only, 150 q, fuzzy, 0 errors):** **regressed
  overall** — recall@5 0.44 → **0.393**, MRR 0.317 → 0.271. Per category: tables
  **0.32 → 0.34** (+0.02, noise), **domain-relevant 0.34 → 0.20 (−0.14)**, prose
  0.66 → 0.64. recall@10 flat (0.54 → 0.533 — same pool, just reordered).
- **Why:** a general-domain cross-encoder isn't free. It captured almost none of
  the promotable 26% and **demoted good dense hits** — for every chunk promoted it
  pushed another out of top-5. The bi-encoder is already reasonable on this domain;
  a generic reranker trades wins for losses.
- **Decision:** BGE-base rerank-over-dense is not the lever. Kept as a config knob
  (`rerank: off` default) + documented negative result. Next: a cheap
  `candidates=20` confirm (does a smaller pool stop the bleeding?), then pivot to
  the **embedding lever** (3-large / Voyage finance) — the only thing that reaches
  the 32% deep-miss band a reranker never sees.
- **Depth-round lesson:** a diagnostic that proves an *opportunity* (26%
  promotable) does NOT prove a given *tool* can capture it. Measure the tool;
  don't assume the opportunity is yours.
- **Confirmed retired (2026-06-27, retested over 3-large candidates):** with the
  much stronger 3-large bi-encoder, the reranker hurts *worse* — recall@5 0.573 ->
  **0.38**, tables 0.62 -> **0.34**. More good candidates = more for it to demote.
  The BGE-base cross-encoder is wrong for this financial-retrieval task,
  independent of candidate quality. **Definitively retired; the config is dense
  (3-large) with no rerank.**

### Chunk size — smaller is WORSE (256 vs 512); dilution hypothesis rejected
- **Setup:** local in-memory exact-cosine ablation. Neon's free tier is maxed
  (468/512 MB; a second corpus `DiskFull`s), so the experiment runs off a numpy
  index, not the prod store — same model (3-small), only chunk size differs. 512
  reuses prod embeddings; 256 re-chunked + embedded locally.
- **Result (2026-06-27, 150 q, fuzzy, exact search):**
  - **512-exact reproduces the committed 0.44 exactly** (recall@5 0.44, @10 0.54,
    MRR 0.317, tables 0.32) — validates the local harness AND shows Neon's
    approximate HNSW was NOT costing recall.
  - **256 is much worse everywhere:** recall@5 0.44 → **0.28**, @10 0.54 → 0.36,
    MRR → 0.208; tables 0.32 → 0.18, prose 0.66 → 0.44.
- **Why (+ honest caveat):** the dilution hypothesis was backwards. recall@k is
  fuzzy ≥0.5 token-overlap vs the gold evidence, and FinanceBench spans are large
  multi-line tables. A 512-token chunk holds more of the span so it clears the 0.5
  bar more often; 256 splits the span across chunks (none clears it) AND doubles
  the distractors. So recall@k is partly **coupled to chunk size** — bigger chunks
  inflate it. Real signal + a measurement artifact, both pushing the same way.
- **Decision:** 256 rejected; 512 stays. The lever points toward LARGER chunks,
  but that trades against citation precision + generation cost and partly games the
  overlap metric — not a clean win. Next real lever: the embedding MODEL
  (3-large / Voyage), now testable via the same local index (bypasses Neon's cap).
- **Bonus fix this surfaced:** `embed.py` had no rate-limit retry, so any large
  embed (this ablation, or a real corpus ingest) died on a 429. Added bounded
  exponential backoff + tests.

### Embedding model — 3-large is THE lever (tables 0.32 -> 0.62) — first win
- **Setup:** local exact-cosine ablation, same 512 chunk texts, only the model
  differs. 3-small reuses prod embeddings; 3-large (3072-dim) embeds the same
  texts locally.
- **Result (2026-06-27, 150 q, fuzzy):** recall@5 **0.44 -> 0.5733** (+0.13),
  recall@10 0.54 -> 0.667, MRR 0.317 -> 0.403. Per category: **tables 0.32 -> 0.62
  (+0.30, nearly doubled)**, domain 0.34 -> 0.44, prose 0.66 -> 0.66 (flat, already
  adequate). Crosses the V0 recall@5 floor (0.55) that 3-small missed.
- **This validates the whole hunt:** five negatives (hybrid, table-extraction,
  reranker, chunk-size, HNSW) localized the bottleneck to the *embedding
  representation*; the model swap delivered. The depth-round arc: a methodical
  elimination, not a lucky guess — "I changed one variable at a time against a
  committed baseline until the data pointed at the embedding, then proved it."
- **The catch (productionizing):** adopting 3-large needs `vector(3072)` in Neon =
  ~2x per-vector storage. The 3-small corpus already uses 468/512 MB, so a 3-large
  corpus (~700 MB+) does NOT fit the free tier -- adoption requires a Neon paid
  tier (which also unblocks corpus expansion). The ablation proves the lever; the
  measured +0.13 / +0.30-on-tables is the evidence that justifies the upgrade.
- **Open:** Voyage finance-2 (domain-tuned) may lift tables further (optional next
  test). Cost: 3-large embeddings are ~6.5x 3-small ($0.13 vs $0.02 / 1M tokens).

### Larger chunks over 3-large — bigger helps (1024: recall@5 0.64, tables 0.72)
- **Result (2026-06-27, local exact, all 3-large):** recall@5 512=0.573, 768=0.60,
  **1024=0.64**; recall@10 up to **0.767**; tables 0.62 -> 0.64 -> **0.72**; domain
  0.44 -> **0.54**. Monotonic with chunk size.
- **Real, not just metric inflation — the prose control proves it:** the worry was
  that fuzzy >=0.5 overlap rewards bigger chunks (they hold more of the large gold
  spans). But **prose stays flat** (0.66/0.64/0.66) — prose spans are small, so a
  pure artifact would lift prose too. It doesn't. The gains land exactly on the
  table-heavy categories: genuine retrieval improvement + some metric coupling.
- **Tradeoffs (1024 isn't a free win):** coarser citations (~2x larger source,
  worse for verify-the-line financial QA), more generation tokens/latency, some
  recall@k inflation. **Side benefit:** fewer chunks (15k vs 26k) ~= half the
  storage of 512 -> makes 3-large much more affordable to productionize.
- **Best config so far: dense + 3-large + ~1024-token chunks** — recall@5 ~0.64,
  tables ~0.72, vs the 0.44/0.32 baseline; approaching the 0.75 target. Going
  larger (1536+) keeps chasing the metric at rising citation cost — 1024 is a
  defensible stopping point.

### Embedding dimensions — 3-large@1536 keeps the win (free-tier productionization)
- **Problem:** 3-large@3072 is 2x storage; 3-large + 1024 chunks ~= 525 MB, just
  over the 512 MB Neon free tier (so it looked like adoption needed a paid tier).
- **Matryoshka:** OpenAI 3-* embeddings are trained so truncating to the first N
  dims + renormalizing == the native reduced-`dimensions` output. Tested by
  truncating the cached 3-large@3072 vectors (512-chunk corpus) — near-free.
- **Result (2026-06-28):** **3-large@1536 == 3-large@3072** on recall@5 (0.573)
  and recall@10 (0.667); tables 0.62 vs 0.60 (noise). Even @256 holds ~0.567 —
  still far above 3-small@1536's 0.44. The recall gain lives in the first 1536 dims.
- **Consequence:** productionize 3-large at **1536 dims** -> SAME `vector(1536)`
  schema, SAME storage as today -> **fits the free tier, no upgrade, no schema
  change.** With 1024-token chunks (fewer rows) the best config is ~320 MB.
- **Decision — the deployable winning config: dense + text-embedding-3-large
  @1536-dim + 1024-token chunks** (recall@5 ~0.64) at ZERO infra cost. Needs
  embed.py to pass the OpenAI `dimensions` param + a destructive re-ingest to
  replace the 3-small corpus. Depth-round lesson: a constraint ("needs a paid DB")
  dissolved under one more cheap measurement — check before you spend.

### Productionized — v2 baseline 0.64 (deployed == measured)
- Adopted the winning config in the live system: dense + text-embedding-3-large
  @1536-d + 1024-token chunks (`configs/v2.yaml`). Re-ingested into Neon: 15,192
  chunks, 84 docs, **274 MB** — fits the free tier (vs the 512 cap), confirming the
  Matryoshka path needs no upgrade.
- **New baseline (v2, retrieval-only, 150 q, fuzzy, 0 errors):** recall@5 **0.64**
  (v0 0.44), recall@10 **0.747** (0.54), MRR **0.492** (0.317); tables **0.70**
  (0.32), domain **0.56** (0.34), prose 0.66 (flat). Measured through the *same*
  QueryEngine the API uses — the deployed system IS this number. Reproduces the
  offline ablation -> productionization validated; recall@10 ~= the 0.75 target.
- **Ops lessons (both surfaced as failures, both fixed):** (1) a full corpus swap
  on a near-full Neon DB must `TRUNCATE` first — per-doc DELETE+INSERT leaves dead
  tuples that blow the 512 MB cap mid-swap. (2) Long ingests need `--resume` — a
  transient connection drop shouldn't force re-embedding the whole corpus.
- **Full v2 baseline (2026-06-29, 150 q, full pipeline, 0 errors):** recall@5 0.64,
  recall@10 0.74, **faithfulness 0.929** (holds above the 0.80 target — generation
  quality survived the retrieval change), cost **$0.009/q**, latency p95 15.3 s.
  Caveat (rule #2): the eval runs top_k=10 + judge on, so its cost/latency
  *overstate* production — the live API (top_k=5, judge off) is ~$0.005–6 and
  faster. The larger 1024-chunks raised cost vs v0 ($0.0063) — an honest tradeoff
  of the recall win.

### Latency — generation is the wall; faithfulness judge taken off the API path
- **Measured breakdown (v2, per /query, 5-q sample):** retrieval ~2.2 s (19%),
  generation ~5.9 s (52%), faithfulness judge ~3.3 s (29%). Total ~11.4 s.
- **Honest finding:** **p95 e2e <2.5 s (the design-doc target) is NOT reachable
  with synchronous Haiku generation** — a grounded answer over 5×1024-token chunks
  is ~6 s on its own. The target was set without accounting for generation cost.
  Retrieval latency *is* met (~0.4–2 s). This is a target-mis-specification caught
  by measuring before optimizing — exactly the depth-round move.
- **Shipped:** the faithfulness judge (a 2nd LLM call) is now **off the /query
  critical path by default** (`with_faithfulness` opt-in on the request; falls back
  to `cfg.eval.faithfulness` so eval still computes the committed number; the demo
  opts in for the live badge). ~29% off request latency (~11 s → ~8 s).
- **Remaining latency levers:** (1) **streaming** — time-to-first-token <1 s, the
  real UX fix for a generation-bound RAG; (2) **connection pool** (concurrency, the
  known-debt single-connection item). The <2.5 s target should be reframed as TTFT
  or retrieval-latency, not e2e-with-generation.

### Streaming — SSE endpoint for low time-to-first-token
- Added **`/query/stream`** (Server-Sent Events): streams answer deltas, then a
  final frame with citations + metrics. Anthropic `messages.stream`; same retrieval
  + prompt as `/query` (so the streamed answer == the `/query` answer); judge off
  (can't stream). Demo renders it with `st.write_stream`. `/query` + the eval path
  are unchanged.
- **Measured TTFT ~3.4 s** (first token) vs ~8 s waiting for the full non-streamed
  answer — a real perceived-latency win. **Honest catch:** TTFT is now
  *retrieval-bound* — the 3-large query embedding (one OpenAI call) is ~2–3 s, so
  TTFT can't dip below that without caching / a faster query embed. The query-embed
  is the new latency floor; streaming hides the generation time, not the retrieval.

### Generation — Claude Haiku 4.5, temperature 0, grounded prompt, numbered citations
- **Alternatives:** Sonnet/Opus (stronger, slower, pricier).
- **Tradeoff:** generation is **not** the bottleneck — faithfulness is already
  **0.941** (> 0.80 target), so a bigger model buys nothing on the metric that
  matters; retrieval is where all the error is. Simpler-first holds.
- Temp 0 for reproducibility; the prompt forces grounding and the parser pulls
  `[n]` citation markers back out to link answer → source.

### Faithfulness — self-contained Haiku judge (RAGAS *definition*), not the RAGAS library
- **Alternatives:** RAGAS proper (the locked design doc named it).
- **Tradeoff:** RAGAS is built for LangChain 0.x and won't import under LangChain
  1.x (imports `ChatVertexAI` paths that no longer exist); pinning back breaks
  `langchain-openai`/`langgraph` and risks the `openai`/`anthropic` deps. A working
  RAGAS = a fragile dependency tower → violates reproducibility (rule #4). So: one
  judge call (temp 0) scoring the fraction of answer claims supported by retrieved
  sources — RAGAS's definition, zero added deps, reproducible. Verified
  discriminating (grounded → 1.0, hallucinated → 0.0; grounded refusal → 1.0).
- **Known limitation (say it before they do):** the judge is itself an LLM →
  LLM-judge bias. The mitigation is spot-checking ~20 judgments/run for agreement;
  that's on the list, not yet done.

### Eval cost / pricing
- Haiku 4.5 confirmed at **$1.00 / 1M input, $5.00 / 1M output**; measured V0
  **$0.0063/query** (under the <$0.01 floor). `cost_is_estimate` is *derived*
  (`model not in PRICING`), not hardcoded — a future unpriced model auto-flags
  itself rather than silently reporting a fake number.

### Deployment — FastAPI on Cloud Run, single shared DB connection
- **Tradeoff:** scales-to-zero, free tier. **Known debt:** one long-lived
  connection serializes concurrent queries (safe — verified 6/6 concurrent — but a
  throughput ceiling and a latency contributor). Connection pool is the fix; named
  as debt, not hidden.

### Retrieval depth — top_k = 5, report recall@5 and recall@10
- **Alternatives:** larger k (more context to the LLM, more cost/latency, more
  distractors that can pull faithfulness down); k=3 (sharper, riskier on recall).
- **Tradeoff:** 5 chunks is enough context for grounded single-/few-fact answers
  while keeping the prompt cheap; recall@5 is the headline, recall@10 the
  "how-much-is-just-out-of-reach" diagnostic. The benchmark floor is defined at @5.

### Architecture — one shared `QueryEngine` (the API and the eval call the same path)
- **Alternative:** a separate, eval-only retrieval/generation path (common, and
  faster to hack).
- **Tradeoff:** a second path means the numbers describe something the user never
  hits. One engine guarantees the committed metrics describe the *deployed*
  system. Costs some flexibility (eval can't shortcut around generation) — worth
  it; this is the single most important honesty decision in the codebase.

### Eval — retrieval-only mode (`--no-generate`), recall measured without the LLM
- **Decision (2026-06-15):** added a retrieval-only eval path. recall@k / MRR are
  pure *retrieval* metrics — they depend only on the query embedding + the DB, not
  on generation — so the runner scores them with **zero Anthropic calls**.
- **Alternative (what we had):** always run the full pipeline; every question
  generates + judges, so a credit outage killed the run *and* the recall numbers
  (the misleading 0.274 partial result).
- **Tradeoff / why it stays honest:** `run()` and the new `retrieve()` share the
  *same* retrieval code, so retrieval-only recall == full-pipeline recall (verified:
  dense reproduced the committed 0.44 exactly). The JSON self-marks
  `mode: retrieval_only` and nulls cost/faithfulness so a free run can't be misread
  as a full one. Bonus: isolates the retrieval ablation from generation cost +
  latency, and is the fix for the credit-aborted-eval footgun — infra failure no
  longer masquerades as a result.

### Methodology — V1 as separate, individually-measured increments (one variable at a time)
- **Alternative:** the locked design doc bundled V1 = corpus expansion + hybrid +
  reranker + 100 custom queries as one phase.
- **Tradeoff:** changing the corpus *and* the retrieval method at once makes the
  recall delta un-attributable (more docs = more distractors, which moves recall
  independent of the algorithm). Unbundling into V1.1 hybrid → V1.2 reranker →
  V1.3 corpus → V1.4 full eval, each A/B'd against the committed 0.44 baseline,
  is what "ablation-friendly, one variable at a time" actually requires. Slower to
  the impressive-sounding end state; the *attribution* is the whole point.

### Strategic direction — feature-complete to the doc's ambition, adapted by findings
- **Alternative I proposed and we rejected:** trim scope to a pure "ablation
  story" (drop S&P 100 / observability / Next.js), maximize measurement depth.
- **Decision (2026-06-11):** keep the full design-doc ambition, but let V0 data
  reshape *sequence and emphasis* — pull table-extraction forward (it's the
  dominant recall lever at 0.32), treat retrieval as the only bottleneck
  (faithfulness already solved at 0.94), and **decouple the faithfulness judge
  from the production latency number** (keep the demo badge, move the judge
  async/batch so p95 is measured without the second LLM call). Reasoning: the
  ambition is what makes it portfolio-grade; the findings just aim it.

### Demo surface — Streamlit, model hidden, cited-vs-retrieved badges
- **Alternatives:** Next.js playground (V2-optional); exposing a model selector.
- **Tradeoffs:** the project's value is eval rigor, not frontend — Streamlit + a
  tight README GIF is enough signal. Model choice stays a dev/eval concern (it's
  in the JSON for traceability, not a UI affordance) so the demo doesn't invite
  "try GPT-4" noise. **Cited vs retrieved** chunks get different badge styles so a
  viewer sees what was *in context* vs what the LLM *actually drew from* — honesty
  made visible, no overclaiming.

### Benchmark — FinanceBench (150 q), license-aware
- **Alternative:** only hand-built queries (no public anchor, un-comparable).
- **Tradeoff:** a public benchmark gives numbers that can be set against published
  baselines; the cost is that its gold spans are messy tables (drove the
  fuzzy-match decision). CC-BY-NC-4.0 → non-commercial portfolio use only, PDFs
  not redistributed. The 100 hand-built custom queries (V1.3) complement, not
  replace, it.

---

## Meta-answer (if they ask "how did you keep yourself honest?")
A written contract (`CLAUDE.md`) with non-negotiable rules — no fake APIs, never
cherry-pick numbers, pair every choice with its reason, one variable per ablation —
and a **locked design doc where every scope deviation needs a dated amendment with
rationale**. Those amendments are a paper trail of *why* the system is what it is.
This file is largely an extraction of that trail.
