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
  **temp 0**, **pinned lockfile**, **timestamped JSON committed per run** — so every
  past headline traces to the exact config that produced it and the harness is
  deterministic. **Honest reproducibility scope** (measured on a pristine `git
  archive` of HEAD — 139 tracked files, no `.env`, no `data/`, no committed
  embeddings): the **offline test suite reproduces from a clean clone** (115 tests,
  no network/keys/DB), but the **headline recall numbers do NOT** — the FinanceBench
  PDFs are CC-BY-NC and gitignored, and the 274 MB embedded corpus lives in Neon, not
  the repo, so re-running the benchmark needs API keys + a populated DB, not just
  `git clone`. The README's "Reproducibility" section spells out the three tiers.
  (Earlier this line claimed "one command reproduces the headline numbers from a clean
  clone" — corrected: that was false, and a rule-#2 overclaim.)
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
  inflate it. Real signal + a measurement artifact, both pushing the same way. (This
  early caveat was right; the confound crossing later *quantified* it — ~half the
  512->1024 overlap recall gain is this inflation. See "Crossing the confound" below.)
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
- **Confirmed by the confound crossing (below), with a magnitude caveat:** embedding IS
  the real driver — chunk-invariant answer accuracy rises **+0.08** across 3-small->3-large
  and refusals fall 0.51->0.39. But the +0.13 recall is the OVERLAP/fuzzy number and is
  matcher-dependent: **+0.05 under semantic, +0.00 under exact-substring at recall@5**.
  3-large pulls chunks with more token/meaning overlap, not more *exact* gold spans. So
  "THE lever" holds (it's the one real system win), but its headline *size* is a property
  of the fuzzy metric as much as the retriever.
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
- **Real, not just metric inflation — the prose control (partly) shows it:** the worry
  was that fuzzy >=0.5 overlap rewards bigger chunks (they hold more of the large gold
  spans). Prose stays flat (0.66/0.64/0.66) — prose spans are small, so a *pure* artifact
  would lift prose too; it doesn't. So the gain isn't pure inflation. But the prose
  control only rules out "all inflation" — it doesn't size the coupling.
- **CORRECTED by the confound crossing** (`confound_study_20260731T170928Z.json`, see
  the section below): crossing chunk size against all three matchers + chunk-invariant
  answer accuracy quantified the split. Of the +0.080 overlap chunk-gain, only **~+0.037
  is real** (survives strict/semantic; accuracy rises a matching +0.033) and **~+0.043 is
  overlap-metric inflation.** So "some metric coupling" undersold it — the inflation is
  ~half the chunk gain, not a footnote. This section's original "genuine retrieval
  improvement" claim stands but was over-confident about the size.
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
- **Latency levers — all shipped by the post-V2 hardening round:** (1) **streaming**
  (TTFT ~1 s on the live API); (2) **connection pool** (4.9× concurrent); (3)
  **concision prompt** (generation tokens −25%). The <2.5 s target is best read as
  TTFT or retrieval-latency, not e2e-with-generation; live p50 (2.06 s) now meets it,
  p95 stays generation-bound. See the post-V2 hardening section for the measured numbers.

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
  LLM-judge bias. The mitigation is spot-checking ~20 judgments/run for agreement —
  **done this round: 19/20 (95%), the one miss judge-too-harsh** (see the post-V2
  hardening section + `docs/faithfulness-spotcheck.md`).

### Eval cost / pricing
- Haiku 4.5 confirmed at **$1.00 / 1M input, $5.00 / 1M output**; measured V0
  **$0.0063/query** (under the <$0.01 floor). `cost_is_estimate` is *derived*
  (`model not in PRICING`), not hardcoded — a future unpriced model auto-flags
  itself rather than silently reporting a fake number.

### Deployment — FastAPI on Cloud Run, connection pool
- **Tradeoff:** scales-to-zero, free tier. **Was known debt:** one long-lived
  connection serialized concurrent queries (safe — verified 6/6 concurrent — but a
  throughput ceiling and a latency contributor). **Fixed this round** with a psycopg
  pool — 4.9× on 24-way concurrent retrieval (see the post-V2 hardening section).

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

## Post-V2 hardening round (2026-07-30/31) — measurement + latency engineering

A review pass turned up gaps between what the docs *claimed* and what was
*measured*. This round closed them in the project's own spirit: measure first, be
honest when it hurts, commit an artifact for every number. Each item below traces
to a committed JSON.

### Substring recall floor — the honest bracket for v2
- Headline recall@5 0.64 is fuzzy(0.5): generous (set token-overlap, blind to
  number swaps like "5%" vs "25%"). Substring is unfairly strict (gold spans cross
  chunk boundaries). The strict floor had only ever been run on v0.
- **Measured (v2, 150 q, retrieval-only):** substring recall@5 **0.093**, recall@10
  0.127 → true recall@5 lives in the bracket **[0.093, 0.64]**. The gain survives the
  *strict* metric too: v0→v2 substring recall@5 **0.067 → 0.093** — so 3-large is
  real signal, not a fuzzy-overlap artifact. Artifact: `financebench_20260730T232517Z.json`.

### Faithfulness judge — the spot-check the design doc promised (now done)
- The design doc committed to manually adjudicating ~20 judgments/run for LLM-judge
  bias; it had never been produced.
- **Done:** an independent adjudicator (Opus — a *different* model from the Haiku
  judge) re-scored the 20 sampled (answer, sources, verdict) triples, verifying every
  headline figure by exact-string search against the full chunks. **Agreement 19/20
  (95%)**, and the one miss was the judge being too *harsh* (scored a fully-grounded
  answer 0.0) — so the committed 0.929 is not inflated by a lenient self-grader.
- **Caveat it surfaced:** faithfulness is *grounding, not correctness* — several
  answers are faithful-but-wrong (grounded in the wrong retrieved evidence). 0.93
  faithful and 0.64 recall are consistent. Writeup: `docs/faithfulness-spotcheck.md`.

### Connection pool — the single-connection debt, fixed and measured
- The known-debt single shared DB connection serialized concurrent retrieval.
  Replaced with a psycopg pool (`register_vector` per connection, autocommit,
  min=1/max=8, env-tunable).
- **A/B (Neon, 24-way concurrency, retrieval isolated):** wall 979 ms → 201 ms
  (**4.9×**), throughput 24 → 120 qps, tail p95 934 → 193 ms. Also removes the
  single-socket point of failure (the pool auto-reconnects; the manual reconnect
  dance is gone). **Honest scope:** single-*request* latency is unchanged — retrieval
  was never the single-query wall; this is a concurrency/throughput + robustness fix.
  Artifact: `pool_bench_20260731T011348Z.json`.

### Live-API latency — measured, not asserted
- The docs long asserted the live API was "~$0.005–6 and faster than the eval's 15 s"
  with no committed run — a rule-#4 violation. **Measured (40 warm requests to the
  deployed `/query`):** server p50 **2.06 s** / p95 **6.19 s**, cost **$0.0045/q**,
  streaming TTFT **~1.0 s** (supersedes the earlier local ~3.4 s TTFT figure — the
  deployed service is faster). True cold start ~19 s (Cloud Run scale-to-zero). p50
  meets the <2.5 s target; p95 is generation-bound. Artifact: `api_latency_20260731T004411Z.json`.

### Concision prompt — trimming the generation tail (the one that fought back)
- Generation time scales with output tokens, and Haiku padded answers with preamble
  ("Based on the sources provided…"), restated questions, and closing summaries. A
  concision block in `generate/answer._SYSTEM` cut it: **output tokens −25%,
  generation p50 −~20%** (back-to-back A/B), 0 answers losing citations.
- **The honest part:** the first, blunter pass **regressed faithfulness 0.90 → 0.775**
  on the spot-check. Reading the per-question judgments showed it was *mostly a judge
  artifact* (terse refusals mis-scored; it even penalized a concise answer that was
  MORE correct — it stopped hallucinating a wrong-year figure) plus one real loss
  (dropped derivable figures on a "compute COGS" question). Two targeted clauses fixed
  both — derive-from-components, and an explicit refusal the judge reliably recognizes
  — and faithfulness came back to **0.9375**, above the verbose baseline.
  **Would-do-differently:** gate prompt changes on the spot-check from the start.
- **Validated at full scale (150 q, concise, 0 errors):** recall@5 0.64 (unchanged —
  retrieval untouched), faithfulness 0.929 → **0.934**, cost $0.0090 → **$0.0087** — a
  strictly-positive change. Artifact: `financebench_20260731T025337Z.json`. (First
  attempt died mid-run on Anthropic credit depletion — 110 billing errors, a live
  instance of the "eval runner swallows infra failures" debt; the partial was
  discarded, not cited.)

### Live-demo liveness audit — and the fix I *didn't* make
- A stranger-with-no-keys audit of the deployed demo, before touching anything: both
  advertised URLs were up (the Cloud Run web host + a stale GitHub Pages link the docs
  still pointed at — consolidated to the one cache-proof Cloud Run host).
  `SEC_RAG_REQUIRE_KEYS` is unset, so keyless visitors run on the owner's keys
  (rate-limited), not a 401. Neon was awake; the real first-visit cost is a **~13 s
  cold start** (Cloud Run scale-to-zero), which the README's "~15–25 s" never
  mentioned — corrected to measured numbers (warm ~2 s corpus / ~6 s a fresh EDGAR
  filing).
- **The honest part — I flagged a defect that wasn't one.** The audit noted the live
  cache averaged "~115 chunks/filing" and I inferred the live path indexed only a
  *slice* of each 10-K. When asked to fix it, I measured first: it indexes the **full**
  document — AAPL 10-K = 87 chunks, JPM 10-K = 331 chunks, both reaching Item 15/16 +
  the signature page, with all sections and financial-statement tables intact. The
  "~115" was a mean across mixed filing sizes (8-K/10-Q/10-K), not truncation. So I made
  **no change to the indexer** — fixing a working path is churn, and rule #2 (honest
  even when inconvenient) says report the wrong premise, not invent a diff.
- **What I added instead — a coverage guardrail**, so a *real* future truncation can't
  hide behind a plausible answer: `_coverage_check` flags a 10-K/10-Q whose extracted
  text doesn't reach the SEC §13 signature attestation (the one end-of-document marker
  stable across issuers — Item numbering varies), and `_index` logs coverage + WARNs on
  a miss. Observability only (still indexes what it got); pure helper, unit-tested;
  verified on live AAPL HTML (full → ok, a 5 % slice → warned).

### Crossing the confound — how much of V0->V2 is real vs metric inflation
- The V0->V2 headline moved TWO variables (3-small->3-large AND 512->1024), so it was
  never decomposed and the 3-small@1024 cell was never run. Ran the **full grid**:
  {3-small,3-large} × {512,1024,2048} × {strict,overlap,semantic}, retrieval-only, plus
  answer accuracy (the chunk-invariant tiebreak) at three cells.
  `eval_results/confound_study_20260731T170928Z.json` (+ `confound_grid_*.json`).
- **The two effects are additive (interaction ≈ 0).** Decomposing the +0.207 overlap
  recall@5 gain: embedding **+0.127** + chunk **+0.080**. Of the chunk +0.080, only
  **~+0.037 is real** (survives strict/semantic; accuracy chunk-effect +0.033); the
  other **~+0.043 is overlap-metric inflation** (bigger chunk -> easier 50%-token bar).
- **Accuracy is the arbiter** (can't be inflated by chunk size): over-all answer
  accuracy climbs V0->V2 **0.36 -> 0.47 (+0.11)**, split embedding +0.08 / chunk +0.033;
  refusals fall 0.51 -> 0.37. So **~80% of the headline is real, ~20% is chunk inflation**,
  and the embedding model is the real driver — not the chunk change. Corrects the earlier
  "roughly a third is chunk-size" line, which counted the inflated recall, not the real gain.
- **Chunk benefit saturates by 1024:** 2048 is flat/negative under overlap+semantic (only
  strict keeps rising — the "big chunk swallows the multi-line span" artifact). 1024 was
  the right stopping point. (One caveat: the V2 accuracy cell reuses the S2 Neon run — the
  local V2 accuracy arm hit an Anthropic credit-out at 66/150; same corpus/top_k/prompt.)

### Eval runner fail-fast — a partial run can no longer look like a result
- **The debt (it bit twice this round):** the per-question retry meant to survive a
  transient blip also swallowed *account-level* outages. The concise-prompt full run
  died on Anthropic credit depletion (110 "errors"); the confound accuracy arm at
  66/150. In both, every remaining question failed the same way, yet the runner still
  emitted an aggregate over the prefix that ran before the outage — a partial that
  reads exactly like a real benchmark number (a rule #2 landmine).
- **The distinction that makes it non-trivial:** the fatal signal is NOT a clean
  status code. Anthropic credit-out is a **400** ("Your credit balance is too low…"),
  not a 401/402 — so it must be caught by the *message*, not the status. OpenAI
  quota-out is a **429** that is byte-for-byte a transient rate limit except for an
  `insufficient_quota` marker — so a plain 429 must stay *retryable* while a quota 429
  is fatal. A naive status-code check gets both wrong.
- **Fix:** `eval/errors.py:fatal_reason(exc)` — a pure, version-robust classifier
  (keys off the `status_code` attribute + real observed message markers, walks the
  `__cause__`/`__context__` chain, no SDK-class imports). The runner aborts the
  instant it fires; the report carries `complete: false` + `aborted: {id, reason,
  error}`; `main()` exits non-zero with a "DO NOT CITE" banner so `make eval`/CI fail
  loudly. Same fix in `confound_accuracy.py`. Transient errors keep the bounded retry.
- **Honest scope:** this is a *guardrail*, not a new number — it changes how a broken
  run reports, not any committed metric. Signatures are the real ones (copied from the
  committed error strings in `eval_results/*.json`), not guessed. Verified: 11 tests
  (`tests/test_eval_errors.py`) — classifier cases + a monkeypatched abort that proves
  the loop short-circuits (never reaches the question after the fatal one) and marks
  the run incomplete; plus a live retrieval-only smoke confirming a *clean* run still
  stamps `complete: true` and exits 0. It also makes `ingest/embed.py`'s "same rule as
  the eval runner" comment true — the runner now actually enforces it.

### Net after this round
- Every headline number now traces to a committed artifact. The three latency levers
  (pool, judge-off-path, concision) are shipped; live p50 meets <2.5 s and streaming
  TTFT ~1 s covers perceived latency; the residual p95 is content-legitimate answer
  length, not padding. The eval harness can no longer pass off a partial as a result
  (fail-fast above). What's left is honest and small: correctness-vs-gold isn't
  directly scored (faithfulness ≠ accuracy), and the faithfulness judge should emit
  per-claim verdicts so its ~5% error rate is auditable.

---

## Meta-answer (if they ask "how did you keep yourself honest?")
A written contract (`CLAUDE.md`) with non-negotiable rules — no fake APIs, never
cherry-pick numbers, pair every choice with its reason, one variable per ablation —
and a **locked design doc where every scope deviation needs a dated amendment with
rationale**. Those amendments are a paper trail of *why* the system is what it is.
This file is largely an extraction of that trail.

---

# Appendix — full build log (every commit, what it did, why)

The sections above are organized by *topic* (the decision log) and by *theme* (the
hardening round). This appendix is the **chronological record**: all 91 commits, in
order, grouped into the phases the project actually moved through, each annotated
with **what was done / the choice / the result** — so you can read it top to bottom
and understand the whole build, or `git show <hash>` any line to see the diff.

Read the phases as the story: **V0** (an honest baseline) → **V1** (a ten-lever
ablation hunt that found the one real lever) → **V2** (productionize it) →
**product** (frontend + live EDGAR) → **hardening** (measure everything the docs had
only asserted, and catch the project's own overclaims). Counts shown (tests, recall)
are the numbers *as of that commit*; they grew over time (70 tests → **118** today).

## Phase V0 — an honest baseline (2026-06-02 → 06-06)

*Goal: the simplest thing that works, measured on a public benchmark, deployed.*

- `650857e` **Initial scaffold + reproducible env** — the whole skeleton in one
  commit: the shared `QueryEngine` (`pipeline.py`), ingest (parse/chunk/embed/load),
  dense retrieval, Haiku generation, FastAPI `/health`+`/query`, the FinanceBench
  eval runner, Streamlit demo, pinned `requirements.lock`, seed 13, `CLAUDE.md`
  contract. The *architecture* decision that matters is here: **one engine for API
  and eval** so numbers can't drift from production.
- `6c18162` **Fix ingest against real PDFs** — first contact with real FinanceBench
  files broke the parser; fixed `parse.py` + pinned the deps that actually worked.
- `eac9569` **Send the query vector as a real `pgvector` type** — pgvector's `<=>`
  needs a `vector`, not a Python list; retrieval silently returned garbage until this.
- `636707a` **V0 baseline committed + design-doc amendment** — the first honest
  number: **recall@5 0.44** (below the 0.55 floor, reported as-is). Two eval JSONs
  committed as the frozen record. The amendment is where **fuzzy(0.5) primary /
  substring lower-bound** was decided (substring 2/10 vs fuzzy 7/10 → substring was
  measuring text-extraction agreement, not retrieval).
- `c7d14f1` **Fix the falsy-zero page-0 bug** — `if page:` dropped page 0 (a real
  cover page) as evidence. This is the exact bug the `financebench.py` refactor's
  `_first_present` comment still guards against; fixed + regression test.
- `2617162` **Fix `/query` 500 after idle** — Neon kills idle-in-transaction
  connections; switched the engine's connection to **autocommit** so a warm instance
  doesn't 500 on the first request after a lull.
- `fbc580f` **Validate `top_k` at the API boundary (1..50)** — reject bad input at
  the schema, not deep in retrieval; test added.
- `78115df` **Make the eval runner resilient to per-question failures** — retry a
  question rather than kill the run. *(This well-intentioned change became the footgun
  fixed much later in `6dddc9c`: it also swallowed billing outages as "question
  failures." Noting it here so the arc is visible.)*
- `b174fee` **Confirmed Haiku 4.5 pricing; derive `cost_is_estimate`** — real prices
  ($1/$5 per 1M in/out), and the estimate flag is *derived* from `model not in
  PRICING` so an unpriced model auto-flags itself instead of printing a fake cost.
- `d848854` **Faithfulness judge (badge) + engine reconnect** — the self-contained
  Haiku judge (RAGAS *definition*, no fragile RAGAS dep) lands as a badge.
- `9a02951` **Aggregate faithfulness + real cost → full V0 baseline** — the complete
  committed V0: recall@5 0.44, **faithfulness 0.94**, **$0.0063/q**.
- `f5baf56` **`--sleep` throttle** — a rate-limit release valve for full eval runs.
- `4bbf492`, `3db70ac`, `6534be8` **V0 docs + deploy** — record pricing/limitations
  in the README; **Cloud Run deploy** (`f0a03fb`: Dockerfiles, `X-API-Key` guard,
  `DEPLOY.md`); live URLs; the **Mermaid architecture diagram** that renders on GitHub.

## Phase V1 — the ablation program (2026-06-06 → 06-28)

*The substance: hypothesis → change one variable → measure against 0.44 → keep/kill →
write down why. Ten levers, one winner. Full reasoning is in the Decision log above;
this is the timeline.*

- `8e7c661` **V1 plan + the methodology call** — execute V1 as *separate,
  individually-measured increments*, not a bundle (changing corpus + method at once
  makes the delta un-attributable). Also records the **`pg_search` on Neon is
  deprecated** finding → lexical = core Postgres FTS.
- `a9dd29e` **V1.1: hybrid retrieval built** — dense + Postgres FTS fused with RRF;
  new `lexical.py`, `hybrid.py`, schema tsvector/GIN, `v1.yaml`.
- `c4fdbc6` **The retrieval-only eval mode + the hybrid A/B + fusion sweep** — the
  tooling win that unblocked the whole program: `--no-generate` scores recall/MRR with
  **zero Anthropic calls** (pennies, and a billing outage can't fake a result). Dense
  reproduced **0.44 exactly**, validating the harness. Verdict: **hybrid regressed**
  (0.44→0.35); the fusion-weight sweep showed **no blend beats dense** (lexical-only
  0.04, tables 0.00).
- `26674b3`, `7011dd3` **CLAUDE.md kept honest after each finding** — the "where we are
  now" section is re-synced to the *true* state after V1.1 (`26674b3`) and again once
  hybrid is retired and dense is confirmed the ceiling (`7011dd3`). Keeping the working
  contract's state block accurate is itself part of the discipline, not an afterthought.
- `42c3798` → `04b9934` **The hypothesis I got wrong, caught cheap** — `42c3798`
  amended the design doc to blame *parsing* for the tables gap and promote table
  extraction. Then `04b9934` ran two ~5-minute diagnostics (`diag_table_parse`,
  `diag_retrieval_depth`) that **disproved my own amendment**: table evidence
  *survives* pypdf (8/8), and the gap is **ranking (26% at rank 6–20) + recall (32%
  miss top-100)**, not parsing. A *correcting* amendment followed. Cheap tests before
  an expensive re-ingest.
- `4a31698` **V1.2 reranker (BGE cross-encoder) — built, measured, rejected** —
  0.44→0.39; it demoted good dense hits. Kept as an off-by-default config knob +
  documented negative result.
- `bb7d4c0` **Chunk-size 256 — worse (0.28)** — the dilution hypothesis was backwards;
  smaller chunks split the big table spans. Ran on a **local exact-cosine numpy index**
  (Neon's free tier was maxed) — which also proved HNSW wasn't costing recall. Bonus:
  surfaced that `embed.py` had no 429 retry → added backoff.
- `d938a4f` **Embedding → 3-large: THE lever** — 0.44→**0.57**, **tables 0.32→0.62**.
  Five prior negatives had localized the bottleneck to the *representation*; the model
  swap delivered. Same 512 chunks, only the model differs.
- `f914ad0` **Reranker over 3-large — hurts *worse*** — 0.57→0.38; definitively
  retired (more good candidates = more for a generic reranker to demote).
- `66ac6f8` **`embed.py` fail-fast on `insufficient_quota`** — a billing 429 is not a
  transient rate limit; stop retrying it. (The rule the eval runner didn't yet obey.)
- `34336fc` **Larger chunks (1024) over 3-large** — →**0.64**, tables **0.72**; prose
  stays flat (so it's not *pure* metric inflation — the confound crossing later sized
  the real-vs-inflated split).
- `63a6f54` **Dimension ablation — the constraint that dissolved** — 3-large is 3072-d
  (2× storage, wouldn't fit the free tier). Matryoshka truncation to **1536-d keeps the
  full recall** at the *same* storage/schema → productionizable at zero infra cost.

## Phase V2 — productionize the win (2026-06-28 → 06-29)

- `dc9313d` **v2 config** — `configs/v2.yaml` = dense + 3-large@1536 + 1024 chunks;
  `embed.py` passes the OpenAI `dimensions` param; FTS dropped from the default path.
- `2bafea6` **v2 live baseline recall@5 0.64** — re-ingested to Neon (15,192 chunks /
  **274 MB**, fits the free tier). Measured through the *same* engine the API uses →
  deployed == measured. Ops lessons surfaced as failures: **TRUNCATE-before-swap** (a
  per-doc swap on a near-full DB blew the cap via dead tuples) and `load.py --resume`.
- `bc5784b` **Document `SEC_RAG_CONFIG`** — the query and corpus embed model *must*
  match or retrieval is incoherent; made explicit in `DEPLOY.md`.
- `e7491dd` **Faithfulness judge off the `/query` critical path** — the 2nd LLM call
  (~29% of latency) is now opt-in; eval still computes the committed number.
- `cfd59db` **Full v2 baseline committed** — 150 q, full pipeline, 0 errors:
  **faithfulness 0.93** (survived the retrieval change), recall@5 0.64.

## Phase — the product surface: frontend (2026-06-29 → 06-30)

*Decision (with the owner): keep the benchmarked backend, add a static frontend on the
deployed API. Streamlit can't do the target design or run on Pages, so: vanilla
HTML/CSS/JS.*

- `f980fbf` **Streaming `/query/stream` (SSE)** — answer deltas then a final frame with
  citations + metrics; big perceived-latency win (TTFT).
- `ae90c74` **Static frontend + GitHub Pages** — the first dark landing page over the
  live API; Pages deploy workflow.
- `c77a17d` → `b0c32c3` **The frontend/API-key UX iteration** — light turquoise theme
  (`c77a17d`, owner's call); auto-run the pending question after a key save (`fa4be1a`);
  key made dismissable/optional (`cc335e5`, `a985f9f`); a **temporary hardcoded gate
  key** for frictionless testing (`265f801`, explicitly "rotate + remove after");
  cache-busting `?v=` + a **visible build marker** to diagnose stale loads (`4d6627b`,
  `8016819`); then **removed the key UI entirely** for a zero-friction open API
  (`b0c32c3`). The build-marker/cache-bust work is the start of the caching saga below.

## Phase — live EDGAR: any company, on demand (2026-06-30)

*Decision (with the owner): add a live path alongside the benchmark. Architecture flip:
static pre-indexed corpus → fetch-and-index-on-demand (in-memory exact-cosine index →
no storage cap; Neon cache for cold starts). EDGAR was verified live before building.*

- `f3778cd` **Live EDGAR backend** — `edgar/client.py` (ticker→CIK, submissions API,
  document fetch, HTML→text) + `edgar/live_engine.py` (fetch→parse→chunk→embed→retrieve→
  stream, same event shape as the benchmark path).
- `2ce0cea` **Wire to API + frontend** — `/query/live/stream`; ticker box → any
  company. Caught a real prod-only gap here: `bs4` wasn't a declared dependency (worked
  locally, 500'd deployed) → added to `pyproject`.
- `2508c09` **Better HTML parsing** — strip XBRL noise, preserve tables.
- `7fec695` **All filing types (10-K/10-Q/8-K) with auto-detection** from the question.
- `5f52c2a` **BYOK** — visitors run on their *own* OpenAI + Anthropic keys via
  per-request headers (their credits), making safe public sharing possible.
- `41b0272` **Dedicated table extractor** — merges split currency cells (`$` and the
  number landing in separate `<td>`s).
- `35d24e9` **Persistent Neon cache** for embedded filings — fast cold starts, bounded.
- `e561c74` **Multi-filing compare across periods** — "vs last year" / "5-year trend"
  pulls 2–5 filings.
- `0ec5864` **Section-labeled citations** ("Item 1A. Risk Factors") + `require-BYOK`
  flag + EDGAR tests.
- `365f657` **Frontier polish** — per-IP **rate limit**, deeper multi-filing, clean
  section handling, optional background **pre-warm** of popular tickers.
- `78d0221` **Test fix (70 passing)** — `detect_multi` parses digits, not the word
  "ten".

## Phase — docs + the caching saga (2026-06-30 → 07-02)

- `2324377`, `31baa03` **Docs** — `versions.md`, `decisions-and-steps.md`; refreshed
  README + CLAUDE.md to the shipped live-EDGAR system.
- `c6fdbaf` **Cache-proof hosting** — GitHub Pages forces a 10-min HTML cache we can't
  change (and had flaky deploys), so also serve `web/` from **Cloud Run with `no-store`
  headers** (`web_server.py`, `Dockerfile.web`).
- `b449c07` **The bug that only *looked* like caching (the big lesson)** — for many
  rounds "the old page keeps loading" was chased as a cache/CDN/browser problem. It
  reproduced on a *fresh device* (owner's call) → proving it wasn't cache. Real cause:
  a **CSS bug**, `.modal { display:flex }` overrode the HTML `hidden` attribute, so the
  key popup was permanently open for everyone. One line fixed it: `[hidden] {
  display:none !important }`. **Lesson: verify the rendered artifact, not the served
  files; a fresh-device repro is the fastest way to split "our bug" from "your cache."**
- `0dbfb76` → `a5948f5` **The demo-link flip-flop** — pointed back to GitHub Pages by
  request (`0dbfb76`), then settled on the **cache-proof Cloud Run host** two commits
  later (`a5948f5`) once Pages proved unreliable. The waffle is itself part of the
  hosting saga — Pages *looked* simpler until its cache + flaky deploys bit.
- `4b53277` **`docs/PROJECT-LOG.md`** — the exhaustive phase narrative (this appendix's
  companion; it stops here at 62 commits).

## Phase — post-V2 hardening + measurement (2026-07-30 → 07-31)

*A review pass found gaps between what the docs **claimed** and what was **measured**.
This round closed them in the project's own spirit — measure first, commit an artifact
for every number, and catch the project's own overclaims. The thematic write-ups are in
"Post-V2 hardening round" above; here they are in commit order.*

- `35bc7fb` **v2 substring floor + the faithfulness spot-check** — the strict recall
  bracket for v2 (`[0.093, 0.64]`) and the Opus-adjudicated audit of the Haiku judge
  (**19/20**, the one miss *harsh*) — the spot-check the design doc had long promised.
- `ae9cc89` **Measure the live API latency** — replaces an *asserted* "~$0.005 and
  faster" with a committed run: server **p50 2.06 s / p95 6.19 s**, **$0.0045/q**, TTFT
  ~1.0 s. (A rule-#4 fix: the number existed only as a claim before.)
- `e8fa6ac` **Connection pool** — the known single-connection debt, replaced with a
  psycopg pool; **4.9×** on 24-way concurrent retrieval. Honest scope: a
  concurrency/throughput fix, *single-request* latency unchanged.
- `a307d83` **Concision answer prompt (the one that fought back)** — output tokens
  −25%, generation p50 −~20%. The blunt first pass **regressed faithfulness 0.90→0.775**
  (mostly a judge artifact on terse refusals + one real loss); two targeted clauses
  brought it to **0.9375**. Would-do-differently: gate prompt changes on the spot-check
  from the start.
- `0f00c6a` **Authoritative 150-q run under the concise prompt** — 0 errors: recall@5
  0.64 unchanged, faithfulness →**0.934**, cost →**$0.0087** (strictly-positive change).
- `00f999a`, `31e2ae0` **Docs** — the hardening round written up; demo consolidated to
  the Cloud Run host with an honest **~13 s cold-start** note (the old "~15–25 s" was
  never measured).
- `4a72fb0` → `00fc967` → `112e4ff` **The coverage guardrail + the fix I *didn't*
  make** — a liveness audit inferred the live path indexed only a *slice* of each 10-K;
  measuring first showed it indexes the **full** document (the "~115 chunks" was a mean
  across mixed filing sizes). So **no indexer change** — instead added `_coverage_check`
  (warn if a 10-K/10-Q doesn't reach the §13 signature marker) as observability
  (`4a72fb0`), wrote the audit + "the fix I didn't make" into this doc (`00fc967`), and
  surfaced `sec_rag` logs in Cloud Run so the warning is actually visible (`112e4ff`).
- `a762965` **Correct four overstated README claims** — tests count, corpus size,
  V0→V2 attribution, and the metric framing. A deliberate self-audit (rule #2).
- `195f8ec` → `e067572` **Answer accuracy — is the answer *right*, not just retrieved?**
  Built a numeric matcher + LLM judge (`195f8ec`), committed the 150-q artifact
  (`0e789a2`: LLM 0.76 of attempted, 37% refusal), then used it as a **lever**: raising
  `top_k` 5→20 (`e067572`) cut refusals 0.37→0.32 and lifted over-all accuracy 0.47→0.50
  at ~2× cost — a deliberate, measured trade now baked into `v2.yaml`.
- `8d6d689` → `b6838eb` **Pluggable evidence matchers** — recall re-run under
  **strict / overlap / semantic** (`8d6d689`), then every headline ablation re-checked
  across all three (`b6838eb`) to see which conclusions survive the metric. This is what
  turned "recall@5 0.64" into "recall is a *bracket*."
- `10ff614` → `349a185` **Matcher validity — and the false positive** — built a human
  labeling harness (`10ff614`), then an **LLM-proxy** adjudication that reported the
  shipped overlap matcher agreeing best at **κ 0.67** (`349a185`). *This number was
  later shown to be an artifact of the LLM labeler* — see `5f869a2`.
- `ecc9460` → `ff6bde3` **Cross the model×chunk confound** — the full 2×3×3 grid +
  chunk-invariant answer accuracy: the +0.207 gain is **~80% real, ~20% chunk-metric
  inflation**, effects additive, embedding the real driver. Docs reconciled (`ff6bde3`).
- `d521988` **eval-as-CI** — a per-PR retrieval-only smoke gate on recall@5 vs a frozen
  baseline, params in `configs/ci_eval.yaml`, skips gracefully without secrets, ~$0.001/run.
- `6dddc9c` **Eval fail-fast on billing/account outages** — closes the `78115df`
  footgun (it bit *twice* this round). The hard part: a credit-out is a **400** (caught
  by message, not status) and a quota-out is a **429** indistinguishable from a transient
  limit except for a marker. `eval/errors.py:fatal_reason` classifies it; the run aborts
  with a "DO NOT CITE" banner. 11 tests.
- `ff4ef14` **Independently re-measure the pool win** — same A/B, **5.6×** this time;
  the exact ratio floats with Neon jitter, the win is robustly large.
- `a44a435` **Correct the "clean clone reproduces the headline" claim** — it was false
  (the corpus lives in Neon, PDFs are gitignored); the README now spells out the three
  reproducibility tiers. Another rule-#2 self-correction.
- `c050719` **Live-path eval harness + 25 hand-verified EDGAR questions** — begins
  scoring the *live* product (previously unscored — the gap PROJECT-LOG §8A flagged).
- `5f869a2` **The metric-validity study + README reframe** — hand-labeling the 50 pairs
  myself dropped the shipped matcher to **κ 0.18 (third of three, no matcher beats chance
  at n=50)**, exposing the `349a185` κ 0.67 as an **LLM-labeler artifact** — shipped once,
  then caught. README reframed as an *evaluation platform*. This is the honesty flagship.

## Phase — repo polish (2026-08-01)

- `714e312` **Refactor `financebench.py`** — extract a pure, testable `_normalize_row`;
  fold column resolution + required-field validation into `_resolve_field`; lift magic
  keys into constants. Public API + behavior unchanged; the 6 evidence tests still pass.
- `3200ffc` **Professional README makeover** — centered hero (demo screenshot), badges,
  TOC, scannable highlights with the deep caveats in `<details>`; added an **MIT
  `LICENSE`** (FinanceBench data stays CC-BY-NC), a real no-secrets **tests CI**
  workflow (backs the badge), and rendered `assets/`. Corrected the stale test count to
  the true **118 passing / 104 functions**.
