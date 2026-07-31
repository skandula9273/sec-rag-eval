# Is my own evaluation metric valid? A study of the matcher and the judge

*A report on what turned out to be true — not a defense of the project.*

Every retrieval number in this repo is an **evidence-hit rate**: the fraction of
questions for which a retrieved chunk "contains" the gold evidence, under some
matcher. Every answer-quality number leans on an **LLM judge**. Both are things I
built, so both are things I could have gotten wrong in a way that flatters the
result. This study measures whether they did.

Two findings up front, the unflattering ones first:

1. **The headline retrieval metric is mostly a property of the matcher, not the
   retriever.** The identical V2 retrieval scores read as recall@5 = **0.093**,
   **0.64**, or **0.807** depending only on which matcher scores them — a ~9×
   spread. The committed headline (0.64) is one point in that range, and it is a
   *slight over-estimate* of what a grader calls a real support.
2. **The labels I used to decide which matcher is right are not human.** The
   harness was built for human labels; the committed result was produced by a
   Claude (Haiku) judge standing in, because the manual pass was never run. So the
   validation shares failure modes with the thing it validates. It narrows the
   question; it does not settle it. `metric_validity_labels.jsonl` (the human file)
   does not exist in this repo; `metric_validity_labels_llm.jsonl` (the proxy) is
   what the numbers below come from.

Neither finding sinks the project — the V0→V2 improvement survives every matcher and
survives a chunk-invariant accuracy check — but both make specific committed claims
smaller than they read, and §4 names them.

---

## 1. The question

Two questions, one per layer of the eval:

- **Retrieval:** is "recall@5 = 0.64" a fact about retrieval, or a fact about the
  fuzzy matcher that scored it? If a stricter or looser matcher moves the number
  more than a real retrieval change does, the metric is measuring itself.
- **Generation:** the faithfulness number (0.929) is produced by a Haiku judge
  grading a Haiku-written answer — same model family on both sides. Does that judge
  discriminate a grounded answer from a fabricated one, or does it rubber-stamp its
  own family's output?

Both are the same underlying worry: **a metric I designed, validated against a
grader I also designed.** The only way out is to check each against something
external — ideally a human — and report the gap honestly, including when the gap
makes earlier numbers look worse.

## 2. Method

### Matchers (three, deliberately spanning strict→loose)

Same retrieval, three definitions of "this chunk supports the answer"
(`src/sec_rag/eval/metrics.py`, `configs/matchers.yaml`):

- **strict** — the gold span must appear as a verbatim substring in one chunk.
- **overlap** — ≥50% of the gold-evidence *tokens* appear anywhere in the chunk,
  order-free. This is the committed headline metric ("fuzzy(0.5)").
- **semantic** — cosine ≥ 0.62 between the gold span and a chunk (embedding-based).

### Labels (50 pairs, disagreement-stratified)

A random sample is nearly useless here: of 1,500 (question, chunk) pairs, **888 are
unanimous "no match"** — every matcher and any human agrees, so labeling them buys
no signal. The decision lives where the matchers **disagree** (596 pairs). So the
sample (`eval_results/metric_validity_sample.jsonl`, seed 13, committed) is
**oversampled on disagreement**:

| stratum | in sample | available |
|---|---|---|
| disagree (matchers split) | 30 | 596 |
| unanimous-yes | 10 | 16 |
| unanimous-no | 10 | 888 |

Category-stratified too (domain 19, metrics 13, novel 18). The labeling CLI
(`label_cli.py`) shows the question, gold answer, gold evidence, and chunk, and asks
one binary question — *does this chunk support the gold answer?* — **hiding the
matcher verdicts**, so a label can't be a rubber-stamp of them. Output is resumable
JSONL.

**What actually produced the labels.** Not a human. `label_auto.py` had a Claude
judge (`claude-haiku-4-5`) label all 50 pairs, to get a first result without waiting
on the manual pass. This is a proxy, and a compromised one by construction: an LLM
grading matchers is *exactly the kind of grader this study was built to check
against a human*. It is reported as an LLM proxy throughout, never as human
judgment.

### Scoring

Per matcher, treating the label as ground truth and a matcher "match" as the
positive class: **Cohen's κ** (with an approximate 95% CI), precision, recall, and
the confusion matrix (`label_score.py`, unit-tested on a hand-computed case,
`tests/test_label_score.py`). Artifact:
`eval_results/metric_validity_scores_20260731T140147Z.json`.

### The judge, and the second model that checked it

The faithfulness judge (Haiku, `generate/faithfulness.py`) was audited two ways:

- **Discrimination (constructed cases):** grounded claim → 1.0, hallucinated → 0.0,
  grounded refusal → 1.0; the parse/score arithmetic is unit-tested against
  adversarial judge output (miscounts clamped, embedded JSON, garbage → 0.0)
  (`tests/test_faithfulness.py`).
- **A second judge model:** an independent **Opus** adjudicator (different family
  from the Haiku judge, standing in for a human) re-scored 20 sampled (answer,
  sources, verdict) triples, verifying each figure by exact-string search against
  the *full* chunks (`docs/faithfulness-spotcheck.md`,
  `eval_results/faithfulness_spotcheck_20260730T232959Z.json`).

### Refusal instrumentation and the adversarial numeric set

The answer scorer (`eval/answer_accuracy.py`) separates three outcomes so a high
refusal rate can't quietly inflate accuracy: **refusal** (detected by an anchored
regex — "I cannot answer… from the provided sources" — not the mere presence of
"cannot"), **numeric-exact** (a deterministic normalizer), and **LLM-graded**. The
numeric normalizer is the piece most likely to lie on financial data, so it carries
an adversarial test set (`tests/test_answer_accuracy.py`): scale bridging
(`$1.2B` = `1,200 million`), a genuine 1000× wrong scale that must *stay* wrong
(`24.26` vs `2426`), accounting-parens negatives, percentages that must **never**
scale-bridge to a raw number (`1.9%` ≠ `1900`), and ~1% rounding tolerance. This is
the "grew 5% vs grew 25%" failure mode that fuzzy overlap is blind to, caught
deterministically.

## 3. Results

### The metric is a matcher choice (the number that doesn't flatter)

Identical V2 retrieval, three matchers, 150 questions
(`matcher_study_baseline_20260731T123250Z.json`):

| matcher | recall@5 | recall@10 |
|---|---|---|
| strict (substring) | **0.093** | 0.127 |
| overlap (fuzzy 0.5, committed) | **0.64** | 0.747 |
| semantic (cos ≥ 0.62) | **0.807** | 0.873 |

The choice of matcher moves recall@5 further (0.093 → 0.807) than any retrieval
change in the entire project does. "recall@5 = 0.64" is therefore only meaningful if
overlap tracks what a grader means by support — which is what the labels test.

### Which matcher agrees with the grader (LLM proxy, n=50)

Judge said "supports" on **19 / 50** pairs (0.38, on a disagreement-heavy sample).
Against that label, treating a matcher hit as the positive class:

| matcher | Cohen's κ | κ 95% CI | precision | recall | TP/FP/FN/TN |
|---|---|---|---|---|---|
| **overlap** | **0.674** | [0.47, 0.88] | 0.74 | 0.89 | 17 / 6 / 2 / 25 |
| strict | 0.579 | [0.33, 0.83] | **1.00** | 0.53 | 10 / 0 / 9 / 31 |
| semantic | 0.300 | [0.05, 0.55] | 0.50 | 0.89 | 17 / 17 / 2 / 14 |

The error modes are clean and opposite:

- **strict under-counts** — precision 1.00 (every substring hit is a real support)
  but recall 0.53 (misses 9 of 19 real supports). A trustworthy floor, not the rate.
- **semantic over-counts** — recall 0.89 but **precision 0.50**: half the pairs it
  fires on, the grader says do *not* support the answer. Its 0.807 recall@5 is
  inflated by false positives.
- **overlap sits between**, closest to the grader (κ 0.674, "substantial" on
  Landis–Koch), with precision **0.74** — so **~1 in 4 overlap "hits" is not a real
  support.** The committed 0.64 is a slight over-estimate; the true rate sits a
  little below it, and well above strict's 0.093.

One check against circularity: if the LLM judge simply favored the most "AI-like"
matcher, the embedding-based **semantic** matcher would win. It scored **lowest**
(κ 0.30) — the judge penalizes its over-matching — which makes the ranking harder to
dismiss as a pure judge/embedding artifact. That is the strongest thing this proxy
can say, and it is not much: at n=50 the overlap and strict CIs overlap almost
entirely, so **overlap ≈ strict is a statistical tie**; only semantic's over-matching
is clearly established.

### The judge discriminates, and is if anything too harsh

Opus vs Haiku on 20 faithfulness judgments: **agreement 19/20 (95%)**. The single
disagreement is a **false negative** — the judge scored a well-grounded answer 0.0
(record [10], CVS: three stated figures all verified present in source [1]).
Direction matters: the error is *harsh*, not lenient. Correcting it would *raise* the
sample mean 0.90 → 0.95. So on this sample there is **no evidence the committed
0.929 is inflated by a soft self-grader**; if anything the judge is marginally
conservative. Sample mean 0.90 is consistent with the committed 150-q 0.929.

### But faithfulness is grounding, not correctness

The judge measures whether the answer's claims are supported by the *retrieved*
sources — not whether they match the gold. Several answers are **faithful yet
wrong** because retrieval surfaced the wrong evidence (American Water Works: grounded
arithmetic on a mis-attributed prior-year input → −$1,257M vs gold −$1,561M). So
**0.929 faithful and 0.64 recall are consistent**: the system is well-grounded in
whatever it retrieves, and retrieves the wrong evidence about a third of the time. A
faithfulness number is not an accuracy number.

The accuracy number, measured directly (150 q, serving depth top_k=20,
`financebench_20260731T115740Z.json`): LLM-graded **0.74 of attempted (75/102)**,
**0.50 over all 150**, numeric-exact **0.85 (45/53)**, **refusal 0.32 (48/150)**.
That 0.50 is the honest "how often is the live answer right" figure, and it is far
below the 0.64 recall headline, let alone the 0.807 semantic ceiling.

## 4. What this means for every number previously committed here

Named, specifically:

- **`recall@5 = 0.64` (fuzzy) — a slight over-estimate.** Best-adjudicated of the
  three matchers, but precision 0.74 says ~1 in 4 hits isn't a real support, so the
  true rate is a little *below* 0.64. Report it as a bracket: **strict 0.093 floor →
  overlap 0.64 → semantic 0.807 ceiling**, with the truth near the overlap end, not
  the midpoint. Same for **recall@10 0.747**.
- **The embedding-model win (the project's central result) is real but overstated
  by the headline metric.** Crossing the V0→V2 change (2 models × 3 chunk sizes × 3
  matchers, `confound_study_20260731T170928Z.json`), the embedding component of the
  recall@5 gain is **+0.127 under overlap, +0.053 under semantic, and +0.000 under
  strict.** It is invisible to the strictest matcher at recall@5. The honest size of
  the embedding effect is closer to the **~+0.05 semantic** figure than the +0.127
  the headline shows.
- **~20% of the V0→V2 headline gain is metric inflation, not retrieval.** The
  +0.207 overlap recall@5 gain decomposes to embedding +0.127 + chunk +0.080, with
  interaction ≈ 0. Of the chunk +0.080, only **~+0.037 is real** (survives strict
  and semantic); the other **~+0.043 is overlap-metric inflation** — a 1024-token
  chunk clears the 50%-token bar more easily than a 512, independent of retrieval
  quality. The chunk-invariant arbiter, answer accuracy, rises **0.36 → 0.47
  (+0.11)** across the same change, embedding +0.08 / chunk +0.033 — so the real
  system gain is ~80% of the headline.
- **`tables@5 = 0.70` (from 0.32) inherits the same overstatement.** It is an
  overlap number driven by the embedding swap, and the embedding effect is
  overlap-weighted (above); the tables sub-metric was not separately crossed against
  strict/semantic, so read the 0.70 as overlap-optimistic, not as a strict rate.
- **`faithfulness = 0.929` stands as a grounding number** — audited, not inflated —
  **but must never be read as accuracy.** The correctness number is **0.50 over-all**.
- **The CI gate baseline (recall@5 0.6333 on the 30-q subset)** uses the overlap
  matcher — the best-adjudicated of the three by this study, and the defensible
  choice — but it inherits the same "slight over-estimate" property. It gates against
  *regression*, which is valid regardless of the absolute level.

Unaffected: cost and latency are direct measurements, not matcher/judge outputs, and
this study says nothing about them.

## 5. Limitations

Stated plainly, worst first:

1. **The labels are not human.** They are Haiku's. The study's entire premise is
   that an LLM grader might share failure modes with a learned metric — and then it
   used an LLM grader as ground truth. This is a screening result, not a verdict.
   The human harness exists and is committed; the human pass was not run.
2. **n = 50, single labeler, no inter-annotator agreement.** One grader, so there is
   no κ between annotators — the standard check on whether the "ground truth" is
   itself stable. At n=50 the CIs are wide (κ half-width ≈ ±0.2, first-order) and
   the sample is *oversampled on disagreement*, so these κ/precision/recall describe
   the **contested region**, not the population; the population agreement would be
   higher and is not what's quoted.
3. **overlap vs strict is not distinguishable here.** Their CIs overlap almost
   entirely ([0.47, 0.88] vs [0.33, 0.83]). The study separates *gross* differences
   (semantic's over-matching) but not the fine one that would actually pick between
   the two defensible matchers.
4. **The faithfulness adjudicator is also an LLM** (Opus). A different family from
   the judge, which is worth something, but still not a human. 19/20 is one sample,
   one run.
5. **Single runs, no confidence intervals on the recall grid itself.** The
   confound grid is one pass per cell (temperature 0 for determinism, but no
   resampling), so the recall deltas carry no error bars — only the label study
   does, and those are approximate.

## 6. What I would do with more budget

In rough order of how much it would change the conclusions:

- **A few hundred human labels, ≥2 annotators, report inter-annotator κ.** This is
  the one that converts "overlap agrees best (proxy)" into a defensible pick, and
  the only one that resolves the overlap-vs-strict tie. Everything else is secondary
  to replacing the LLM labeler with people.
- **Bootstrap CIs on the recall numbers** by resampling questions, and run the grid
  under ≥3 seeds — so the +0.037 "real chunk gain" and the +0.05 embedding effect
  come with error bars instead of point estimates.
- **Make the judge emit per-claim verdicts** (claim text + supported bool), not just
  counts. The one judge error (record [10]) was invisible in the aggregate and only
  found by re-reading sources; per-claim output makes the audit cheap and repeatable.
- **Persist per-question detail in the main eval run** so a spot-check can *replay*
  the exact committed answers rather than regenerate them.
- **A cross-model judge panel** (not one Haiku) for both faithfulness and accuracy,
  with disagreement surfaced — the same diversity idea as the matcher study, applied
  to the grader.
- **A human-labeled correctness set for the live EDGAR path**, which currently has a
  harness but no committed number at all.

---

*Artifacts referenced: `eval_results/metric_validity_{sample,labels_llm,scores_*}.jsonl/json`,
`matcher_study_{baseline,ablations}_*.json`, `confound_study_20260731T170928Z.json`,
`faithfulness_spotcheck_20260730T232959Z.json`, `financebench_20260731T115740Z.json`.
Reproduce: `docs/metric-validity.md` §Reproduce (matcher study),
`docs/faithfulness-spotcheck.md` (judge audit), `docs/depth-round.md` (the confound
crossing). Every number here traces to a committed file.*
