# Is my own evaluation metric valid? A study of the matcher and the judge

*A report on what turned out to be true — not a defense of the project.*

Every retrieval number in this repo is an **evidence-hit rate**: the fraction of
questions for which a retrieved chunk "contains" the gold evidence, under some
matcher. Every answer-quality number leans on an **LLM judge**. Both are things I
built, so both are things I could have gotten wrong in a way that flatters the
result. This study checks them against 50 of my own hand labels and a second judge
model, and reports the gaps — including the ones that make earlier numbers worse.

Three findings up front, the unflattering ones first:

1. **My earlier "the metric is validated" claim did not survive human labeling.** An
   earlier pass had a Claude (Haiku) judge stand in for the human labeler; it
   reported the committed fuzzy matcher as **best-adjudicated, Cohen's κ 0.67
   ("substantial")**. I then labeled the same 50 pairs myself. The fuzzy matcher
   drops to **κ 0.18 ("slight")** and is no longer even top-ranked — and **all three
   matchers' κ confidence intervals include zero**, so at n=50 I cannot show any of
   them agrees with me better than chance. The clean "overlap is validated" result
   was an artifact of the LLM labeler, and this repo shipped it once.
2. **The LLM labeler was a biased stand-in, not just a noisy one.** Human-vs-Haiku
   agreement is only **κ 0.37 (fair)**, and the disagreement is one-directional: I
   called "supports" on **34/50** pairs, Haiku on **19/50**, and **16 of the 17
   disagreements are cases I accept and Haiku rejected.** Haiku was systematically
   stricter, which is exactly what produced the tidy "overlap over-counts / semantic
   over-counts" story that my own labels do not reproduce.
3. **The headline retrieval metric is still mostly a property of the matcher.**
   Identical V2 retrieval reads as recall@5 = **0.093 / 0.64 / 0.807** depending only
   on which matcher scores it — a ~9× spread. The committed headline (0.64) is one
   point in that range, and the human labels do not pin it there; they show fuzzy
   correlates only slightly with what I mean by "support."

None of this sinks the project — the V0→V2 improvement survives every matcher and a
chunk-invariant accuracy check (§3, §4) — but the *metric-validity* claim is much
smaller than it read, and §4 names every number it touches.

---

## 1. The question

Two questions, one per layer of the eval:

- **Retrieval:** is "recall@5 = 0.64" a fact about retrieval, or a fact about the
  fuzzy matcher that scored it? If the choice of matcher moves the number more than a
  real retrieval change does, the metric is partly measuring itself — and the only
  fix is to check the matcher against a human and see how well it tracks.
- **Generation:** the faithfulness number (0.929) is produced by a Haiku judge
  grading a Haiku-written answer — same model family on both sides. Does that judge
  discriminate a grounded answer from a fabricated one, or does it rubber-stamp its
  own family's output?

Both are the same worry: **a metric I designed, validated against a grader I also
designed.** The way out is to check each against something external — here, my own
labels for the matcher, and a different-family judge for the generation — and report
the gap honestly, including when the gap makes earlier numbers look worse. It did.

## 2. Method

### Matchers (three, deliberately spanning strict → loose)

Same retrieval, three definitions of "this chunk supports the answer"
(`src/sec_rag/eval/metrics.py`, `configs/matchers.yaml`):

- **strict** — the gold span must appear as a verbatim substring in one chunk.
- **overlap** — ≥50% of the gold-evidence *tokens* appear anywhere in the chunk,
  order-free. This is the committed headline metric ("fuzzy(0.5)").
- **semantic** — cosine ≥ 0.62 between the gold span and a chunk (embedding-based).

### Labels (50 pairs, disagreement-stratified, hand-labeled by me)

A random sample is nearly useless here: of 1,500 (question, chunk) pairs, **888 are
unanimous "no match"** — every matcher and any human agrees, so labeling them buys no
signal. The decision lives where the matchers **disagree** (596 pairs). So the sample
(`eval_results/metric_validity_sample.jsonl`, seed 13, committed) is **oversampled on
disagreement**:

| stratum | in sample | available |
|---|---|---|
| disagree (matchers split) | 30 | 596 |
| unanimous-yes | 10 | 16 |
| unanimous-no | 10 | 888 |

Category-stratified too (domain 19, metrics 13, novel 18). I rendered each pair —
question, gold answer, gold evidence, full retrieved chunk — with **the matcher
verdicts and the stratum hidden**, and made one binary call per pair: *does this
chunk contain the evidence needed to produce the gold answer?* Output is
`eval_results/metric_validity_labels.jsonl` (`supports` ∈ {0,1}), scored by
`label_score.py`.

**What this replaces.** An earlier pass (`label_auto.py`) had `claude-haiku-4-5`
label all 50 to get a first result without waiting on the manual pass. That was a
proxy, and a compromised one by construction — an LLM grading matchers is *exactly
the kind of grader this study was built to check against a human*. The proxy result
(`metric_validity_scores_20260731T140147Z.json`) is kept only for the contrast in §3;
the human labels supersede it.

**A false start, disclosed.** My first labeling pass was a rubber-stamp: I marked
46/50 "supports" (including 10/10 of the unanimous-no stratum), which drove every
matcher's κ *negative* — the statistical signature of a degenerate, near-single-class
label set, not a real result. I discarded it and re-labeled reading each chunk
against the gold evidence. The committed labels are the second pass (34 yes / 16 no).
This is disclosed because it is itself evidence of how fragile single-annotator
labeling is (§5).

### Scoring

Per matcher, treating my label as ground truth and a matcher "match" as the positive
class: **Cohen's κ** (with an approximate first-order 95% CI), precision, recall, and
the confusion matrix (`label_score.py`, unit-tested on a hand-computed case,
`tests/test_label_score.py`). Artifact:
`eval_results/metric_validity_scores_20260801T211535Z.json`.

### The judge, and the second model that checked it

The faithfulness judge (Haiku, `generate/faithfulness.py`) was audited two ways:

- **Discrimination (constructed cases):** grounded claim → 1.0, hallucinated → 0.0,
  grounded refusal → 1.0; the parse/score arithmetic is unit-tested against
  adversarial judge output (miscounts clamped, embedded JSON, garbage → 0.0)
  (`tests/test_faithfulness.py`).
- **A second judge model:** an independent **Opus** adjudicator (different family from
  the Haiku judge, standing in for a human) re-scored 20 sampled (answer, sources,
  verdict) triples, verifying each figure by exact-string search against the *full*
  chunks (`docs/faithfulness-spotcheck.md`,
  `eval_results/faithfulness_spotcheck_20260730T232959Z.json`).

### Refusal instrumentation and the adversarial numeric set

The answer scorer (`eval/answer_accuracy.py`) separates three outcomes so a high
refusal rate can't quietly inflate accuracy: **refusal** (an anchored regex — "I
cannot answer … from the provided sources" — not the mere presence of "cannot"),
**numeric-exact** (a deterministic normalizer), and **LLM-graded**. The numeric
normalizer is the piece most likely to lie on financial data, so it carries an
adversarial test set (`tests/test_answer_accuracy.py`): scale bridging (`$1.2B` =
`1,200 million`), a genuine 1000× wrong scale that must *stay* wrong (`24.26` vs
`2426`), accounting-parens negatives, percentages that must **never** scale-bridge to
a raw number (`1.9%` ≠ `1900`), and ~1% rounding tolerance. This is the "grew 5% vs
grew 25%" failure mode that fuzzy overlap is blind to, caught deterministically.

## 3. Results

### The metric is a matcher choice (the number that doesn't flatter)

Identical V2 retrieval, three matchers, 150 questions
(`matcher_study_baseline_20260731T123250Z.json`):

| matcher | recall@5 | recall@10 |
|---|---|---|
| strict (substring) | **0.093** | 0.127 |
| overlap (fuzzy 0.5, committed) | **0.64** | 0.747 |
| semantic (cos ≥ 0.62) | **0.807** | 0.873 |

The choice of matcher moves recall@5 further (0.093 → 0.807) than any retrieval change
in the entire project does. "recall@5 = 0.64" is therefore only meaningful if overlap
tracks what I mean by support — which the labels test next, and largely fail to
confirm.

### Which matcher agrees with me (50 human labels — supersedes the LLM proxy)

I labeled "supports" on **34/50** pairs (0.68, on a disagreement-heavy sample).
Against that, treating a matcher hit as the positive class
(`metric_validity_scores_20260801T211535Z.json`):

| matcher | Cohen's κ | κ 95% CI | precision | recall | TP/FP/FN/TN | Landis–Koch |
|---|---|---|---|---|---|---|
| strict | **0.211** | [−0.02, 0.44] | **1.00** | 0.29 | 10 / 0 / 24 / 16 | fair |
| overlap | **0.184** | [−0.08, 0.45] | 0.78 | 0.53 | 18 / 5 / 16 / 11 | slight |
| semantic | **0.081** | [−0.23, 0.39] | 0.71 | 0.71 | 24 / 10 / 10 / 6 | slight |

Read against the LLM proxy it replaced — which reported **overlap κ 0.674, strict
0.579, semantic 0.300, overlap best** — almost nothing holds:

- **The ranking changed and the magnitudes collapsed.** Under me, strict is nominally
  best (κ 0.21) and overlap is *second* (0.18), not first; the proxy's "substantial"
  0.67 becomes "slight" 0.18. And **every κ CI includes zero** — at n=50 on this
  contested sample I cannot establish that any matcher beats chance. The proxy's clean
  verdict was manufactured by the LLM labeler.
- **The dominant error is under-counting, not over-counting.** The proxy's story was
  "semantic over-counts (precision 0.50)." My labels don't show that — semantic
  precision is **0.71**. What they show instead is that *all* matchers miss support I
  see: recall **0.29 / 0.53 / 0.71** for strict / overlap / semantic. The matchers are
  keyed to FinanceBench's one narrow gold span, so they miss evidence that supports the
  answer but lives elsewhere in the chunk. I labeled **8 of the 10 unanimous-no pairs
  "supports"** for exactly this reason (or, in part, my own leniency — see §5).
- **What survives:** strict's **precision 1.00** — when a gold span appears verbatim,
  I always agree it's real support. Strict is a trustworthy *floor*, confirmed. Its
  recall 0.29 confirms it is only that.
- **semantic's balance is a coincidence.** Its precision ≈ recall ≈ 0.71 looks even,
  but it matches my *marginal* yes-rate (0.68), not my *pair-by-pair* calls — hence
  κ 0.08, near chance. Matching the rate is not agreeing.

One thing the proxy got backwards is worth stating: it penalized the embedding-based
**semantic** matcher hardest (κ 0.30, "over-matches"); my labels rank it no worse than
overlap on precision and *better* on recall. If anything the LLM judge was biased
*against* the most embedding-like matcher, not for it — the opposite of the
circularity I'd worried about, and a different bias than I expected.

**How good a stand-in was the LLM labeler?** Directly: human-vs-Haiku raw agreement
**33/50 (0.66)**, **κ 0.37 (fair)**, CI [0.13, 0.62]. Not noise, but not a substitute
— and **biased**: I say "supports" 0.68 of the time, Haiku 0.38, and 16 of 17
disagreements are mine-yes/Haiku-no. The proxy under-credited support, which is the
single reason it produced a flattering, over-confident matcher ranking.

### The judge discriminates, and is if anything too harsh

Opus vs Haiku on 20 faithfulness judgments: **agreement 19/20 (95%)**
(`docs/faithfulness-spotcheck.md`). The single disagreement is a **false negative** —
the judge scored a well-grounded answer 0.0 (record [10], CVS: three stated figures
all verified present in source [1]). Direction matters: the error is *harsh*, not
lenient. Correcting it would *raise* the sample mean 0.90 → 0.95. So on this sample
there is **no evidence the committed 0.929 is inflated by a soft self-grader**; if
anything the judge is marginally conservative, and the sample mean 0.90 is consistent
with the committed 150-q 0.929. (This audit is unaffected by the matcher finding
above — different layer, different second-model.)

### But faithfulness is grounding, not correctness

The judge measures whether the answer's claims are supported by the *retrieved*
sources — not whether they match the gold. Several answers are **faithful yet wrong**
because retrieval surfaced the wrong evidence (American Water Works: grounded
arithmetic on a mis-attributed prior-year input → −$1,257M vs gold −$1,561M). So
**0.929 faithful and 0.64 recall are consistent**: the system is well-grounded in
whatever it retrieves, and retrieves the wrong evidence about a third of the time. A
faithfulness number is not an accuracy number.

The accuracy number, measured directly (150 q, serving depth top_k=20,
`financebench_20260731T115740Z.json`): LLM-graded **0.74 of attempted (75/102)**,
**0.50 over all 150**, numeric-exact **0.85 (45/53)**, **refusal 0.32 (48/150)**. That
0.50 is the honest "how often is the live answer right" figure, and it sits below the
0.64 fuzzy recall headline, well below the 0.807 semantic ceiling.

## 4. What this means for every number previously committed here

Named, specifically:

- **The metric-validity claim itself — retracted as previously stated.** The repo
  (README, `docs/metric-validity.md`, the prior `metric-validity-study.md`) said the
  fuzzy matcher was *best-adjudicated at κ 0.67 ("substantial")*. That was the **LLM
  proxy**. Under my labels overlap is **κ 0.18 ("slight")**, nominally **second**
  (behind strict, ahead of semantic), and no matcher's κ clears zero at n=50. The correct statement is: *on the contested
  pairs, no matcher is a demonstrably good proxy for my judgment; strict is a
  high-precision floor, and the fuzzy matcher correlates only slightly.* Every place
  that quoted "κ 0.67 / best-adjudicated" is corrected in this pass.
- **`recall@5 = 0.64` (fuzzy) — a weak proxy, reported as a bracket.** overlap agrees
  with me only slightly (κ 0.18); it both admits non-support (precision 0.78) and
  misses real support (recall 0.53), so 0.64 is a noisy indicator, not a calibrated
  point. Report it as **strict 0.093 floor → overlap 0.64 → semantic 0.807 ceiling**;
  the human labels *weaken*, not strengthen, the case that 0.64 is the "true" support
  rate. Same for **recall@10 0.747**.
- **The embedding-model win (the project's central result) is real but overstated by
  the headline metric — and the human study removes the excuse for the overstatement.**
  Crossing V0→V2 (2 models × 3 chunk sizes × 3 matchers,
  `confound_study_20260731T170928Z.json`), the embedding component of the recall@5 gain
  is **+0.127 (overlap), +0.053 (semantic), +0.000 (strict)**. Invisible to the
  strictest matcher; I previously leaned on "overlap is the best-adjudicated matcher"
  to prefer the +0.127. That crutch is gone — overlap is *not* best-adjudicated — so the
  honest size of the embedding effect is the **~+0.05 semantic** figure, corroborated
  by answer accuracy (+0.08, chunk-invariant), not the +0.127 the headline shows.
- **~20% of the V0→V2 headline gain is metric inflation, not retrieval.** The +0.207
  overlap recall@5 gain decomposes to embedding +0.127 + chunk +0.080, interaction ≈ 0.
  Of the chunk +0.080, only **~+0.037 is real** (survives strict and semantic); the
  other **~+0.043 is overlap-metric inflation** — a 1024-token chunk clears the
  50%-token bar more easily than a 512, independent of retrieval quality. The
  chunk-invariant arbiter, answer accuracy, rises **0.36 → 0.47 (+0.113)**, embedding
  +0.08 / chunk +0.033 — so the real system gain is ~80% of the headline. This
  decomposition is unchanged by the label study; what changed is that I can no longer
  point to a validated matcher to defend the overlap column.
- **`tables@5 = 0.70` (from 0.32) inherits the same overstatement** — an overlap number
  driven by the (overlap-weighted) embedding swap, never separately crossed against
  strict/semantic. Read 0.70 as overlap-optimistic, not a strict rate.
- **`faithfulness = 0.929` stands as a grounding number** — audited (19/20, harsh-not-
  lenient), not inflated — **but must never be read as accuracy.** The correctness
  number is **0.50 over-all**. Unaffected by the matcher finding.
- **The CI gate baseline (recall@5 0.6333 on the 30-q subset) uses the overlap matcher,
  and its stated justification was wrong.** The gate note claimed overlap "agreed best
  with human/LLM labels (κ 0.67)"; the human κ is 0.18 and overlap is not best. The
  gate is still a valid *regression* detector — it flags a drop against a frozen
  baseline regardless of the absolute matcher — but "best-adjudicated" is not the reason
  to keep overlap. Retained for continuity and low cost; corrected in the README.

Unaffected: cost and latency are direct measurements, not matcher/judge outputs, and
this study says nothing about them.

## 5. Limitations

Stated plainly, worst first:

1. **n = 50, single annotator (me), no inter-annotator agreement.** One labeler, so
   there is no κ between annotators — the standard check on whether the "ground truth"
   is itself stable. It demonstrably is not: my *first* pass was a degenerate
   rubber-stamp I had to discard, and even on the committed pass the assistant flagged
   pair 15 (a 3M special-items narrative that cites a 2022 bankruptcy and carries no
   PP&E figure) as clearly non-supporting, while I labeled it "supports." Two readers
   already diverge on the hard pairs — which is the entire reason inter-annotator κ
   exists, and I don't have it.
2. **The κ CIs are wide and all cross zero.** At n=50 the first-order κ half-width is
   ≈ ±0.2. This sample can separate *gross* differences but not the fine ones that would
   actually rank the three matchers — and here it cannot separate any of them from
   chance. The point-estimate ranking (strict > overlap > semantic) is indicative, not
   established.
3. **The sample is oversampled on disagreement, on purpose.** So these κ / precision /
   recall characterize the **contested region** — the pairs where matchers differ — not
   the whole population. The 888 unanimous-no pairs (all easy "no") are under-
   represented, so the *population* agreement would be higher than the numbers here.
   This answers "when the matchers disagree, who's right?", which governs the recall gap
   — but it is not a population agreement rate and shouldn't be quoted as one.
4. **The faithfulness adjudicator is also an LLM** (Opus), a different family from the
   Haiku judge, standing in for a human. 19/20 is one sample, one run.
5. **Single runs, no confidence intervals on the recall grid itself.** The confound and
   matcher grids are one pass per cell (temperature 0 for determinism, no resampling),
   so the recall deltas carry no error bars — only the label study does, and those are
   approximate.

## 6. What I would do with more budget

In rough order of how much it would change the conclusions:

- **A few hundred labels, ≥2 independent annotators, report inter-annotator κ.** This
  is the one that turns "no matcher clearly beats chance (n=50)" into a real ranking,
  and the only one that establishes whether the ground truth is stable at all. Given
  §5.1, adding annotators matters as much as adding pairs.
- **Balance the sample, or estimate the population rate.** Re-weight the strata back to
  the 888/596/16 population so the κ reflects the whole pair distribution, not just the
  contested slice — then the "true recall@5" bracket could be tightened instead of left
  at [0.093, 0.807].
- **Bootstrap CIs on the recall numbers** by resampling questions, and run the grids
  under ≥3 seeds — so the +0.037 "real chunk gain" and the ~+0.05 embedding effect come
  with error bars, not point estimates.
- **Make the judge emit per-claim verdicts** (claim text + supported bool), not just
  counts. The one judge error (record [10]) was invisible in the aggregate and only
  found by re-reading sources; per-claim output makes the audit cheap and repeatable.
- **A cross-model judge panel** (not one Haiku) for both faithfulness and accuracy, with
  disagreement surfaced — the same diversity idea as the matcher study, applied to the
  grader. The one clear lesson here is that a single LLM grader (Haiku) was a biased
  proxy; a panel plus a human anchor is the fix.
- **A human-labeled correctness set for the live EDGAR path**, which currently has a
  harness but no committed number at all.

---

*Artifacts: `eval_results/metric_validity_{sample,labels,scores_20260801T211535Z}` (human,
authoritative), `metric_validity_{labels_llm,scores_20260731T140147Z}` (LLM proxy, superseded),
`matcher_study_baseline_20260731T123250Z.json`, `confound_study_20260731T170928Z.json`,
`faithfulness_spotcheck_20260730T232959Z.json`, `financebench_20260731T115740Z.json`.
Reproduce: `docs/metric-validity.md` §Reproduce (label the sample → score), `docs/faithfulness-
spotcheck.md` (judge audit), `docs/depth-round.md` (the confound crossing). Every number here
traces to a committed file.*
