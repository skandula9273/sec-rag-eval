# Metric validity — which evidence matcher agrees with a human?

Every recall number in this repo is an *evidence-hit rate*: the fraction of questions
whose gold evidence a retrieved chunk "contains", under some matcher. The matcher
S3 study (`eval/matcher_study.py`, `docs/depth-round.md`) showed the headline is
mostly a property of the **matcher**, not the retriever — the same V2 retrieval scores:

| matcher | recall@5 | recall@10 |
|---|---|---|
| strict (exact substring) | 0.093 | 0.127 |
| overlap (fuzzy 0.5, the committed metric) | 0.64 | 0.747 |
| semantic (cosine ≥ 0.62 over spans) | 0.807 | 0.873 |

So "recall@5 = 0.64" is only meaningful if the overlap matcher tracks what a human
means by "this chunk supports the answer". This study measures that directly:
**human labels vs each matcher**, via Cohen's kappa, precision, and recall.

> **Status: LLM-ADJUDICATED results below (a proxy); human pass still supersedes.**
> The harness was built for HUMAN labels, and that remains the gold standard. On
> request, the 50 pairs were also labeled by a Claude judge (`label_auto.py`, judge
> = `claude-haiku-4-5`) to produce a first result without waiting on manual labeling.
> Read these as an LLM proxy, **not** human judgement: an LLM grading matchers shares
> failure modes with the learned matchers and is exactly the kind of judge this study
> was meant to check against a human. The LLM labels live in a separate file
> (`metric_validity_labels_llm.jsonl`); a human pass writes `metric_validity_labels
> .jsonl` and overrides.

## Method

**Sample (`eval_results/metric_validity_sample.jsonl`, seed 13, committed).** 50
(question, retrieved-chunk) pairs drawn from the top-10 retrieved per question on the
live V2 config. A *random* sample is ~useless here: 888/1500 pairs are unanimous "no
match" (a random chunk nothing matches) — easy agreements that spend the labeling
budget on nothing. The signal is where the three matchers **disagree** (596 pairs),
because those are the pairs that decide which matcher is right. So the sample is
**disagreement-oversampled** and category-stratified:

| stratum | in sample | available |
|---|---|---|
| disagree (matchers split) | 30 | 596 |
| unanimous-yes (all 3 match) | 10 | 16 |
| unanimous-no (all 3 miss) | 10 | 888 |

Across categories: domain-relevant 19, metrics-generated 13, novel-generated 18.
The per-pair matcher verdicts are stored in the sample (for scoring) but the labeling
CLI hides them, so the human label is independent, not a rubber-stamp.

**Labeling (`eval/label_cli.py`).** For each pair it shows the question, gold answer,
gold evidence, and the chunk, and asks one binary question — *does this chunk support
the gold answer?* — plus an optional note. Output appends to
`eval_results/metric_validity_labels.jsonl` (resumable). No auto-labeling.

**Scoring (`eval/label_score.py`).** Per matcher, treating the human label as ground
truth and a matcher "match" as the positive class: Cohen's kappa (with an approximate
95% CI), precision, recall, and the confusion matrix. Pure, unit-tested (kappa=0.40 /
precision=0.60 / recall=0.75 on a hand-computed case, `tests/test_label_score.py`).

## The honest statement about n = 50

Fifty labels is a *screening* sample, not a definitive one. Two consequences, stated
up front so the numbers aren't over-read:

1. **Wide intervals.** A binary agreement proportion at n=50 carries a 95% CI of
   roughly **±0.11–0.14** (half-width `1.96·√(p(1−p)/50)`). Cohen's kappa is wider —
   its first-order 95% CI at this n is roughly **±0.2** (`1.96·√(po(1−po)/n)/(1−pe)`).
   So this can separate *gross* differences (kappa ≈ 0.2 vs ≈ 0.7) but **not** fine
   ones (e.g. overlap 0.55 vs 0.60). Read rankings as indicative; a definitive pick
   would need a few hundred labels, ideally ≥2 annotators + inter-annotator kappa.
   *Observed below:* overlap [0.47, 0.88] and strict [0.33, 0.83] overlap heavily —
   **overlap ≈ strict is not distinguishable at n=50**; only semantic (CI upper bound
   0.55) is clearly separated below them.

2. **The sample is oversampled on disagreement, on purpose.** So the kappa /
   precision / recall here characterize the **contested region** — the pairs where
   matchers differ — not the whole pair population. Because unanimous pairs (mostly
   easy "no", which every matcher and a human get right) are under-represented, the
   *population* kappa would be **higher** than what this sample reports. This study
   answers "when the matchers disagree, who's right?", which is exactly the question
   that governs the recall gap between them — but it is not a population agreement
   rate, and shouldn't be quoted as one.

## Results (LLM-adjudicated proxy, judge = claude-haiku-4-5, n=50)

Source: `eval_results/metric_validity_scores_20260731T140147Z.json`. Judge said
"supports" on **19 / 50** pairs (0.38 — on this disagreement-heavy sample). Ranked
by agreement with the judge:

| matcher | Cohen's κ | κ 95% CI | precision | recall | TP / FP / FN / TN |
|---|---|---|---|---|---|
| **overlap** (fuzzy 0.5) | **0.674** | [0.47, 0.88] | 0.74 | 0.89 | 17 / 6 / 2 / 25 |
| strict (substring) | 0.579 | [0.33, 0.83] | **1.00** | 0.53 | 10 / 0 / 9 / 31 |
| semantic (cos ≥ 0.62) | 0.300 | [0.05, 0.55] | **0.50** | 0.89 | 17 / 17 / 2 / 14 |

**Best agreement with the judge:** overlap (κ 0.674 — "substantial" on Landis–Koch).
The error modes are clean and opposite:
- **strict UNDER-counts** — precision 1.00 (every substring hit is a real support) but
  recall 0.53 (it misses 9 of 19 real supports). A hard, trustworthy floor.
- **semantic OVER-counts** — recall 0.89 but precision **0.50**: it fires on 34 pairs,
  **half of which the judge says do NOT support the answer**. Its high recall@k is
  inflated by false positives.
- **overlap sits between**, closest to the judge, with a moderate false-positive rate
  (precision 0.74 on contested pairs).

**A note on circularity:** if an LLM judge simply favored the most "AI-like" matcher,
the embedding-based **semantic** matcher would score *highest*. It scored *lowest*
(κ 0.30) — the judge penalizes its over-matching — which makes the ranking harder to
dismiss as a judge/embedding artifact.

## What this implies about the committed recall numbers

Under the LLM-adjudicated proxy, **overlap agrees best** — so the committed metric is
the best-supported of the three, with these consequences (all subject to the LLM-proxy
and n=50 caveats above):

- The committed **`recall@5 = 0.64`** (and the whole `eval_results/` history, all
  overlap/fuzzy) is the closest-to-adjudicator number, with **strict 0.093 as a
  pessimistic floor and semantic 0.807 as an optimistic ceiling**. The true rate is
  *not* the midpoint — it sits near the overlap end.
- overlap is a **slight over-estimate**: precision 0.74 on contested pairs means ~1 in
  4 overlap "hits" isn't judged a real support, so true recall is likely a little
  *below* 0.64 — but far above strict's 0.093, and clearly not as high as semantic's
  0.807 (which is half false positives).
- **Cross-check with the S3 ablations:** the chunk-size win survives all matchers
  (robust); the embedding-model win was overlap-inflated (+0.127 overlap vs +0.053
  semantic vs +0.000 strict). Since overlap is the best-adjudicated matcher, the
  embedding win is *real but overstated* by the headline — the ~+0.05 semantic figure
  is the more honest size, and by the high-precision strict matcher it's invisible at
  recall@5.

**Caveat that governs all of the above:** this is a Claude judge, not a human, on 50
disagreement-heavy pairs. overlap ≈ strict is a statistical tie here (§ n=50); only
semantic's over-matching is clearly established. A human pass (write
`metric_validity_labels.jsonl`, re-run `label_score.py`) supersedes these numbers and
is the honest way to *pick* a matcher — this proxy only narrows the field.

## Reproduce

```bash
python -m sec_rag.eval.label_sample     # (already run; sample committed, seed 13)

# Human path (gold standard; supersedes the LLM proxy):
python -m sec_rag.eval.label_cli        # a human labels the 50 pairs (resumable)
python -m sec_rag.eval.label_score      # -> eval_results/metric_validity_scores_*.json

# LLM-proxy path (what produced the results above):
python -m sec_rag.eval.label_auto       # Claude judge -> metric_validity_labels_llm.jsonl
python -m sec_rag.eval.label_score --labels eval_results/metric_validity_labels_llm.jsonl
```
