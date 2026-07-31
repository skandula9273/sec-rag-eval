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

> **Status: harness built; awaiting human labels.** The results tables below are
> intentionally empty — they are filled by running `label_score.py` after a human
> labels the sample. No labels were generated automatically; an LLM-labeled study
> would just measure the LLM, which is the thing under suspicion.

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

2. **The sample is oversampled on disagreement, on purpose.** So the kappa /
   precision / recall here characterize the **contested region** — the pairs where
   matchers differ — not the whole pair population. Because unanimous pairs (mostly
   easy "no", which every matcher and a human get right) are under-represented, the
   *population* kappa would be **higher** than what this sample reports. This study
   answers "when the matchers disagree, who's right?", which is exactly the question
   that governs the recall gap between them — but it is not a population agreement
   rate, and shouldn't be quoted as one.

## Results

<!-- Fill from: python -m sec_rag.eval.label_score  (after labeling) -->

| matcher | Cohen's κ | κ 95% CI | precision | recall | TP / FP / FN / TN |
|---|---|---|---|---|---|
| strict | _pending_ | | | | |
| overlap | _pending_ | | | | |
| semantic | _pending_ | | | | |

**Best agreement with human judgement:** _pending labeling._
**n labeled:** _pending._ **Human "supports" rate on the sample:** _pending._

## What this will imply about the committed recall numbers

Once labeled, the kappas convert the S3 *dependence* into a *validity* judgement:

- If **overlap** has the highest kappa → the committed `recall@5 = 0.64` (and the
  whole `eval_results/` history, all overlap/fuzzy) is the best-supported number, and
  strict (0.093) / semantic (0.807) are the pessimistic / optimistic brackets.
- If **semantic** wins → the committed numbers systematically *under*-count real hits,
  and 0.64 is a floor, not the estimate.
- If **strict** wins → nearly every committed recall number is inflated, and the true
  hit rate is near 0.1.
- Cross-check with the S3 ablation finding: the chunk-size win survives all matchers
  (robust) but the embedding-model win is overlap-inflated (+0.127 overlap vs +0.053
  semantic vs +0.000 strict). Whichever matcher wins here tells us which of *those*
  deltas to believe.

No matcher is declared "best" until the labels exist — picking one before measuring
human agreement is exactly the mistake this study is built to avoid.

## Reproduce

```bash
python -m sec_rag.eval.label_sample     # (already run; sample committed, seed 13)
python -m sec_rag.eval.label_cli        # a human labels the 50 pairs (resumable)
python -m sec_rag.eval.label_score      # kappa/precision/recall/confusion -> eval_results/
```
