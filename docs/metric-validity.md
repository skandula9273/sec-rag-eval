# Metric validity — which evidence matcher agrees with a human?

Every recall number in this repo is an *evidence-hit rate*: the fraction of questions
whose gold evidence a retrieved chunk "contains", under some matcher. The matcher
study (`eval/matcher_study.py`, `docs/depth-round.md`) showed the headline is mostly a
property of the **matcher**, not the retriever — the same V2 retrieval scores:

| matcher | recall@5 | recall@10 |
|---|---|---|
| strict (exact substring) | 0.093 | 0.127 |
| overlap (fuzzy 0.5, the committed metric) | 0.64 | 0.747 |
| semantic (cosine ≥ 0.62 over spans) | 0.807 | 0.873 |

So "recall@5 = 0.64" is only meaningful if the overlap matcher tracks what a human
means by "this chunk supports the answer". This doc records the method, the reproduce
steps, and the headline result. **The full narrative — including what the result does
to every committed number — is in [`metric-validity-study.md`](metric-validity-study.md).**

> **Status: HUMAN-LABELED (authoritative).** The 50 pairs were hand-labeled by the
> owner (`metric_validity_labels.jsonl`), and those labels supersede the earlier
> Claude-Haiku proxy (`label_auto.py` → `metric_validity_labels_llm.jsonl`), kept only
> for the contrast below. **The human result overturns the proxy's:** the proxy called
> the committed fuzzy matcher best-adjudicated at κ 0.674 ("substantial"); the human
> puts it at κ 0.184 ("slight"), second of three (behind strict), and no matcher's κ
> clears zero at n=50. Treat the proxy numbers as retracted; treat these as a screening result, not a
> verdict (n=50, single annotator — see the limitations in the study writeup).

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

Across categories: domain-relevant 19, metrics-generated 13, novel-generated 18. The
per-pair matcher verdicts are stored in the sample (for scoring) but the labeling CLI
hides them, so the human label is independent, not a rubber-stamp.

**Labeling (`eval/label_cli.py`).** For each pair it shows the question, gold answer,
gold evidence, and the chunk, and asks one binary question — *does this chunk support
the gold answer?* — plus an optional note. Output appends to
`eval_results/metric_validity_labels.jsonl` (resumable). No auto-labeling.

**Scoring (`eval/label_score.py`).** Per matcher, treating the human label as ground
truth and a matcher "match" as the positive class: Cohen's kappa (with an approximate
95% CI), precision, recall, and the confusion matrix. Pure, unit-tested (kappa=0.40 /
precision=0.60 / recall=0.75 on a hand-computed case, `tests/test_label_score.py`).

## The honest statement about n = 50

Fifty labels is a *screening* sample, not a definitive one, and there is one annotator
(no inter-annotator agreement). Two consequences, stated up front:

1. **Wide intervals.** Cohen's kappa at this n carries a first-order 95% CI of roughly
   **±0.2** (`1.96·√(po(1−po)/n)/(1−pe)`). *Observed below:* all three matchers' κ CIs
   include zero — so at n=50 this sample cannot show that **any** matcher agrees with
   the human better than chance. The point-estimate ranking is indicative, not
   established.
2. **The sample is oversampled on disagreement, on purpose.** So the κ / precision /
   recall here characterize the **contested region** — the pairs where matchers differ
   — not the whole pair population. Because unanimous pairs (mostly easy "no") are
   under-represented, the *population* kappa would be **higher** than what this sample
   reports. This answers "when the matchers disagree, who's right?", not "what's the
   population agreement rate?", and shouldn't be quoted as the latter.

## Results (human-labeled, n=50)

Source: `eval_results/metric_validity_scores_20260801T211535Z.json`. The owner labeled
"supports" on **34 / 50** pairs (0.68 — on this disagreement-heavy sample). Ranked by
agreement with the human:

| matcher | Cohen's κ | κ 95% CI | precision | recall | TP / FP / FN / TN | Landis–Koch |
|---|---|---|---|---|---|---|
| **strict** (substring) | **0.211** | [−0.02, 0.44] | **1.00** | 0.29 | 10 / 0 / 24 / 16 | fair |
| overlap (fuzzy 0.5) | **0.184** | [−0.08, 0.45] | 0.78 | 0.53 | 18 / 5 / 16 / 11 | slight |
| semantic (cos ≥ 0.62) | **0.081** | [−0.23, 0.39] | 0.71 | 0.71 | 24 / 10 / 10 / 6 | slight |

**No matcher is demonstrably better than chance at n=50** (every κ CI crosses zero).
The point estimates rank strict ≥ overlap ≥ semantic, but within noise. What can be
said:

- **strict is a trustworthy floor** — precision **1.00** (every substring hit is real
  support, human-confirmed) but recall **0.29** (misses 71% of real supports).
- **overlap correlates only slightly** — precision 0.78 (~1 in 5 hits the human
  rejects) *and* recall 0.53 (misses ~half of real supports). Noisy in both directions;
  κ 0.18.
- **semantic matches the human's yes-*rate* but not the human's yes-*pairs*** —
  precision ≈ recall ≈ 0.71 looks balanced, but κ 0.08 (near chance): same marginal
  0.68, different specific pairs.

**The LLM proxy was a biased stand-in.** Human-vs-Haiku raw agreement 33/50 (0.66),
κ 0.37 ("fair"); the human labels "supports" 0.68 of the time vs Haiku's 0.38, and 16
of the 17 disagreements are human-yes / Haiku-no. Haiku systematically under-credited
support — which is what produced its flattering "overlap best, semantic over-counts"
ranking that the human does not reproduce. (The proxy also penalized the *embedding*-
based semantic matcher hardest — the opposite of a judge/embedding circularity.)

## What this implies about the committed recall numbers

See [`metric-validity-study.md`](metric-validity-study.md) §4 for the full, named list.
In short: the fuzzy matcher is **not** the validated choice it was reported to be
(κ 0.18, not 0.67); **recall@5 = 0.64 is a weak proxy for human support** and should be
read as a bracket (strict 0.093 floor → semantic 0.807 ceiling), not a calibrated
point; and the embedding-model win is **real but overstated** by the overlap column
(+0.127 overlap vs +0.053 semantic vs +0.000 strict), best sized by the chunk-invariant
accuracy gain (+0.08). The CI gate keeps the overlap matcher for regression detection
and continuity, but not on the "best-adjudicated" grounds previously claimed.

## Reproduce

```bash
python -m sec_rag.eval.label_sample     # (already run; sample committed, seed 13)

# Human path (authoritative):
python -m sec_rag.eval.label_cli        # a human labels the 50 pairs (resumable)
python -m sec_rag.eval.label_score      # -> eval_results/metric_validity_scores_*.json

# LLM-proxy path (superseded; kept for the contrast above):
python -m sec_rag.eval.label_auto       # Claude judge -> metric_validity_labels_llm.jsonl
python -m sec_rag.eval.label_score --labels eval_results/metric_validity_labels_llm.jsonl
```
