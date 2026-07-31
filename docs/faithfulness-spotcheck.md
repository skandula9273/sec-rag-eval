# Faithfulness-judge spot-check (v2)

Fulfils the design-doc commitment (L3 + the "LLM-judge bias" risk): **manually
adjudicate ~20 faithfulness judgments per eval run and report the agreement
rate**, to check whether the automated Haiku judge — which grades a Haiku-generated
answer, same model family on both sides — is trustworthy. This had never been
produced; the committed 0.929 mean stood unaudited.

## Method

- **Sample:** 20 questions, seed 13, the same `_select` the runner uses
  (`src/sec_rag/eval/faithfulness_spotcheck.py`). Reproducible.
- **Pipeline:** the live v2 config (`configs/v2.yaml`) — 3-large@1536, 1024-token
  chunks, dense top_k=5, temperature 0. Identical path to the committed eval.
- **Captured per question:** the generated answer, the exact retrieved sources,
  and the judge's `{n_claims, n_supported, score}`. Artifact:
  `eval_results/faithfulness_spotcheck_20260730T232959Z.json`.
- **Adjudicator:** an independent reviewer (Claude Opus — a *different* model from
  the Haiku judge, standing in for the human spot-check) read each (answer,
  sources, verdict) triple and re-decided whether each factual claim the answer
  makes is actually present in the retrieved sources. Every headline figure was
  verified by exact-string search against the **full** source chunks (not
  excerpts) — this caught two cases where a figure looked absent but was only past
  a display truncation, so no "error" was reported on a grounded answer.

**Disclosed limitations.** (1) The committed run stored only the aggregate mean,
not per-question detail, so this spot-check *regenerates* the answers rather than
replaying them; temperature 0 keeps it reproducible but a server-side model change
could shift wording. (2) The judge emits only counts, not which claims it docked,
so disagreements are diagnosed from the counts + the answer, not the judge's
reasoning. (3) The adjudicator is an LLM, not a human — a genuine third-party human
pass is still the gold standard; this narrows, not closes, the bias question.

## Result

- **Sample judge mean faithfulness: 0.90** (20 q) — consistent with the committed
  150-q mean of **0.929**.
- **Adjudicator–judge agreement: 19 / 20 (95%).** One clear judge error.
- **Direction of the one error is *harsh*, not lenient** — the judge *under*-scored
  a well-grounded answer. So the spot-check found **no evidence the headline 0.929
  is inflated by a lenient self-grader**; if anything the judge is marginally
  conservative. (Correcting the one error would *raise* the sample mean 0.90 → 0.95.)

### The one disagreement — record [10], `financebench_id_05915` (CVS fixed-asset turnover)

Judge scored **0/1 = 0.0**. The answer correctly refuses to compute the ratio
(PP&E isn't in the retrieved chunks) but states three figures along the way —
FY2018 revenue **$194,579M**, total assets **$196,456M** (2018) and **$95,131M**
(2017) — all three verified present in source [1] (the CVS five-year summary). By
the judge's own rule ("ignore refusals; score the factual claims"), the supported
claims should score ~1.0. The judge both under-counted the claims and marked the
one it kept as unsupported. **A false negative — the judge is too harsh here.**

### Two soft notes (verdict defensible, score debatable) — these roughly cancel

- **[2] American Water Works working capital (0.75):** the answer's "current
  assets Dec-31-2022 = $1,554M [5]" is a *misattributed* number — $1,554M is the
  **2021** current-assets figure (source [4]), and 2022 current assets aren't in
  any retrieved source; only current liabilities $2,811M is correctly grounded.
  The judge docked 1 of 4, which is directionally right but arguably **too lenient**
  for a fabricated-year key input.
- **[17] Coca-Cola ROA (0.75)** and **[9] Best Buy (0.75):** all *input* figures
  are grounded; the judge docked the *derived* value (the computed ROA / the
  mislabeled "Q2" tag). Defensible, mildly **harsh**.

### What the spot-check does NOT tell you (the important caveat)

Faithfulness measures **grounding in the retrieved sources, not correctness.**
Several answers are faithful yet wrong against the gold answer because retrieval
surfaced the wrong evidence:

- **[2]** grounded arithmetic on a wrong-year input → WC −$1,257M vs gold −$1,561M.
- **[9]** grounds a full-year store count and mislabels it "Q2"; never answers the
  actual YoY question (gold 982→969).
- **[0]** faithfully reports the filing's stated effective rate (−0.6% / 14.7%),
  which differs from gold's computed −14.76% convention.

So **0.929 faithful and recall@5 0.64 are consistent**: the system is well-grounded
in whatever it retrieves, while still retrieving the wrong evidence a third of the
time. A high faithfulness number is *not* an accuracy number and should never be
read as one.

## Per-question adjudication

| # | id | type | judge | grounded? | verdict |
|---|----|------|-------|-----------|---------|
| 0 | 00585 | novel | 2/2=1.0 | ✓ rates in src[2] | agree |
| 1 | 04209 | metrics | 1/1=1.0 | ✓ 59,268 | agree |
| 2 | 00070 | domain | 3/4=0.75 | partial (CA misattributed) | agree (judge lenient) |
| 3 | 10285 | metrics | 2/2=1.0 | ✓ 12,645/31,213/18,568 | agree |
| 4 | 00757 | novel | 3/3=1.0 | ✓ verbatim quote | agree |
| 5 | 03838 | metrics | 5/5=1.0 | ✓ 9,497,578/4,713,500 | agree |
| 6 | 01198 | domain | 13/13=1.0 | ✓ 23,601/16,434 + drivers | agree |
| 7 | 06655 | metrics | 0/0=1.0 | refusal, no claims | agree |
| 8 | 00746 | domain | 0/0=1.0 | refusal (wrong-co sources) | agree |
| 9 | 00460 | novel | 3/4=0.75 | 925 grounded, mislabeled Q2 | agree (mild harsh) |
| 10 | 05915 | metrics | **0/1=0.0** | ✓ 194,579/196,456/95,131 | **DISAGREE (too harsh → should be ~1.0)** |
| 11 | 01858 | novel | 5/5=1.0 | ✓ dividend figures | agree |
| 12 | 04700 | metrics | 3/3=1.0 | ✓ 85,320/52,540 (past truncation) | agree |
| 13 | 00995 | domain | 28/28=1.0 | ✓ all products (EPYC…Versal) | agree (not rubber-stamped) |
| 14 | 01226 | domain | 11/11=1.0 | ✓ 20.8/19.1/2.4 + drivers | agree |
| 15 | 10130 | metrics | 3/4=0.75 | ✓ AP/inventory inputs; docked derived DPO | agree |
| 16 | 03882 | metrics | 3/3=1.0 | ✓ 1,615.9 | agree |
| 17 | 03473 | metrics | 3/4=0.75 | ✓ inputs; docked derived ROA | agree (mild harsh) |
| 18 | 03531 | metrics | 1/1=1.0 | ✓ 16,525 | agree |
| 19 | 00552 | domain | 0/0=1.0 | refusal | agree |

## Recommendations (fall out of this exercise)

1. **Make the judge emit per-claim verdicts** (claim text + supported bool), not
   just counts. The [10] error was invisible in the aggregate and only findable by
   re-reading the sources; per-claim output makes spot-checks auditable and cheap.
2. **Persist per-question detail in the main eval run** (answer + source ids +
   judge verdict), so a spot-check can *replay* rather than *regenerate*.
3. **Report faithfulness next to a correctness metric, never alone** — the
   faithful-but-wrong cases ([0]/[2]/[9]) show the number is a grounding signal,
   not an accuracy claim, and is easy to misread as the latter.
4. A **true third-party human** pass on the same 20 remains the gold standard; this
   LLM-adjudicated pass narrows the bias question but doesn't close it.
