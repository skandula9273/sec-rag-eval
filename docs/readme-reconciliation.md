# README ↔ committed-artifact reconciliation

Every quantitative claim in `README.md` traced to the `eval_results/*.json` that
produced it, and verified. The committed JSON is the truth; the prose is a claim about
it. Where they disagreed, the **prose was corrected to match the artifact** — never the
reverse. Run: 2026-08-02.

Verdict legend: **MATCH** (value agrees, rounding allowed) · **MISMATCH** (fixed) ·
**PROVENANCE** (value right, source unclear/wrong-run — fixed by citing) · **NO
ARTIFACT** (no committed JSON behind it — flagged, not deleted).

## Results table + headline

| Claim | README | Artifact | Artifact value | Verdict |
|---|---|---|---|---|
| recall@5 V0 (fuzzy) | 0.44 | `financebench_20260605T020304Z.json` | 0.44 | MATCH |
| recall@5 V2 (fuzzy) | 0.64 | `financebench_20260629T160938Z.json` | 0.64 | MATCH |
| recall@10 V0 | 0.54 | `…20260605T020304Z` | 0.54 | MATCH |
| recall@10 V2 | 0.74 | `…20260629T193049Z` (full) | 0.74 | MATCH (retrieval-only run reads 0.7467) |
| tables@5 V0 | 0.32 | `…20260605T020304Z` `metrics-generated` | 0.32 | MATCH¹ |
| tables@5 V2 | 0.70 | `…20260629T160938Z` `metrics-generated` | 0.70 | MATCH¹ |
| faithfulness V0 | 0.94 | `…20260605T020304Z` | 0.941 | MATCH |
| faithfulness V2 | 0.93 | `…20260629T193049Z` / `…20260731T025337Z` | 0.9293 / 0.9338 | **PROVENANCE → fixed** (issue 2) |
| cost/query V0 | $0.0063 | `…20260605T020304Z` | 0.006265 | MATCH |
| cost/query V2 | ~$0.010–0.017 (top_k=20) | `…20260731T115740Z` | 0.016565 (top_k=20) | **MISMATCH → fixed** (0.010 unsourced) |
| correctness (over 150) | 0.50 | `…20260731T115740Z` | 0.50 | MATCH |

¹ `tables@5` is a display alias for FinanceBench's `metrics-generated` category, not a
tabular classifier — see `docs/tables-metric-provenance.md`. Values agree; label is a
separately-documented issue.

## Answer-accuracy table (all from `financebench_20260731T115740Z.json`, top_k=20)

| Claim | README | Artifact value | Verdict |
|---|---|---|---|
| LLM accuracy (of attempted) | 0.74 (75/102) | 0.7353, n_correct 75 / n_answered 102 | MATCH |
| LLM accuracy (over 150) | 0.50 | 0.50 | MATCH |
| numeric-exact | 0.85 (45/53) | 0.8491, 45/53 | MATCH |
| refusal rate | 0.32 (48/150) | 0.32, 48/150 | MATCH |

## Recall bracket (issue 3)

| Claim | README | Artifact | Artifact value | Verdict |
|---|---|---|---|---|
| bracket strict | 0.093 | `matcher_study_baseline_20260731T123250Z.json` | 0.0933 | MATCH |
| bracket fuzzy | 0.64 | `matcher_study_baseline_…` | 0.64 | MATCH |
| bracket semantic | 0.807 | `matcher_study_baseline_…` | 0.8067 | MATCH |

The bracket is the **live v2 config through Neon** (`matcher_study_baseline`), **not**
the confound grid. The confound grid's 3-large/1024 cell
(`confound_grid_20260731T162218Z.json`) is a *different* run (local exact-cosine index)
and reads **0.0933 / 0.6467 / 0.8133**. Values were correct but uncited → **fixed by
citing the source** so the two runs aren't conflated.

## Confound decomposition (all from `confound_study_20260731T170928Z.json`)

| Claim | README | Artifact value | Verdict |
|---|---|---|---|
| V0→V2 overlap gain | +0.207 | 0.2067 | MATCH |
| embedding component (61%) | +0.127 | 0.1267 (61.3%) | MATCH |
| chunk component (39%) | +0.080 | 0.08 (38.7%) | MATCH |
| chunk real | ~+0.037 | 0.0366 | MATCH |
| chunk inflation | ~+0.043 | 0.0434 | MATCH |
| ~80% real / ~20% inflation | 80/20 | 0.1633/0.2067 = 79% / 21% | MATCH |
| accuracy embedding effect | +0.08 | 0.08 | MATCH |
| refusals (embedding) | 0.51→0.39 | 0.5133→0.3933 | MATCH |
| accuracy chunk effect | +0.033 | 0.0333 | MATCH |
| accuracy V0→V2 | +0.11 (0.36→0.47) | 0.1133 (0.36→0.4733) | MATCH |
| interaction ≈ 0 | ≈0 | 0.0 | MATCH |

(V2 accuracy 0.4733 is the top_k=10 run `financebench_20260731T111452Z.json`, held at
top_k=10 for the apples-to-apples decomposition; the headline 0.50 is the deployed
top_k=20. Both correct, different depths.)

## Substring floor, metric validity, judge audit, latency, CI, corpus

| Claim | README | Artifact | Artifact value | Verdict |
|---|---|---|---|---|
| V0 substring recall@5 | 0.0667 | `financebench_20260604T022143Z.json` | 0.0667 | MATCH |
| V0 substring tables@5 | 0.00 | `…20260604T022143Z` `metrics-generated` | 0.0 | MATCH |
| κ 0.67 (LLM proxy, best) | 0.67, best of 3 | `metric_validity_scores_20260731T140147Z.json` | overlap 0.6737, rank 1 | MATCH |
| κ 0.18 (hand-labeled) | 0.18 | `metric_validity_scores_20260801T211535Z.json` **(UNTRACKED)** | overlap 0.1835 | MATCH-value, **artifact uncommitted (flag)** |
| overlap rank (hand) | "third of three" | `…20260801T211535Z` | **rank 2** (strict 0.2105 > overlap 0.1835 > semantic 0.0809) | **MISMATCH → fixed** (issue 1) |
| "no matcher beats chance" | — | `…20260801T211535Z` 95% CIs | all three CIs include 0 | MATCH |
| oversampling caveat | (absent) | both scores files `sample_note` | "disagreement-oversampled … contested pairs, not a population rate" | **MISSING → added** (issue 1) |
| judge audit agreement | 19/20 | `docs/faithfulness-spotcheck.md` + `faithfulness_spotcheck_20260730T232959Z.json` (20 records) | 19/20 (adjudication in the .md; 20 records committed) | MATCH |
| committed faithfulness | 0.929 | `financebench_20260629T193049Z.json` | 0.9293 | MATCH |
| warm server p50 / p95 | 2.1 s / 6.2 s | `api_latency_20260731T004411Z.json` `warm_server_latency_ms` | 2058 / 6190 ms | MATCH (labeled "warm") |
| cost/query (live API) | $0.0045 | `api_latency_…` `cost_per_query_usd.mean` | 0.004461 | MATCH |
| CI baseline / max_drop | 0.6333 / 0.10 | `configs/ci_eval.yaml` | 0.6333 / 0.10 | MATCH |
| tests | 118 (104 fns) | `pytest` (self-run, no secrets) | 118 passed / 104 defs | MATCH |
| corpus chunks | 15,192 | `confound_grid_20260731T162218Z.json` `n_chunks_per_cell` | 15192 | MATCH |

## NO ARTIFACT — flagged, not deleted, not back-filled

Per the rule "if a claim has no artifact, flag it; don't delete it silently and don't go
find a number that fits":

- **~13 s cold start** (demo note): no committed JSON. It is the Cloud Run
  **scale-to-zero** wake of the *web* host — an observation. The `api_latency` JSON's
  `cold_start` (1863 ms server / 2077 ms client) is a **different thing**: the first
  request in a batch to an *already-warm* API container, not a scale-to-zero boot. Fixed
  by **labeling** the ~13 s as scale-to-zero (observed) and the p50/p95 as *warm*, so the
  two can't be read as contradicting each other (issue 4). Value itself still unbacked.
- **~6 s (a new EDGAR filing)** (demo note): observational; no committed timing artifact
  for the live fetch-and-index path. Labeled "observed".
- **84 filings · 274 MB · ~10,400 companies**: true ingest/DB/EDGAR facts, but **not in
  any committed eval JSON** (only `CLAUDE.md` asserts them). Flagged; left as-is.
- **~$0.001 / CI run**: an estimate (~30 query embeddings), labeled "~"; no artifact.

## Related findings outside README

- ✅ **RESOLVED — `configs/ci_eval.yaml` comment.** It called overlap "the S4
  human-validated matcher (Cohen's kappa 0.67 vs labels)". That κ 0.67 is the
  **LLM-proxy** number, since disproven (hand-labeled κ 0.18, 2nd of 3). The comment now
  states overlap is kept for continuity/regression detection, not because it's the
  validated best.
- ✅ **RESOLVED — the κ 0.18 truth-file is now committed.**
  `metric_validity_scores_20260801T211535Z.json` (overlap κ 0.1835) and
  `metric_validity_labels.jsonl` (50 hand labels, annotator "human (owner)", 0.68
  yes-rate) are committed, so the corrected claim rests on committed data.
- ✅ **RESOLVED — `docs/metric-validity.md` and `docs/metric-validity-study.md`
  reconciled.** Every number in both was verified against the artifacts (κ table,
  human-vs-Haiku κ 0.374 / 33-50 agreement / 16-of-17 direction, 8-of-10 unanimous-no,
  matcher recall table incl. semantic recall@10 0.873, the confound figures — all
  MATCH). The one defect — overlap ranked **"third"** in 3 places — was corrected to
  **"second"** (strict 0.211 > overlap 0.184 > semantic 0.081), matching each doc's own
  κ table.

## Still unbacked (accepted as-is, flagged)

The NO-ARTIFACT items above remain claims without a committed eval JSON: `~13 s` /
`~6 s` latency observations (now labeled "observed"), and `84 filings / 274 MB /
~10,400 companies` (ingest/EDGAR facts asserted only in `CLAUDE.md`). Left in place per
the "flag, don't back-fill" rule.
