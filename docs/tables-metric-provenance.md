# `tables@5` provenance — investigation (no fix applied)

Investigation of where the README results-table numbers `tables@5 (fuzzy)` = **0.32
(V0)** and **0.70 (V2)** come from. No README text was changed in this pass.

## The blunt answer, up front

- **There is no table classifier anywhere in the code.** `tables@5` is a **display
  alias** for the recall@5 of FinanceBench's `metrics-generated` question category.
- **Both 0.32 and 0.70 are that same category's recall@5**, under the same fuzzy(0.5)
  matcher, on the same 150-question set, dense retrieval. The only things that differ
  between them are the intended A/B variables (embedding model + chunk size). **So the
  two numbers ARE the same measurement, and the 0.32 → 0.70 delta is a valid
  apples-to-apples comparison** — corroborated independently by the confound study.
- **What is wrong is the label, not the comparison.** "tables" is an undisclosed
  nickname for `metrics-generated`. Bluntly: **0.70 is the `metrics-generated` category
  under a different name — and so is 0.32.** Both columns are the same relabel, which is
  exactly why they stay comparable.

## 1. Is there a scorer that classifies questions/spans as tabular?

**No.** Categories are grouped purely by FinanceBench's own `question_type` field:

- `src/sec_rag/eval/run_financebench.py:190` — `cat = q.question_type or "uncategorized"`
  → `by_cat[cat].append(rank)`; `per_category_recall` (line 240) is emitted per that key.
- **No content-based tabular test** exists in `metrics.py` or `run_financebench.py`
  (grep for `isdigit` / table structure / column markers on the gold span → nothing).

The word "tables" enters only as a **label/nickname on the `metrics-generated` category**:

- `src/sec_rag/eval/ablation_fusion.py:103` prints a column header `tables@5`, and
  lines `78–79` / `109` feed that column `hit_rate_at_k(by_cat.get("metrics-generated"), 5)`.
- `src/sec_rag/eval/ablation_chunksize_local.py:110` and
  `src/sec_rag/eval/ablation_embedding_local.py:79` hardcode `"tables@5": 0.32` inside a
  `committed_baseline_512_neon` reference dict.
- `src/sec_rag/eval/diag_table_parse.py:9,60` and `diag_retrieval_depth.py:42,59` select
  `question_type == "metrics-generated"` and call them "table" questions in prose.

**Basis of classification:** FinanceBench's `question_type == "metrics-generated"`.
Nothing about whether the gold evidence is actually a table. (`metrics-generated`
questions ask for a specific financial figure, whose evidence is *usually* a
statement line-item — so "tables" is a defensible nickname, but a nickname, not a
measured property.)

## 2. Every eval JSON with a `tables` key

Two files, both containing the **hardcoded V0 reference** (not a fresh measurement):

| File | Key | Value | What it is |
|---|---|---|---|
| `eval_results/ablation_chunksize_local_20260627T181333Z.json` | `committed_baseline_512_neon.tables@5` | **0.32** | hardcoded pointer to the V0 number (3-small/512 arm) |
| `eval_results/ablation_embedding_local_20260627T185100Z.json` | `committed_baseline_512_neon.tables@5` | **0.32** | same hardcoded pointer (3-small vs 3-large ablation) |

**No other JSON has a `tables` key.** The main runner never writes one — it writes
`per_category_recall.{domain-relevant, metrics-generated, novel-generated}`. Every
"tables" number in the docs is a hand-copy of a `metrics-generated` value.

## 3. The specific runs behind 0.32 and 0.70

Both are `per_category_recall["metrics-generated"]["recall@5"]`, `match_mode: fuzzy`
(threshold 0.5), n=150, dense.

**V0 = 0.32** — `eval_results/financebench_20260605T020304Z.json` (committed full V0
baseline)
- config: dense · `text-embedding-3-small` · 512-token chunks · fuzzy(0.5) · mode=full
- `per_category_recall["metrics-generated"] = {recall@5: 0.32, recall@10: 0.44}`
- (identical 0.32 in `financebench_20260604T011712Z.json` and the retrieval-only
  `financebench_20260615T210625Z.json`.)

**V2 = 0.70** — `eval_results/financebench_20260629T160938Z.json` (v2 retrieval-only
baseline)
- config: dense · `text-embedding-3-large@1536` · 1024-token chunks · fuzzy(0.5) ·
  mode=retrieval_only
- `per_category_recall["metrics-generated"] = {recall@5: 0.70, recall@10: 0.78}`
- (identical 0.70 in the full run `financebench_20260629T193049Z.json`, the concise-prompt
  run `financebench_20260731T025337Z.json`, and the accuracy run
  `financebench_20260731T115740Z.json` you cited.)

## 4. Direct answer to the question asked

> Are 0.32 and 0.70 the same measurement on the same definition of "table question"?
> Or is 0.70 the `metrics-generated` category relabeled?

**Both.** They are the *same measurement* (metrics-generated recall@5, fuzzy(0.5), same
150-q set) — **and** that measurement is `metrics-generated` relabeled "tables." 0.70 is
the metrics-generated category under a different name; so is 0.32. Because it is the
*same* relabel in both columns, the numbers stay comparable to each other, and the
V0→V2 improvement on that category is real (not a metric artifact — accuracy on the
category rises too, per the confound study).

The defect is **not** comparability. It is that the README presents this as `tables@5`,
which implies a classifier of tabular *evidence* that does not exist. The honest answer
to an interviewer's "how do you decide a question is a table question?" is: *"I don't —
I use FinanceBench's `metrics-generated` question type as a proxy and nickname it
tables."* That is fine **if disclosed**; the README does not disclose it.

## Honesty verdict + the README edit

Per your conditional — *"only if the two are NOT comparable, give the exact README
edit"* — **the two ARE comparable, so no comparability edit is required.** I will not
manufacture one.

There is, separately, a real but lesser **labeling** defect (undisclosed nickname). It
is your call whether to fix it; the honest options, for completeness, are:

- **Rename the row** so it names what is measured:
  `| tables@5 (fuzzy) | 0.32 | 0.70 | — |` → `| metrics-generated@5 (fuzzy) | 0.32 | 0.70 | — |`
- **Or keep "tables@5" and footnote the mapping:** *"tables = FinanceBench's
  `metrics-generated` question category (numeric/statement questions), used as a proxy —
  not a classifier of table structure."*

Either makes the label honest without touching the numbers, which stand.
