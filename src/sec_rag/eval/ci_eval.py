"""CI smoke eval — a per-PR retrieval-only gate on recall@5.

Runs the committed 30-question subset (seed 13, 10/category) through the production
retrieval config and scores recall@5 under the S4 human-validated matcher (overlap,
Cohen's kappa 0.67 vs labels — docs/metric-validity.md). FAILS the PR check if
recall@5 falls more than ``max_drop`` below the committed ``baseline_recall@5``. All
gate parameters live in configs/ci_eval.yaml, not here and not in the workflow YAML.

Retrieval-only on purpose: no generation, so no Anthropic call and near-zero cost
(~30 small OpenAI query embeddings + DB reads). Operational safety:
  * skips gracefully (exit 0, clear message) when OPENAI_API_KEY / DATABASE_URL are
    absent — so fork PRs and Dependabot don't fail confusingly;
  * a ``max_questions`` cost ceiling caps how much CI ever evaluates;
  * writes a PR-comment table + a GITHUB_OUTPUT ``status`` so the workflow can post
    the delta and fail the check independently.

  python -m sec_rag.eval.ci_eval                 # run the gate
  python -m sec_rag.eval.ci_eval --update-baseline  # recompute + rewrite the baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from sec_rag.config import Secrets, load_config
from sec_rag.eval.metrics import evidence_match_rank, hit_rate_at_k, load_matchers

_MARKER = "<!-- ci-eval-smoke -->"  # lets the workflow update one sticky comment


def _gh_output(**kv) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            for k, v in kv.items():
                f.write(f"{k}={v}\n")


def _load_subset(path: str, cap: int) -> list[dict]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "_meta" not in r]
    if len(rows) > cap:  # cost ceiling: never evaluate more than max_questions
        print(f"note: subset has {len(rows)} > max_questions {cap}; truncating to {cap}.")
        rows = rows[:cap]
    return rows


def _measure(cfg: dict) -> tuple[float, float, int]:
    """(recall@5, recall@10, n) on the subset under the configured matcher."""
    from sec_rag.pipeline import QueryEngine

    subset = _load_subset(cfg["subset_file"], cfg["max_questions"])
    matcher = load_matchers("configs/matchers.yaml", Secrets())[cfg["matcher"]]
    engine = QueryEngine(load_config(cfg["retrieval_config"]))
    ranks = []
    try:
        for q in subset:
            chunks, _ = engine.retrieve(q["question"], top_k=cfg["top_k"])
            ranks.append(evidence_match_rank([c.content for c in chunks],
                                             q["evidence_texts"], matcher=matcher))
    finally:
        engine.close()
    return hit_rate_at_k(ranks, 5), hit_rate_at_k(ranks, 10), len(subset)


def _comment(cfg: dict, cur5: float, cur10: float, n: int, passed: bool) -> str:
    base = cfg["baseline_recall@5"]
    delta = cur5 - base
    verdict = "✅ pass" if passed else "❌ FAIL"
    return (
        f"{_MARKER}\n### Eval-as-CI smoke gate — {verdict}\n\n"
        f"Retrieval-only on the committed **{n}-question** subset (`{cfg['subset_file']}`), "
        f"scored under the S4 human-validated **{cfg['matcher']}** matcher.\n\n"
        f"| metric | baseline | this PR | Δ | gate |\n|---|---|---|---|---|\n"
        f"| recall@5 | {base:.4f} | **{cur5:.4f}** | {delta:+.4f} | "
        f"fail if Δ < −{cfg['max_drop']:.2f} |\n"
        f"| recall@10 | — | {cur10:.4f} | — | (informational) |\n\n"
        f"_Smoke test only — 30 q, not the n=150 benchmark; each question is ±0.033 recall@5. "
        f"It catches a retrieval regression, not small quality shifts or answer accuracy._"
    )


def _write_comment(text: str) -> None:
    Path("ci_eval_comment.md").write_text(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="CI smoke eval gate")
    ap.add_argument("--config", default="configs/ci_eval.yaml")
    ap.add_argument("--update-baseline", action="store_true",
                    help="recompute recall@5 on the subset and rewrite it into the config")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    # Graceful skip: no keys (fork PR / Dependabot) -> don't fail confusingly.
    secrets = Secrets()
    missing = [k for k in ("openai_api_key", "database_url") if not getattr(secrets, k)]
    if missing:
        msg = (f"eval-ci: SKIPPED — missing {', '.join(missing).upper()} "
               "(expected on fork PRs; the gate needs repo secrets). Not a failure.")
        print(msg)
        _write_comment(f"{_MARKER}\n### Eval-as-CI smoke gate — ⏭️ skipped\n\n{msg}")
        _gh_output(status="skip")
        sys.exit(0)

    cur5, cur10, n = _measure(cfg)

    if args.update_baseline:
        raw = Path(args.config).read_text()
        raw = raw.replace(f"baseline_recall@5: {cfg['baseline_recall@5']}",
                          f"baseline_recall@5: {round(cur5, 4)}")
        Path(args.config).write_text(raw)
        print(f"Updated baseline_recall@5 -> {cur5:.4f} (n={n}) in {args.config}")
        sys.exit(0)

    passed = cur5 >= cfg["baseline_recall@5"] - cfg["max_drop"]
    _write_comment(_comment(cfg, cur5, cur10, n, passed))
    _gh_output(status="pass" if passed else "fail",
               recall5=f"{cur5:.4f}", delta=f"{cur5 - cfg['baseline_recall@5']:+.4f}")
    print(f"recall@5 = {cur5:.4f}  baseline = {cfg['baseline_recall@5']}  "
          f"delta = {cur5 - cfg['baseline_recall@5']:+.4f}  "
          f"gate(min {cfg['baseline_recall@5'] - cfg['max_drop']:.4f}): "
          f"{'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
