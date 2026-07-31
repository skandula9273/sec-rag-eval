"""How much does the headline recall depend on the evidence matcher?

Scores retrieval under ALL matchers in configs/matchers.yaml (strict / overlap /
semantic) so the reader can see how much the number is a property of the retriever
vs a property of the metric. Retrieval is matcher-independent, so each arm is
retrieved ONCE and every matcher is applied to the same retrieved contents.

  --mode baseline  : the live V2 config (Neon), 150 q, recall@5/@10 + MRR per matcher.
  --mode ablations : re-runs the two headline ablations (embedding model, chunk size)
                     under all three matchers and reports which conclusions survive
                     matcher choice. Heavy (re-embeds the 512-token corpus locally).

No matcher is declared "best" here — that needs human-agreement data (a later step).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from sec_rag.config import Secrets, load_config
from sec_rag.eval.metrics import (
    evidence_match_rank,
    hit_rate_at_k,
    load_matchers,
    mean_reciprocal_rank,
)
from sec_rag.ingest.financebench import load_questions
from sec_rag.pipeline import QueryEngine

KS = [5, 10]


def score_arm(retrieved: list[list[str]], questions, matchers: dict, ks=KS) -> dict:
    """recall@k + MRR (overall + per category) for each matcher over the SAME
    retrieved contents. Semantic matchers are warmed (batch-embedded) first."""
    corpus = [c for rc in retrieved for c in rc]
    evidence = [e for q in questions for e in (q.evidence_texts or [])]
    for m in matchers.values():
        m.warm(corpus + evidence)

    out = {}
    for name, m in matchers.items():
        ranks: list[int | None] = []
        by_cat: dict[str, list[int | None]] = defaultdict(list)
        for rc, q in zip(retrieved, questions, strict=True):
            r = evidence_match_rank(rc, q.evidence_texts, matcher=m)
            ranks.append(r)
            by_cat[q.question_type or "uncategorized"].append(r)
        out[name] = {
            "recall_at_k": {f"recall@{k}": round(hit_rate_at_k(ranks, k), 4) for k in ks},
            "mrr": round(mean_reciprocal_rank(ranks), 4),
            "per_category_recall": {
                c: {f"recall@{k}": round(hit_rate_at_k(rs, k), 4) for k in ks}
                for c, rs in sorted(by_cat.items())
            },
        }
    return out


def _retrieve_neon(cfg, top_k: int) -> tuple[list[list[str]], list]:
    questions = load_questions(cfg.eval.dataset)
    engine = QueryEngine(cfg)
    retrieved = []
    try:
        for q in questions:
            chunks, _ = engine.retrieve(q.question, top_k=top_k)
            retrieved.append([c.content for c in chunks])
    finally:
        engine.close()
    return retrieved, questions


def _matcher_specs(path: str) -> dict:
    return (yaml.safe_load(Path(path).read_text()) or {}).get("matchers", {})


def run_baseline(config: str, matchers_path: str) -> dict:
    cfg = load_config(config)
    top_k = max(KS)
    matchers = load_matchers(matchers_path, Secrets())
    retrieved, questions = _retrieve_neon(cfg, top_k)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "matcher_study_baseline",
        "config_file": config,
        "embedding_model": cfg.embedding.model,
        "chunk_tokens": cfg.chunking.max_tokens,
        "n_questions": len(questions),
        "top_k_retrieved": top_k,
        "matcher_specs": _matcher_specs(matchers_path),
        "note": "recall under each matcher over the SAME retrieval. strict = exact "
        "substring; overlap = fuzzy(0.5); semantic = cosine>=threshold over spans.",
        "by_matcher": score_arm(retrieved, questions, matchers),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Matcher-dependence study -> JSON")
    ap.add_argument("--mode", choices=["baseline", "ablations"], default="baseline")
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--matchers", default="configs/matchers.yaml")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    if args.mode == "baseline":
        report = run_baseline(args.config, args.matchers)
        prefix = "matcher_study_baseline"
    else:
        from sec_rag.eval.matcher_ablations import run_ablations

        report = run_ablations(args.matchers)
        prefix = "matcher_study_ablations"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = out_dir / f"{prefix}_{stamp}.json"
    p.write_text(json.dumps(report, indent=2))
    print(f"Wrote {p}\n")

    if args.mode == "baseline":
        print(f"{'matcher':12}{'recall@5':>10}{'recall@10':>11}{'MRR':>8}")
        for name, r in report["by_matcher"].items():
            print(f"{name:12}{r['recall_at_k']['recall@5']:>10}"
                  f"{r['recall_at_k']['recall@10']:>11}{r['mrr']:>8}")
    else:
        print(report.get("summary", ""))


if __name__ == "__main__":
    main()
