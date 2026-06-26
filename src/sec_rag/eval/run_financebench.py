"""FinanceBench eval runner.

Runs each FinanceBench question through the same QueryEngine the API uses,
scores retrieval against the gold evidence spans, and writes a timestamped JSON
to eval_results/. That JSON is committed per run, so any number in the writeup
traces to one file with the config that produced it.

Usage:
  python -m sec_rag.eval.run_financebench --config configs/v0.yaml
  python -m sec_rag.eval.run_financebench --config configs/v0.yaml --limit 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sec_rag.config import Config, load_config
from sec_rag.eval.metrics import evidence_match_rank, hit_rate_at_k, mean_reciprocal_rank
from sec_rag.generate.answer import PRICING
from sec_rag.ingest.financebench import Question, load_questions
from sec_rag.pipeline import QueryEngine


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. p in [0, 100]. Empty -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return float(s[idx])


def _select(questions: list[Question], limit: int | None, seed: int) -> list[Question]:
    if limit is None or limit >= len(questions):
        return questions
    rng = random.Random(seed)
    return rng.sample(questions, limit)


def run(
    cfg: Config,
    limit: int | None = None,
    match_mode: str = "substring",
    sleep_s: float = 0.0,
    retrieval_only: bool = False,
) -> dict:
    ks = sorted(cfg.eval.recall_ks)
    top_k = max(ks)
    questions = _select(load_questions(cfg.eval.dataset), limit, cfg.eval.seed)

    ranks: list[int | None] = []
    by_cat: dict[str, list[int | None]] = defaultdict(list)
    latencies: list[float] = []
    costs: list[float] = []
    faithfulness_scores: list[float] = []  # populated only when eval.faithfulness on
    misses_no_evidence = 0
    errors: list[dict] = []  # questions that failed even after a retry

    # One failing question (a transient Neon drop, an Anthropic timeout, a rate
    # limit) must not throw away a 150-question run that is otherwise complete.
    # Each question gets a bounded retry on a FRESH engine — a dead pooled
    # connection is the failure we already hit during ingest, and a new engine
    # reconnects — and anything still failing is recorded and skipped, not fatal.
    # Failures are counted and disclosed in the report (rule 2: honest numbers).
    # Normalise one question to (contents, latency_ms, cost, faithfulness).
    # retrieval_only stops after retrieval (no Anthropic call): recall@k / MRR are
    # pure retrieval metrics, so they need only the query embedding + the DB. cost
    # and faithfulness are then None — there was no generation to price or judge.
    # The retrieval path is identical to full mode, so the recall is the same.
    def _evaluate(engine: QueryEngine, question: str):
        if retrieval_only:
            chunks, retr_ms = engine.retrieve(question, top_k=top_k)
            return [c.content for c in chunks], retr_ms, None, None
        res = engine.run(question, top_k=top_k)
        m = res.response.metrics
        return [c.excerpt for c in res.response.citations], m.latency_ms, m.cost_usd, m.faithfulness

    # With faithfulness on, each full-mode question makes two Anthropic calls
    # (answer + judge); 150 back-to-back can saturate a low account rate limit.
    # sleep_s spaces them out — set it (e.g. --sleep 1.0) when a full run trips
    # the per-minute limit. 0.0 = no pause (the default; fine for small runs).
    engine = QueryEngine(cfg)
    try:
        for i, q in enumerate(questions):
            if sleep_s and i > 0:
                time.sleep(sleep_s)
            try:
                contents, latency, cost, faith = _evaluate(engine, q.question)
            except Exception:
                # Rebuild the engine (new DB connection) and try once more.
                try:
                    engine.close()
                except Exception:
                    pass
                try:
                    engine = QueryEngine(cfg)
                    contents, latency, cost, faith = _evaluate(engine, q.question)
                except Exception as second_exc:
                    errors.append({"id": q.id, "error": f"{type(second_exc).__name__}: {second_exc}"})
                    continue
            if not q.evidence_texts:
                misses_no_evidence += 1
            rank = evidence_match_rank(contents, q.evidence_texts, mode=match_mode)
            ranks.append(rank)
            by_cat[q.question_type or "uncategorized"].append(rank)
            latencies.append(latency)
            if cost is not None:
                costs.append(cost)
            if faith is not None:
                faithfulness_scores.append(faith)
    finally:
        engine.close()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(questions),       # questions attempted
        "n_scored": len(ranks),              # questions that produced a result
        "match_mode": match_mode,
        "mode": "retrieval_only" if retrieval_only else "full",
        "config": {
            "chunking": cfg.chunking.model_dump(),
            "embedding_model": cfg.embedding.model,
            "retrieval": cfg.retrieval.model_dump(),
            "generation_model": cfg.generation.model,
            "seed": cfg.eval.seed,
        },
        "recall_at_k": {f"recall@{k}": round(hit_rate_at_k(ranks, k), 4) for k in ks},
        "mrr": round(mean_reciprocal_rank(ranks), 4),
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        },
        # No generation in retrieval-only mode -> nothing to price. None (not 0.0)
        # so a free retrieval run can't be misread as a $0 full run.
        "cost_usd": None if retrieval_only else {
            "mean_per_query": round(sum(costs) / len(costs), 6) if costs else 0.0,
            "total": round(sum(costs), 6),
            # Estimate iff the generation model has no confirmed rate in PRICING.
            "is_estimate": cfg.generation.model not in PRICING,
        },
        "per_category_recall": {
            cat: {f"recall@{k}": round(hit_rate_at_k(rs, k), 4) for k in ks}
            for cat, rs in sorted(by_cat.items())
        },
        # Mean faithfulness across scored questions (null if eval.faithfulness off
        # or retrieval-only — no answer was generated to judge).
        "faithfulness": {
            "enabled": False if retrieval_only else cfg.eval.faithfulness,
            "mean": round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
            if faithfulness_scores else None,
            "n_scored": len(faithfulness_scores),
        },
        "questions_without_evidence": misses_no_evidence,
        "n_errors": len(errors),
        "errors": errors,  # ids + messages for any question that failed twice
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run FinanceBench eval -> JSON")
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--limit", type=int, default=None, help="sample N questions (seeded)")
    ap.add_argument("--match-mode", choices=["substring", "fuzzy"], default="substring")
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="seconds to pause between questions (throttle to stay under API rate limits)",
    )
    ap.add_argument(
        "--no-generate",
        action="store_true",
        help="retrieval-only: score recall@k/MRR without generation or faithfulness "
        "(no Anthropic calls; only OpenAI query embeddings + the DB)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = run(
        cfg,
        limit=args.limit,
        match_mode=args.match_mode,
        sleep_s=args.sleep,
        retrieval_only=args.no_generate,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"financebench_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote {out_path}")
    print(
        f"  mode = {report['mode']}  n = {report['n_questions']}  "
        f"scored = {report['n_scored']}  errors = {report['n_errors']}  match = {report['match_mode']}"
    )
    for name, val in report["recall_at_k"].items():
        print(f"  {name} = {val}")
    print(f"  MRR = {report['mrr']}")
    print(f"  latency p50/p95 ms = {report['latency_ms']['p50']}/{report['latency_ms']['p95']}")
    if report["cost_usd"] is not None:
        print(f"  cost/query (est) = ${report['cost_usd']['mean_per_query']}")


if __name__ == "__main__":
    main()
