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
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sec_rag.config import Config, load_config
from sec_rag.eval.answer_accuracy import AnswerScore, score_answer
from sec_rag.eval.errors import fatal_reason
from sec_rag.eval.metrics import evidence_match_rank, hit_rate_at_k, mean_reciprocal_rank
from sec_rag.generate.answer import PRICING
from sec_rag.ingest.financebench import Question, load_questions
from sec_rag.pipeline import QueryEngine


def _accuracy_block(scores: list[AnswerScore]) -> dict:
    """Aggregate answer-accuracy over a set of scored questions.

    Refusals are reported separately and EXCLUDED from the accuracy denominator — an
    unanswered question is not a wrong one. LLM accuracy = correct / answered; the
    numeric matcher only applies to single-figure golds, so its denominator is the
    subset it can score. ``accuracy_over_all`` (correct / everything) is shown too so
    a high refusal rate can't quietly inflate the headline accuracy."""
    n = len(scores)
    answered = [s for s in scores if not s.refused]
    n_ref = n - len(answered)
    llm_correct = sum(1 for s in answered if s.llm_correct)
    num_applic = [s for s in answered if s.numeric is not None]
    num_correct = sum(1 for s in num_applic if s.numeric)
    return {
        "n": n,
        "n_refused": n_ref,
        "refusal_rate": round(n_ref / n, 4) if n else None,
        "n_answered": len(answered),
        "llm": {
            "n_graded": len(answered),
            "n_correct": llm_correct,
            "accuracy": round(llm_correct / len(answered), 4) if answered else None,
            "accuracy_over_all": round(llm_correct / n, 4) if n else None,
        },
        "numeric": {
            "n_applicable": len(num_applic),
            "n_correct": num_correct,
            "accuracy": round(num_correct / len(num_applic), 4) if num_applic else None,
        },
    }


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
    score_accuracy: bool = False,
    accuracy_judge_model: str | None = None,
    top_k: int | None = None,
) -> dict:
    ks = sorted(cfg.eval.recall_ks)
    # Retrieval/generation depth. Defaults to the deepest recall_k; --top-k overrides
    # it (and is added to the reported ks) so depth can be swept without a config edit.
    depth = top_k or max(ks)
    if top_k:
        ks = sorted(set(ks) | {top_k})
    questions = _select(load_questions(cfg.eval.dataset), limit, cfg.eval.seed)
    # Accuracy needs the generated answer, so it forces the full pipeline; the judge
    # model is recorded in the output. Faithfulness is turned off on an accuracy run
    # (the accuracy judge is the added call) — it is measured separately.
    if score_accuracy:
        retrieval_only = False
    judge_model = accuracy_judge_model or cfg.generation.model

    ranks: list[int | None] = []
    by_cat: dict[str, list[int | None]] = defaultdict(list)
    latencies: list[float] = []
    costs: list[float] = []
    faithfulness_scores: list[float] = []  # populated only when eval.faithfulness on
    acc_scores: list[AnswerScore] = []                          # accuracy: overall
    acc_by_cat: dict[str, list[AnswerScore]] = defaultdict(list)  # accuracy: per category
    misses_no_evidence = 0
    errors: list[dict] = []  # questions that failed twice on a *transient* error
    aborted: dict | None = None  # set iff a FATAL error stopped the run early

    # A *transient* failure on one question (a dropped Neon socket, a one-off
    # Anthropic timeout, a momentary rate limit) must not throw away a 150-question
    # run that is otherwise complete: each question gets a bounded retry on a FRESH
    # engine — a dead pooled connection is the failure we already hit during ingest,
    # and a new engine reconnects — and anything still failing is recorded + skipped.
    # But an *account-level* failure (Anthropic credit-out, OpenAI quota-out, a bad
    # key) is NOT transient — every remaining question would fail identically. We do
    # NOT grind through it emitting an aggregate over the lucky prefix (that partial
    # would look like a real result — the bug this guards against, CLAUDE.md rule 2
    # + the "eval runner swallows infra failures" debt). fatal_reason() detects it and
    # we abort at once, marking the report incomplete. Failures are counted and
    # disclosed either way (rule 2: honest numbers).
    # Normalise one question to (contents, latency_ms, cost, faithfulness).
    # retrieval_only stops after retrieval (no Anthropic call): recall@k / MRR are
    # pure retrieval metrics, so they need only the query embedding + the DB. cost
    # and faithfulness are then None — there was no generation to price or judge.
    # The retrieval path is identical to full mode, so the recall is the same.
    def _evaluate(engine: QueryEngine, q: Question):
        if retrieval_only:
            chunks, retr_ms = engine.retrieve(q.question, top_k=depth)
            return [c.content for c in chunks], retr_ms, None, None, None
        # Accuracy folds its judge call into this retried unit, so a judge failure
        # retries with the rest of the question rather than crashing the run.
        wf = False if score_accuracy else None  # accuracy run: faithfulness judge off
        res = engine.run(q.question, top_k=depth, with_faithfulness=wf)
        m = res.response.metrics
        acc = None
        if score_accuracy:
            acc = score_answer(q.question, q.answer, res.response.answer,
                               judge_model=judge_model, secrets=engine.secrets)
        excerpts = [c.excerpt for c in res.response.citations]
        return excerpts, m.latency_ms, m.cost_usd, m.faithfulness, acc

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
                contents, latency, cost, faith, acc = _evaluate(engine, q)
            except Exception as first_exc:
                # An account/billing/auth error is unrecoverable — abort now rather
                # than retry into the same wall and grind out a misleading partial.
                reason = fatal_reason(first_exc)
                if reason:
                    aborted = {"id": q.id, "reason": reason,
                               "error": f"{type(first_exc).__name__}: {first_exc}"}
                    break
                # Transient: rebuild the engine (new DB connection) and try once more.
                try:
                    engine.close()
                except Exception:
                    pass
                try:
                    engine = QueryEngine(cfg)
                    contents, latency, cost, faith, acc = _evaluate(engine, q)
                except Exception as second_exc:
                    reason = fatal_reason(second_exc)
                    if reason:  # the retry surfaced (or hit) a fatal error -> abort
                        aborted = {"id": q.id, "reason": reason,
                                   "error": f"{type(second_exc).__name__}: {second_exc}"}
                        break
                    errors.append({"id": q.id, "error": f"{type(second_exc).__name__}: {second_exc}"})
                    continue
            if not q.evidence_texts:
                misses_no_evidence += 1
            rank = evidence_match_rank(contents, q.evidence_texts, mode=match_mode)
            ranks.append(rank)
            cat = q.question_type or "uncategorized"
            by_cat[cat].append(rank)
            latencies.append(latency)
            if cost is not None:
                costs.append(cost)
            if faith is not None:
                faithfulness_scores.append(faith)
            if acc is not None:
                acc_scores.append(acc)
                acc_by_cat[cat].append(acc)
    finally:
        engine.close()

    # A run is citable only if EVERY attempted question produced a result: no fatal
    # abort, no per-question errors, ranks for all of them. Anything less is a partial
    # and its aggregates are over a biased sample — the `complete` flag + a non-zero
    # exit in main() make that impossible to cite by accident (CLAUDE.md: "don't cite
    # an eval JSON whose n_scored < n_questions or n_errors > 0").
    complete = aborted is None and not errors and len(ranks) == len(questions)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "complete": complete,                # False -> partial run, DO NOT CITE
        "aborted": aborted,                  # None, or {id, reason, error} if fatal
        "n_questions": len(questions),       # questions attempted
        "n_scored": len(ranks),              # questions that produced a result
        "match_mode": match_mode,
        "mode": "accuracy" if score_accuracy else ("retrieval_only" if retrieval_only else "full"),
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
        # Answer accuracy vs FinanceBench gold (null unless --accuracy). Refusal rate
        # is reported separately and excluded from the accuracy denominator.
        "answer_accuracy": None if not score_accuracy else {
            "judge_model": judge_model,
            "numeric_normalizer": "src/sec_rag/eval/answer_accuracy.py (rules in the docstring)",
            "overall": _accuracy_block(acc_scores),
            "per_category": {c: _accuracy_block(s) for c, s in sorted(acc_by_cat.items())},
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
    ap.add_argument(
        "--accuracy",
        action="store_true",
        help="also score answer accuracy vs FinanceBench gold (numeric matcher + LLM "
        "judge), overall + per category, with refusal rate reported separately",
    )
    ap.add_argument(
        "--accuracy-judge-model",
        default=None,
        help="LLM correctness judge model (default: the generation model); "
        "recorded in the JSON",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="retrieval/generation depth (default: deepest recall_k); sweep it to "
        "trade recall coverage against cost/latency",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = run(
        cfg,
        limit=args.limit,
        match_mode=args.match_mode,
        sleep_s=args.sleep,
        retrieval_only=args.no_generate,
        score_accuracy=args.accuracy,
        accuracy_judge_model=args.accuracy_judge_model,
        top_k=args.top_k,
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
    acc = report.get("answer_accuracy")
    if acc:
        o = acc["overall"]
        L, N = o["llm"], o["numeric"]
        print(f"  accuracy (judge {acc['judge_model']}):")
        print(f"    LLM     = {L['accuracy']}  ({L['n_correct']}/{o['n_answered']} answered)")
        print(f"    numeric = {N['accuracy']}  ({N['n_correct']}/{N['n_applicable']} figure-golds)")
        print(f"    refusal = {o['refusal_rate']}  ({o['n_refused']}/{o['n']})")

    # A partial run must never be mistaken for a result: shout, and exit non-zero so
    # `make eval`/CI fail loudly instead of a biased aggregate slipping through.
    if not report["complete"]:
        ab = report.get("aborted")
        print("\n" + "=" * 72, file=sys.stderr)
        print("⚠️  INCOMPLETE RUN — DO NOT CITE THIS ARTIFACT AS A RESULT", file=sys.stderr)
        if ab:
            print(f"   aborted early: {ab['reason']}", file=sys.stderr)
            print(f"   first fatal error (q={ab['id']}): {ab['error']}", file=sys.stderr)
        print(
            f"   n_scored={report['n_scored']}/{report['n_questions']}  "
            f"n_errors={report['n_errors']}  "
            "-> aggregate metrics above are over a PARTIAL sample and are not valid.",
            file=sys.stderr,
        )
        print("=" * 72, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
