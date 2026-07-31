"""Live EDGAR answer-accuracy eval — the missing metric for the product path.

The FinanceBench harness scores the pre-indexed benchmark corpus. The **live EDGAR
path** (the README's lead feature) had ZERO accuracy metrics — PROJECT-LOG §8A and the
design-doc's 2026-06-30 amendment both admit "the live path is not auto-scored." This
is that missing eval: a small, hand-verified question set run through the SAME
`LiveEngine` the API/frontend use, scored with the S2 answer-accuracy scorer (numeric
matcher + LLM judge, `answer_accuracy.py`). It is EXPECTED to score below the
FinanceBench corpus — messy live HTML, no per-filing tuning, a single filing in
context — and that gap is reported honestly, not hidden.

Gold answers were verified BY HAND against the actual filing each question targets:
the filing text was fetched via `edgar/client`, and the value/fact transcribed from
the income statement, cover page, or 8-K body (see each row's `source`). Because the
live path always fetches the NEWEST filing of a form, a gold can go stale when a
company files a newer one. Each row records the `gold_filing_date` it was verified
against; the harness flags any question whose live `filing_date` differs ("filing
drift") and reports accuracy both including and excluding drifted rows, so a rotated
filing cannot silently corrupt the number.

Account/billing outages abort the run (same fail-fast guardrail as the FinanceBench
runner, `eval/errors.py`) so a partial can't masquerade as a result.

  python -m sec_rag.eval.live_edgar                      # all questions, config v2
  python -m sec_rag.eval.live_edgar --limit 5            # first N (debug)
  python -m sec_rag.eval.live_edgar --questions path.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sec_rag.config import Config, Secrets, load_config
from sec_rag.edgar.live_engine import LiveEngine
from sec_rag.eval.answer_accuracy import AnswerScore, score_answer
from sec_rag.eval.errors import fatal_reason
from sec_rag.eval.run_financebench import _accuracy_block

_QUESTIONS_PATH = Path(__file__).parent / "live_edgar_questions.jsonl"


def load_live_questions(path: str | Path = _QUESTIONS_PATH) -> list[dict]:
    """Read the hand-verified question set; skips the leading ``_meta`` row."""
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    return [r for r in rows if "_meta" not in r]


def _answer_via_live(engine: LiveEngine, q: dict) -> tuple[str, str | None, dict]:
    """Run one question through the live streaming path; return (answer, filing_date,
    metrics). ``filing_date`` is the date of the filing actually retrieved (for drift
    detection). Consumes the stream and reads the final ``done`` event. Uses the
    engine's own keys/embedder (no BYOK override — this is the maintainer's eval)."""
    answer, filing_date, metrics = "", None, {}
    for ev in engine.stream(q["ticker"], q["question"], form=q["form"]):
        if ev.get("type") == "done":
            resp = ev["response"]
            answer = resp.answer
            if resp.citations:
                filing_date = resp.citations[0].filing_date
            m = resp.metrics
            metrics = {"latency_ms": m.latency_ms, "cost_usd": m.cost_usd,
                       "chunks_retrieved": m.chunks_retrieved}
    return answer, filing_date, metrics


def run(cfg: Config, limit: int | None = None, judge_model: str | None = None,
        sleep_s: float = 0.0, questions_path: str | Path = _QUESTIONS_PATH) -> dict:
    secrets = Secrets()
    judge = judge_model or cfg.generation.model
    questions = load_live_questions(questions_path)
    if limit is not None:
        questions = questions[:limit]

    engine = LiveEngine(cfg, secrets)
    results: list[dict] = []
    scores_all: list[AnswerScore] = []
    scores_fresh: list[AnswerScore] = []           # excludes filing-drifted rows
    by_form: dict[str, list[AnswerScore]] = defaultdict(list)
    latencies: list[float] = []
    costs: list[float] = []
    errors: list[dict] = []
    aborted: dict | None = None

    for i, q in enumerate(questions):
        if sleep_s and i > 0:
            time.sleep(sleep_s)
        try:
            answer, filing_date, metrics = _answer_via_live(engine, q)
            sc = score_answer(q["question"], q["gold"], answer, judge_model=judge, secrets=secrets)
        except Exception as e:  # noqa: BLE001
            # A credit/quota outage fails every remaining question identically -> abort
            # rather than emit accuracy over the prefix (eval/errors.py, same rule as
            # the FinanceBench runner). Transient errors are recorded + skipped.
            reason = fatal_reason(e)
            if reason:
                aborted = {"id": q["id"], "reason": reason, "error": f"{type(e).__name__}: {e}"}
                break
            errors.append({"id": q["id"], "error": f"{type(e).__name__}: {e}"})
            continue

        drift = filing_date is not None and filing_date != q["gold_filing_date"]
        scores_all.append(sc)
        if not drift:
            scores_fresh.append(sc)
            by_form[q["form"]].append(sc)
        if metrics.get("latency_ms") is not None:
            latencies.append(metrics["latency_ms"])
        if metrics.get("cost_usd") is not None:
            costs.append(metrics["cost_usd"])
        results.append({
            "id": q["id"], "ticker": q["ticker"], "form": q["form"],
            "question": q["question"], "gold": q["gold"], "answer": answer,
            "gold_filing_date": q["gold_filing_date"], "live_filing_date": filing_date,
            "filing_drift": drift,
            "refused": sc.refused, "numeric": sc.numeric, "llm_correct": sc.llm_correct,
            **({"latency_ms": metrics.get("latency_ms"), "cost_usd": metrics.get("cost_usd")}),
        })

    engine_close = getattr(engine, "close", None)
    if callable(engine_close):
        engine_close()

    n_drift = sum(1 for r in results if r["filing_drift"])
    complete = aborted is None and not errors and len(scores_all) == len(questions)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "live_edgar_accuracy",
        "complete": complete,                 # False -> partial, DO NOT CITE
        "aborted": aborted,
        "n_questions": len(questions),
        "n_scored": len(scores_all),
        "n_errors": len(errors),
        "errors": errors,
        "filing_drift_count": n_drift,        # rows whose live filing != the verified one
        "judge_model": judge,
        "config": {
            "embedding_model": cfg.embedding.model,
            "generation_model": cfg.generation.model,
            "top_k": cfg.retrieval.top_k,
            "chunking": cfg.chunking.model_dump(),
        },
        "note": "Live EDGAR path scored with the S2 answer scorer. 'fresh' excludes "
        "filing-drifted rows (live filing newer than the hand-verified gold). Expected "
        "below the FinanceBench corpus — messy live HTML, single filing in context.",
        # Primary number: accuracy over the non-drifted (still-valid-gold) rows.
        "accuracy_fresh": _accuracy_block(scores_fresh),
        "accuracy_all": _accuracy_block(scores_all),
        "per_form": {f: _accuracy_block(s) for f, s in sorted(by_form.items())},
        "cost_usd": {
            "mean_per_query": round(sum(costs) / len(costs), 6) if costs else None,
            "total": round(sum(costs), 6) if costs else None,
        },
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
        },
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Live EDGAR answer-accuracy eval -> JSON")
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--questions", default=str(_QUESTIONS_PATH))
    ap.add_argument("--limit", type=int, default=None, help="first N questions (debug)")
    ap.add_argument("--judge-model", default=None,
                    help="LLM correctness judge (default: generation model)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between questions (rate-limit throttle)")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = run(cfg, limit=args.limit, judge_model=args.judge_model,
                 sleep_s=args.sleep, questions_path=args.questions)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"live_edgar_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))

    fresh, allb = report["accuracy_fresh"], report["accuracy_all"]
    print(f"Wrote {out_path}")
    print(f"  n = {report['n_questions']}  scored = {report['n_scored']}  "
          f"errors = {report['n_errors']}  filing_drift = {report['filing_drift_count']}")
    print(f"  LLM accuracy (fresh) = {fresh['llm']['accuracy']}  "
          f"({fresh['llm']['n_correct']}/{fresh['n_answered']} answered)")
    print(f"  numeric (fresh)      = {fresh['numeric']['accuracy']}  "
          f"({fresh['numeric']['n_correct']}/{fresh['numeric']['n_applicable']} figure-golds)")
    print(f"  refusal (fresh)      = {fresh['refusal_rate']}  ({fresh['n_refused']}/{fresh['n']})")
    print(f"  LLM accuracy (all incl. drift) = {allb['llm']['accuracy']}")
    for form, blk in report["per_form"].items():
        print(f"  [{form}] LLM acc = {blk['llm']['accuracy']}  "
              f"({blk['llm']['n_correct']}/{blk['n_answered']})")
    if report["cost_usd"]["mean_per_query"] is not None:
        print(f"  cost/query = ${report['cost_usd']['mean_per_query']}  "
              f"latency mean = {report['latency_ms']['mean']} ms")

    if not report["complete"]:
        ab = report.get("aborted")
        print("\n" + "=" * 72, file=sys.stderr)
        print("⚠️  INCOMPLETE RUN — DO NOT CITE THIS ARTIFACT AS A RESULT", file=sys.stderr)
        if ab:
            print(f"   aborted early: {ab['reason']}", file=sys.stderr)
            print(f"   first fatal error (q={ab['id']}): {ab['error']}", file=sys.stderr)
        print(f"   n_scored={report['n_scored']}/{report['n_questions']}  "
              f"n_errors={report['n_errors']}", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
