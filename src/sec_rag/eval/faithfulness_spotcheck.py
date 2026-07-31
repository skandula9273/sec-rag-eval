"""Faithfulness-judge spot-check (design-doc L3 / 'LLM-judge bias' mitigation).

The committed faithfulness mean (0.929, eval_results/financebench_20260629T193049Z
.json) comes from an automated Haiku judge grading a Haiku-generated answer — the
same model family on both sides. The design doc commits to a **manual spot-check
of ~20 judgments per eval run, reporting the agreement rate** (design-doc.md L3 +
the 'LLM-judge bias' risk), precisely to catch that self-grading bias. That
spot-check had never been produced.

This script produces the raw material for it, DETERMINISTICALLY: it samples the
same 20 questions the runner would (seed 13, the identical ``_select``), runs the
SAME pipeline the committed eval used (v2 config, top_k=5, temperature 0), and
captures per question:

  * the generated answer,
  * the exact retrieved sources the answer was grounded in,
  * the Haiku judge's atomic-claim verdict ({n_claims, n_supported, score}).

It writes those triples to a JSON artifact. It does NOT self-certify: an
independent reviewer then reads each triple and adjudicates by hand, and the
agreement rate is recorded in docs/. Regenerating (rather than replaying) is a
disclosed limitation — the committed run stored only the aggregate mean, not
per-question answers; temperature 0 keeps it reproducible, and 'store per-question
detail' is a recommendation that falls out of this exercise.

Usage:
  python -m sec_rag.eval.faithfulness_spotcheck --config configs/v2.yaml -n 20
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from sec_rag.config import load_config
from sec_rag.eval.run_financebench import _select
from sec_rag.generate.faithfulness import score_faithfulness
from sec_rag.ingest.financebench import load_questions
from sec_rag.pipeline import QueryEngine


def run(config: str, n: int, sleep_s: float) -> dict:
    cfg = load_config(config)
    top_k = cfg.retrieval.top_k
    questions = _select(load_questions(cfg.eval.dataset), n, cfg.eval.seed)

    engine = QueryEngine(cfg)
    records: list[dict] = []
    errors: list[dict] = []
    try:
        for i, q in enumerate(questions):
            if sleep_s and i > 0:
                time.sleep(sleep_s)
            try:
                # with_faithfulness=False: we call the judge explicitly below so we
                # keep its raw claim counts, not just the collapsed score.
                res = engine.run(q.question, top_k=top_k, with_faithfulness=False)
                answer = res.response.answer
                retrieved = res.retrieved
                fr = score_faithfulness(answer, retrieved, cfg.generation, engine.secrets)
            except Exception as exc:  # noqa: BLE001 — one bad question must not sink the run
                errors.append({"id": q.id, "error": f"{type(exc).__name__}: {exc}"})
                continue

            records.append(
                {
                    "id": q.id,
                    "question_type": q.question_type,
                    "question": q.question,
                    "gold_answer": q.answer,
                    "generated_answer": answer,
                    "sources": [
                        {
                            "i": j,
                            "doc_name": c.doc_name,
                            "page": c.page,
                            "section": c.section,
                            "content": c.content,
                        }
                        for j, c in enumerate(retrieved, start=1)
                    ],
                    "judge": {
                        "n_claims": fr.n_claims,
                        "n_supported": fr.n_supported,
                        "score": fr.score,
                    },
                }
            )
    finally:
        engine.close()

    scores = [r["judge"]["score"] for r in records]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "faithfulness_spotcheck",
        "purpose": "manual/independent adjudication of the Haiku faithfulness judge "
        "(design-doc L3 LLM-judge-bias mitigation)",
        "config_file": config,
        "config": {
            "embedding_model": cfg.embedding.model,
            "chunking": cfg.chunking.model_dump(),
            "generation_model": cfg.generation.model,
            "top_k": top_k,
            "seed": cfg.eval.seed,
        },
        "n_requested": n,
        "n_scored": len(records),
        "n_errors": len(errors),
        "errors": errors,
        "judge_mean_faithfulness": round(sum(scores) / len(scores), 4) if scores else None,
        "records": records,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Faithfulness-judge spot-check -> JSON")
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("-n", "--n", type=int, default=20, help="sample size (seeded)")
    ap.add_argument("--sleep", type=float, default=0.5, help="pause between questions (rate limit)")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args.config, args.n, args.sleep)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"faithfulness_spotcheck_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote {out_path}")
    print(
        f"  scored = {report['n_scored']}/{report['n_requested']}  "
        f"errors = {report['n_errors']}  "
        f"judge mean faithfulness = {report['judge_mean_faithfulness']}"
    )


if __name__ == "__main__":
    main()
