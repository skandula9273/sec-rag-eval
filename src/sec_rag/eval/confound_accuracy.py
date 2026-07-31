"""Answer accuracy per grid cell — the CHUNK-INVARIANT tiebreak.

Recall under the overlap matcher can be inflated by chunk size; answer accuracy
cannot (it scores whether the generated answer matches the gold, not evidence
overlap). So accuracy across cells tells us how much of a recall gain is a REAL
system improvement. This generates + scores one cell at a time (300 Anthropic calls
each), reusing the grid's cached corpus embeddings — run per arm so a credit-out only
costs the current arm.

  python -m sec_rag.eval.confound_accuracy --arm 3-small/512
  python -m sec_rag.eval.confound_accuracy --arm 3-large/512
  python -m sec_rag.eval.confound_accuracy --arm 3-large/1024

Uses the SAME shipped generation pipeline (concise prompt, Haiku) and the same
accuracy scorer as S2, at top_k=10 so it lines up with the recall grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sec_rag.config import EmbeddingConfig, Secrets, load_config
from sec_rag.eval.answer_accuracy import score_answer
from sec_rag.eval.confound_grid import MODELS, _normalize, cell_corpus
from sec_rag.eval.errors import fatal_reason
from sec_rag.eval.run_financebench import _accuracy_block
from sec_rag.generate.answer import generate_answer
from sec_rag.ingest.chunk import tiktoken_encoder
from sec_rag.ingest.embed import Embedder
from sec_rag.ingest.financebench import load_questions
from sec_rag.retrieve.dense import RetrievedChunk

TOP_K = 10


def run_arm(arm: str, judge_model: str | None, sleep_s: float) -> dict:
    model_name, chunk = arm.split("/")
    chunk = int(chunk)
    cfg = load_config("configs/v2.yaml")           # shipped generation config
    secrets = Secrets()
    judge = judge_model or cfg.generation.model
    questions = load_questions(cfg.eval.dataset)
    enc = tiktoken_encoder(cfg.chunking.encoder)

    items, V = cell_corpus(model_name, MODELS[model_name], chunk, questions, enc,
                           cfg.chunking.strategy, secrets)
    emb = Embedder(EmbeddingConfig(provider="openai", model=MODELS[model_name], dim=1536,
                                   batch_size=128), secrets)
    Q = _normalize(np.asarray([emb.embed_one(q.question) for q in questions], dtype=np.float32))
    sims = V @ Q.T

    scores, by_cat, errors = [], defaultdict(list), []
    aborted: dict | None = None
    for j, q in enumerate(questions):
        col = sims[:, j]
        idx = np.argpartition(-col, TOP_K)[:TOP_K]
        idx = idx[np.argsort(-col[idx])]
        chunks = [
            RetrievedChunk(chunk_id=int(i), doc_name=items[i][1], ticker=None,
                           filing_type=None, filing_date=None, page=None,
                           section=items[i][2], content=items[i][0],
                           retrieval_score=float(col[i]))
            for i in idx
        ]
        try:
            gen = generate_answer(q.question, chunks, cfg.generation, secrets)
            sc = score_answer(q.question, q.answer, gen.text, judge_model=judge, secrets=secrets)
        except Exception as e:  # noqa: BLE001
            # A credit-out / quota-out here would fail every remaining question the
            # same way (this arm died at 66/150 exactly this way). Abort rather than
            # emit an accuracy block over the prefix that ran — same rule as the main
            # runner (sec_rag/eval/errors.py). Transient errors are recorded + skipped.
            reason = fatal_reason(e)
            if reason:
                aborted = {"id": q.id, "reason": reason, "error": f"{type(e).__name__}: {e}"}
                break
            errors.append({"id": q.id, "error": f"{type(e).__name__}: {e}"})
            continue
        scores.append(sc)
        by_cat[q.question_type or "uncategorized"].append(sc)
        if sleep_s:
            time.sleep(sleep_s)

    complete = aborted is None and not errors and len(scores) == len(questions)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "confound_accuracy_arm",
        "complete": complete,          # False -> partial arm, DO NOT CITE
        "aborted": aborted,            # None, or {id, reason, error} if fatal
        "arm": arm,
        "top_k": TOP_K,
        "judge_model": judge,
        "n_questions": len(questions),
        "n_scored": len(scores),
        "n_errors": len(errors),
        "errors": errors,
        "overall": _accuracy_block(scores),
        "per_category": {c: _accuracy_block(s) for c, s in sorted(by_cat.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer accuracy for one grid cell (tiebreak)")
    ap.add_argument("--arm", required=True, help="e.g. 3-small/512, 3-large/512, 3-large/1024")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    report = run_arm(args.arm, args.judge_model, args.sleep)
    out = Path("eval_results")
    out.mkdir(exist_ok=True)
    tag = args.arm.replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = out / f"confound_accuracy_{tag}_{stamp}.json"
    p.write_text(json.dumps(report, indent=2))
    o = report["overall"]
    print(f"Wrote {p}")
    print(f"  arm={args.arm} scored={report['n_scored']}/{report['n_questions']} "
          f"errors={report['n_errors']}")
    print(f"  LLM acc over-all={o['llm']['accuracy_over_all']} (attempted {o['llm']['accuracy']}) "
          f"refusal={o['refusal_rate']} numeric={o['numeric']['accuracy']}")

    if not report["complete"]:
        ab = report.get("aborted")
        print("\n" + "=" * 72, file=sys.stderr)
        print("⚠️  INCOMPLETE ARM — DO NOT CITE THIS ARTIFACT AS A RESULT", file=sys.stderr)
        if ab:
            print(f"   aborted early: {ab['reason']}", file=sys.stderr)
            print(f"   first fatal error (q={ab['id']}): {ab['error']}", file=sys.stderr)
        print(f"   scored {report['n_scored']}/{report['n_questions']} -> "
              "accuracy above is PARTIAL.", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
