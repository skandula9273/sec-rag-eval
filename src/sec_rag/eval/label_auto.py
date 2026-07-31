"""LLM auto-labeler for the metric-validity sample (a PROXY for human labels).

The metric-validity study is built around HUMAN labels; this labels the same 50 pairs
with a Claude judge instead. That is a weaker claim — an LLM grading the matchers is
not the independent human judgement the study is designed to get, and it shares
failure modes with the semantic matcher (both are learned language models). It is
enabled explicitly, and its output is kept in a SEPARATE file so a real human pass can
still be run and compared. Results derived from this MUST be labeled "LLM-adjudicated",
never "human".

The judge sees exactly what the human CLI shows — question, gold answer, gold
evidence, chunk — and NOT the matcher verdicts, so it is not primed. One call per
pair; resumable.

  python -m sec_rag.eval.label_auto            # -> eval_results/metric_validity_labels_llm.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sec_rag.config import Secrets
from sec_rag.eval.label_cli import _labeled_ids, _load_sample

_SYSTEM = (
    "You judge retrieval quality for a financial-QA benchmark. Given a QUESTION, its "
    "known-correct GOLD ANSWER, the GOLD EVIDENCE (the span from the filing that "
    "supports that answer), and a retrieved CHUNK, decide one thing: does the CHUNK "
    "SUPPORT the gold answer — i.e., does it contain the information a reader would "
    "need to produce or verify the gold answer? Say yes if that supporting information "
    "is present, even if worded differently or laid out as a table; say no if the "
    "chunk is about something else or lacks it. Reply with EXACTLY one word: yes or no."
)


def _judge(pair: dict, model: str, secrets: Secrets) -> int:
    from anthropic import Anthropic

    client = Anthropic(api_key=secrets.anthropic_api_key)
    evidence = "\n".join(e for e in (pair.get("gold_evidence") or []) if e and e.strip())
    user = (
        f"QUESTION:\n{pair['question']}\n\nGOLD ANSWER:\n{pair.get('gold_answer')}\n\n"
        f"GOLD EVIDENCE:\n{evidence}\n\nCHUNK:\n{pair['chunk_content']}"
    )
    msg = client.messages.create(
        model=model, max_tokens=5, temperature=0.0,
        system=_SYSTEM, messages=[{"role": "user", "content": user}],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    text = "".join(parts).strip().lower()
    return 1 if text.startswith("y") else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM auto-labeler (proxy for human labels)")
    ap.add_argument("--sample", default="eval_results/metric_validity_sample.jsonl")
    ap.add_argument("--labels", default="eval_results/metric_validity_labels_llm.jsonl")
    ap.add_argument("--model", default="claude-haiku-4-5")
    args = ap.parse_args()

    secrets = Secrets()
    secrets.require("anthropic_api_key")
    pairs = _load_sample(args.sample)
    done = _labeled_ids(args.labels)
    todo = [p for p in pairs if p["pair_id"] not in done]
    print(f"Auto-labeling {len(todo)}/{len(pairs)} pairs with judge={args.model} "
          "(proxy for human).")

    out = Path(args.labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("a") as f:
        for p in todo:
            supports = _judge(p, args.model, secrets)
            f.write(json.dumps({
                "pair_id": p["pair_id"],
                "supports": supports,
                "note": "LLM-adjudicated (proxy, not human)",
                "judge_model": args.model,
                "labeled_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
            f.flush()
            n += 1
    print(f"Wrote {n} labels -> {out}  (total {len(done) + n}/{len(pairs)})")
    print("NOTE: LLM labels are a proxy. A human pass -> metric_validity_labels.jsonl "
          "still supersedes.")


if __name__ == "__main__":
    main()
