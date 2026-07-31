"""Human labeling CLI for the metric-validity study.

Shows, for each sampled (question, chunk) pair: the question, the gold answer, the
gold evidence, and the retrieved chunk — and asks ONE binary question: does this
chunk support the gold answer? Plus an optional free-text note.

It deliberately does NOT show the automatic matchers' verdicts, so the label is an
independent human judgement, not a rubber-stamp of a matcher. Labels append to a
JSONL and the tool is resumable (skips already-labeled pairs), so you can label in
sittings. There is no auto-labeling — every label comes from your keypress.

  python -m sec_rag.eval.label_cli                 # label interactively
  python -m sec_rag.eval.label_cli --preview       # render the next pair and exit
  python -m sec_rag.eval.label_cli --status        # how many labeled / remaining
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SAMPLE = "eval_results/metric_validity_sample.jsonl"
LABELS = "eval_results/metric_validity_labels.jsonl"

_RULE = "=" * 78


def _load_sample(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if "_meta" not in r]  # drop the provenance header


def _labeled_ids(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {json.loads(line)["pair_id"] for line in p.read_text().splitlines() if line.strip()}


def _render(pair: dict, idx: int, total: int) -> str:
    ev = "\n".join(f"  - {e}" for e in (pair.get("gold_evidence") or []) if e and e.strip())
    return (
        f"\n{_RULE}\nPair {idx}/{total}   [{pair['category']}]   {pair['pair_id']}\n{_RULE}\n"
        f"QUESTION:\n  {pair['question']}\n\n"
        f"GOLD ANSWER:\n  {pair.get('gold_answer')}\n\n"
        f"GOLD EVIDENCE (what a supporting chunk must back):\n{ev or '  (none)'}\n\n"
        f"RETRIEVED CHUNK (rank {pair['chunk_rank']}, {pair['doc_name']}):\n"
        f"{'-' * 78}\n{pair['chunk_content']}\n{'-' * 78}\n"
    )


def _prompt() -> tuple[int | None, str] | str:
    """Return (label, note) or a control string: 'skip' | 'quit'."""
    while True:
        q = "Does this chunk SUPPORT the gold answer?  [y]es / [n]o / [s]kip / [q]uit: "
        ans = input(q).strip().lower()
        if ans in ("y", "yes", "1"):
            return 1, input("  note (optional, Enter to skip): ").strip()
        if ans in ("n", "no", "0"):
            return 0, input("  note (optional, Enter to skip): ").strip()
        if ans in ("s", "skip"):
            return "skip"
        if ans in ("q", "quit", "exit"):
            return "quit"
        print("  please answer y / n / s / q")


def main() -> None:
    ap = argparse.ArgumentParser(description="Human labeling CLI (no auto-labeling)")
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--preview", action="store_true",
                    help="render the next unlabeled pair and exit")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    args = ap.parse_args()

    pairs = _load_sample(args.sample)
    done = _labeled_ids(args.labels)
    todo = [p for p in pairs if p["pair_id"] not in done]

    if args.status:
        print(f"labeled {len(done)}/{len(pairs)}  ({len(todo)} remaining)  -> {args.labels}")
        return
    if not todo:
        print(f"All {len(pairs)} pairs already labeled -> {args.labels}")
        return
    if args.preview:
        print(_render(todo[0], len(pairs) - len(todo) + 1, len(pairs)))
        print("(preview only — no label written; run without --preview to label)")
        return

    print(f"Labeling {len(todo)} of {len(pairs)} pairs. Resumable — quit anytime; "
          "progress is saved.")
    out = Path(args.labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    session, stopped = 0, False
    with out.open("a") as f:
        for i, pair in enumerate(todo, start=1):
            print(_render(pair, len(done) + i, len(pairs)))
            res = _prompt()
            if res == "quit":
                stopped = True
                break
            if res == "skip":
                continue
            label, note = res
            f.write(json.dumps({
                "pair_id": pair["pair_id"],
                "supports": label,
                "note": note,
                "labeled_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
            f.flush()
            session += 1
    msg = "Stopped" if stopped else "Done"
    print(f"\n{msg}. {session} labeled this session; "
          f"{len(done) + session}/{len(pairs)} total -> {args.labels}")


if __name__ == "__main__":
    main()
