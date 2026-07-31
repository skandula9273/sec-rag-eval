"""Draw a labeling sample stratified by category AND matcher disagreement.

The matcher-validity study needs human labels, but a RANDOM sample of
(question, retrieved_chunk) pairs is ~all "no chunk matched anything" — easy
agreements that waste the labeling budget. The information lives where the three
matchers (strict / overlap / semantic) DISAGREE: those pairs decide which matcher
tracks human judgement. So we oversample disagreement, and stratify across the three
FinanceBench categories, with a fixed seed.

Retrieves top-k per question (v2 config), computes each matcher's verdict per pair,
classes each pair (unanimous_yes / unanimous_no / disagree), and samples a target
mix. The committed sample records the per-matcher verdicts (needed for scoring) but
the labeling CLI hides them so the human isn't primed.

  python -m sec_rag.eval.label_sample            # -> eval_results/metric_validity_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from sec_rag.config import Secrets, load_config
from sec_rag.eval.metrics import load_matchers
from sec_rag.ingest.financebench import load_questions
from sec_rag.pipeline import QueryEngine

TOP_K = 10                       # retrieval depth the pairs are drawn from
TARGET = {"disagree": 30, "unanimous_yes": 10, "unanimous_no": 10}  # ~50, disagree-heavy


def _agreement(verdicts: dict[str, bool]) -> str:
    vals = set(verdicts.values())
    if vals == {True}:
        return "unanimous_yes"
    if vals == {False}:
        return "unanimous_no"
    return "disagree"


def _build_pairs(config: str, matchers_path: str) -> list[dict]:
    cfg = load_config(config)
    matchers = load_matchers(matchers_path, Secrets())
    questions = load_questions(cfg.eval.dataset)

    engine = QueryEngine(cfg)
    retrieved: list[tuple] = []  # (question, list[chunk_content])
    try:
        for q in questions:
            chunks, _ = engine.retrieve(q.question, top_k=TOP_K)
            retrieved.append((q, [c.content for c in chunks]))
    finally:
        engine.close()

    # Warm semantic matchers once over everything they will score.
    corpus = [c for _, cs in retrieved for c in cs]
    evidence = [e for q, _ in retrieved for e in (q.evidence_texts or [])]
    for m in matchers.values():
        m.warm(corpus + evidence)

    pairs = []
    for q, contents in retrieved:
        # Only pairs where the gold evidence exists can inform "does it support".
        if not any(e and e.strip() for e in (q.evidence_texts or [])):
            continue
        for rank, content in enumerate(contents, start=1):
            verdicts = {name: bool(m.matches(content, q.evidence_texts))
                        for name, m in matchers.items()}
            pairs.append({
                "pair_id": f"{q.id}__r{rank}",
                "category": q.question_type or "uncategorized",
                "question": q.question,
                "gold_answer": q.answer,
                "gold_evidence": q.evidence_texts,
                "doc_name": q.doc_name,
                "chunk_rank": rank,
                "chunk_content": content,
                "matchers": verdicts,
                "agreement": _agreement(verdicts),
            })
    return pairs


def _stratified_take(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Take ~n from pool, balanced across the three categories (round-robin), seeded."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in pool:
        by_cat[p["category"]].append(p)
    for cat in by_cat:
        rng.shuffle(by_cat[cat])
    picked, cats = [], sorted(by_cat)
    i = 0
    while len(picked) < n and any(by_cat[c] for c in cats):
        c = cats[i % len(cats)]
        if by_cat[c]:
            picked.append(by_cat[c].pop())
        i += 1
    return picked


def sample(config: str, matchers_path: str, seed: int) -> tuple[list[dict], dict]:
    pairs = _build_pairs(config, matchers_path)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_class[p["agreement"]].append(p)

    rng = random.Random(seed)
    chosen: list[dict] = []
    for cls, target in TARGET.items():
        chosen += _stratified_take(by_class.get(cls, []), target, rng)
    rng.shuffle(chosen)  # present in mixed order (labeler can't infer class)

    stats = {
        "n_pairs_total": len(pairs),
        "pairs_by_class": {c: len(v) for c, v in sorted(by_class.items())},
        "sample_target": TARGET,
        "sample_by_class": {
            c: sum(1 for p in chosen if p["agreement"] == c)
            for c in sorted({p["agreement"] for p in chosen})
        },
        "sample_by_category": {
            c: sum(1 for p in chosen if p["category"] == c)
            for c in sorted({p["category"] for p in chosen})
        },
        "n_sampled": len(chosen),
    }
    return chosen, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw a disagreement-stratified labeling sample")
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--matchers", default="configs/matchers.yaml")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="eval_results/metric_validity_sample.jsonl")
    args = ap.parse_args()

    chosen, stats = sample(args.config, args.matchers, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        # Header line carries provenance; each following line is one pair to label.
        f.write(json.dumps({"_meta": {"seed": args.seed, "top_k": TOP_K, **stats}}) + "\n")
        for p in chosen:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {out}  ({len(chosen)} pairs)")
    print("  by class:", stats["sample_by_class"])
    print("  by category:", stats["sample_by_category"])
    print("  disagree pairs available:", stats["pairs_by_class"].get("disagree", 0))


if __name__ == "__main__":
    main()
