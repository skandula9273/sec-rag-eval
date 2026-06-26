"""Fusion-weight ablation (V1.1b): is hybrid salvageable, or is dense the ceiling?

The V1.1 hybrid A/B regressed (recall@5 0.44 -> 0.347) across every category,
including the tables category it was meant to fix. Hypothesis: equal-weight RRF
fuses a noisy Postgres-FTS list into a stronger dense list and drags it down.
This sweeps the fusion weight to test it.

Efficiency: the dense and lexical CANDIDATE lists for a question do not depend on
the fusion weight, so we retrieve them ONCE per question and re-fuse in memory
across the weight sweep. Re-running retrieval per weight would be ~8x slower for
zero new information (and the FTS query is the slow part).

Reads as: dense_weight=1.0 is dense-only (sanity check: must reproduce recall@5
0.44), 0.0 is lexical-only (the standalone keyword signal), in between is
weighted RRF. Retrieval-only -> no generation, no Anthropic; OpenAI embeds the
queries, the DB does dense + FTS.

Run:  python -m sec_rag.eval.ablation_fusion
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sec_rag.config import load_config
from sec_rag.eval.metrics import evidence_match_rank, hit_rate_at_k, mean_reciprocal_rank
from sec_rag.ingest.financebench import load_questions
from sec_rag.pipeline import QueryEngine
from sec_rag.retrieve.dense import dense_search
from sec_rag.retrieve.hybrid import _rrf_fuse
from sec_rag.retrieve.lexical import lexical_search

# Sampled densely near the dense end, where any improvement is most likely.
WEIGHTS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]


def main() -> None:
    cfg = load_config("configs/v1.yaml")
    ks = sorted(cfg.eval.recall_ks)
    fuse_k = max(ks)
    candidates = cfg.retrieval.candidates
    k_rrf = cfg.retrieval.k_rrf

    questions = load_questions(cfg.eval.dataset)

    # One retrieval pass: cache each question's dense + lexical candidate lists.
    cached: list[tuple] = []  # (dense_list, lexical_list, evidence_texts, category)
    engine = QueryEngine(cfg)
    try:
        for q in questions:
            qvec = engine.embedder.embed_one(q.question)
            dlist = dense_search(engine.conn, qvec, candidates)
            llist = lexical_search(engine.conn, q.question, candidates)
            cached.append((dlist, llist, q.evidence_texts, q.question_type or "uncategorized"))
    finally:
        engine.close()

    rows = []
    for w in WEIGHTS:
        ranks: list[int | None] = []
        by_cat: dict[str, list[int | None]] = defaultdict(list)
        for dlist, llist, evidence, cat in cached:
            fused = _rrf_fuse(dlist, llist, k_rrf=k_rrf, top_k=fuse_k, dense_weight=w)
            contents = [c.content for c in fused]
            rank = evidence_match_rank(contents, evidence, mode="fuzzy")
            ranks.append(rank)
            by_cat[cat].append(rank)
        label = "lexical-only" if w == 0.0 else "dense-only" if w == 1.0 else "weighted"
        rows.append({
            "dense_weight": w,
            "label": label,
            "recall_at_k": {f"recall@{k}": round(hit_rate_at_k(ranks, k), 4) for k in ks},
            "mrr": round(mean_reciprocal_rank(ranks), 4),
            "metrics_generated_recall@5": round(
                hit_rate_at_k(by_cat.get("metrics-generated", []), 5), 4
            ),
            "per_category_recall@5": {
                cat: round(hit_rate_at_k(rs, 5), 4) for cat, rs in sorted(by_cat.items())
            },
        })

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "ablation_fusion_sweep",
        "match_mode": "fuzzy",
        "n_questions": len(questions),
        "candidates": candidates,
        "k_rrf": k_rrf,
        "baseline_dense_recall_at_5": 0.44,  # committed V0 baseline, for reference
        "sweep": rows,
    }
    out_dir = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"ablation_fusion_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote {out_path}  (n={len(questions)}, fuzzy, retrieval-only)")
    print(f"{'dense_w':>8}{'recall@5':>10}{'recall@10':>11}{'mrr':>8}{'tables@5':>10}  note")
    for r in rows:
        note = r["label"] if r["label"] != "weighted" else ""
        print(
            f"{r['dense_weight']:>8}{r['recall_at_k']['recall@5']:>10}"
            f"{r['recall_at_k']['recall@10']:>11}{r['mrr']:>8}"
            f"{r['metrics_generated_recall@5']:>10}  {note}"
        )


if __name__ == "__main__":
    main()
