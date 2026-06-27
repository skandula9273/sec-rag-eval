"""Reranker over 3-large candidates: does the cross-encoder help now?

The BGE reranker HURT over 3-small candidates (it demoted good dense hits). But
3-large retrieves much better candidates, so the picture may flip. This A/Bs
3-large dense vs 3-large + reranker, both on a local exact index (retrieval-only).

The 3-large corpus embeddings are cached to disk after the first run (~$1.70,
~20 min) so re-runs are free.

  python -m sec_rag.eval.ablation_rerank_over_large
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sec_rag.config import EmbeddingConfig, Secrets, load_config
from sec_rag.db.pool import connect
from sec_rag.eval.ablation_chunksize_local import _normalize
from sec_rag.eval.metrics import evidence_match_rank, hit_rate_at_k, mean_reciprocal_rank
from sec_rag.ingest.embed import Embedder
from sec_rag.ingest.financebench import load_questions
from sec_rag.retrieve.rerank import DEFAULT_RERANK_MODEL, _load_reranker

LARGE_MODEL = "text-embedding-3-large"
LARGE_DIM = 3072
CANDIDATES = 50
CACHE = Path("/tmp/sec_rag_large_emb.npz")


def _corpus(secrets, embedder):
    with connect(secrets) as conn, conn.cursor() as cur:
        cur.execute("SELECT content FROM chunks WHERE embedding IS NOT NULL ORDER BY id")
        contents = [r[0] for r in cur.fetchall()]
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        if len(d["contents"]) == len(contents):
            return list(d["contents"]), d["vecs"]
    vecs = _normalize(np.asarray(embedder.embed(contents), dtype=np.float32))
    np.savez(CACHE, contents=np.array(contents, dtype=object), vecs=vecs)
    return contents, vecs


def _summary(ranks, by_cat, ks):
    return {
        "recall_at_k": {f"recall@{k}": round(hit_rate_at_k(ranks, k), 4) for k in ks},
        "mrr": round(mean_reciprocal_rank(ranks), 4),
        "per_category_recall@5": {c: round(hit_rate_at_k(rs, 5), 4) for c, rs in sorted(by_cat.items())},
    }


def main():
    cfg = load_config("configs/v0.yaml")
    secrets = Secrets()
    ks = sorted(cfg.eval.recall_ks)
    questions = load_questions(cfg.eval.dataset)
    large = Embedder(
        EmbeddingConfig(provider="openai", model=LARGE_MODEL, dim=LARGE_DIM, batch_size=128), secrets
    )
    contents, V = _corpus(secrets, large)
    Q = _normalize(np.asarray([large.embed_one(q.question) for q in questions], dtype=np.float32))
    reranker = _load_reranker(DEFAULT_RERANK_MODEL)

    sims = V @ Q.T
    d_ranks, r_ranks = [], []
    d_cat, r_cat = defaultdict(list), defaultdict(list)
    for j, q in enumerate(questions):
        col = sims[:, j]
        cand = np.argpartition(-col, CANDIDATES)[:CANDIDATES]
        cand = cand[np.argsort(-col[cand])]  # dense order
        cand_contents = [contents[i] for i in cand]

        rd = evidence_match_rank(cand_contents, q.evidence_texts, mode="fuzzy")
        d_ranks.append(rd)
        d_cat[q.question_type or "uncategorized"].append(rd)

        scores = reranker.predict([(q.question, c) for c in cand_contents])
        order = np.argsort(-np.asarray(scores))
        rr_contents = [cand_contents[i] for i in order]
        rr = evidence_match_rank(rr_contents, q.evidence_texts, mode="fuzzy")
        r_ranks.append(rr)
        r_cat[q.question_type or "uncategorized"].append(rr)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "ablation_rerank_over_large",
        "match_mode": "fuzzy",
        "index": "local exact cosine",
        "embedding_model": LARGE_MODEL,
        "candidates": CANDIDATES,
        "rerank_model": DEFAULT_RERANK_MODEL,
        "n_questions": len(questions),
        "n_chunks": len(contents),
        "arms": {
            "3-large dense": _summary(d_ranks, d_cat, ks),
            "3-large + rerank": _summary(r_ranks, r_cat, ks),
        },
    }
    out = Path("eval_results")
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = out / f"ablation_rerank_over_large_{stamp}.json"
    p.write_text(json.dumps(report, indent=2))

    a_d, a_r = report["arms"]["3-large dense"], report["arms"]["3-large + rerank"]
    print(f"Wrote {p}\n")
    print(f"{'metric':26}{'3-large dense':>15}{'+ rerank':>12}")
    for k in ks:
        kk = f"recall@{k}"
        print(f"{kk:26}{a_d['recall_at_k'][kk]:>15}{a_r['recall_at_k'][kk]:>12}")
    print(f"{'mrr':26}{a_d['mrr']:>15}{a_r['mrr']:>12}")
    for c in sorted(a_d["per_category_recall@5"]):
        print(f"{'  ' + c + ' @5':26}{a_d['per_category_recall@5'][c]:>15}{a_r['per_category_recall@5'].get(c, '-'):>12}")


if __name__ == "__main__":
    main()
