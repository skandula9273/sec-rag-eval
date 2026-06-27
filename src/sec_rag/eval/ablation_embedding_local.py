"""Embedding-model ablation on a local index: 3-small vs 3-large.

After hybrid / table-extraction / reranker / chunk-size all failed to beat dense,
the remaining representation lever is the embedding MODEL itself. text-embedding-
3-large (3072-dim) is the cheapest test (same provider, no new key). Runs on a
local numpy index (exact cosine) so Neon's 512 MB cap and the 3072-dim schema
mismatch are both irrelevant -- prod is untouched.

One variable: the embedding model. SAME 512-token chunk texts for both arms
(pulled from prod `chunks`); SAME queries; exact search. 3-small reuses the
embeddings already in Neon (= the committed 0.44 baseline); 3-large embeds the
same texts locally (~$1.70). Queries are embedded with each arm's own model.

  python -m sec_rag.eval.ablation_embedding_local
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sec_rag.config import EmbeddingConfig, Secrets, load_config
from sec_rag.db.pool import connect
from sec_rag.eval.metrics import hit_rate_at_k, mean_reciprocal_rank
from sec_rag.eval.ablation_chunksize_local import _eval_arm, _normalize
from sec_rag.ingest.embed import Embedder
from sec_rag.ingest.financebench import load_questions

LARGE_MODEL = "text-embedding-3-large"
LARGE_DIM = 3072


def _load_small_from_neon(secrets):
    """Pull the prod chunk texts + their 3-small embeddings (the committed arm)."""
    contents, vecs = [], []
    with connect(secrets) as conn, conn.cursor() as cur:
        cur.execute("SELECT content, embedding FROM chunks WHERE embedding IS NOT NULL")
        for content, emb in cur:
            contents.append(content)
            vecs.append(np.asarray(emb, dtype=np.float32))
    return contents, _normalize(np.vstack(vecs))


def main():
    cfg = load_config("configs/v0.yaml")  # 3-small baseline config
    secrets = Secrets()
    ks = sorted(cfg.eval.recall_ks)
    questions = load_questions(cfg.eval.dataset)

    small_embedder = Embedder(cfg.embedding, secrets)
    large_embedder = Embedder(
        EmbeddingConfig(provider="openai", model=LARGE_MODEL, dim=LARGE_DIM, batch_size=128),
        secrets,
    )

    # --- 3-small arm: reuse prod embeddings; queries embedded with 3-small ---
    contents, V_small = _load_small_from_neon(secrets)
    Q_small = _normalize(np.asarray([small_embedder.embed_one(q.question) for q in questions], dtype=np.float32))
    res_small = _eval_arm(contents, V_small, Q_small, questions, ks)
    del V_small, Q_small

    # --- 3-large arm: SAME chunk texts, embedded with 3-large; queries too ---
    V_large = _normalize(np.asarray(large_embedder.embed(contents), dtype=np.float32))
    Q_large = _normalize(np.asarray([large_embedder.embed_one(q.question) for q in questions], dtype=np.float32))
    res_large = _eval_arm(contents, V_large, Q_large, questions, ks)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "ablation_embedding_local",
        "match_mode": "fuzzy",
        "index": "local exact cosine",
        "n_questions": len(questions),
        "n_chunks": len(contents),
        "chunk_tokens": cfg.chunking.max_tokens,
        "committed_baseline_512_neon": {"recall@5": 0.44, "recall@10": 0.54, "tables@5": 0.32},
        "arms": {},
    }
    for name, (ranks, by_cat) in [("3-small", res_small), ("3-large", res_large)]:
        report["arms"][name] = {
            "model": cfg.embedding.model if name == "3-small" else LARGE_MODEL,
            "recall_at_k": {f"recall@{k}": round(hit_rate_at_k(ranks, k), 4) for k in ks},
            "mrr": round(mean_reciprocal_rank(ranks), 4),
            "per_category_recall@5": {c: round(hit_rate_at_k(rs, 5), 4) for c, rs in sorted(by_cat.items())},
        }

    out = Path("eval_results")
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = out / f"ablation_embedding_local_{stamp}.json"
    p.write_text(json.dumps(report, indent=2))

    a_s, a_l = report["arms"]["3-small"], report["arms"]["3-large"]
    print(f"Wrote {p}\n")
    print(f"{'metric':26}{'3-small':>11}{'3-large':>11}")
    for k in ks:
        kk = f"recall@{k}"
        print(f"{kk:26}{a_s['recall_at_k'][kk]:>11}{a_l['recall_at_k'][kk]:>11}")
    print(f"{'mrr':26}{a_s['mrr']:>11}{a_l['mrr']:>11}")
    for c in sorted(a_s["per_category_recall@5"]):
        print(f"{'  ' + c + ' @5':26}{a_s['per_category_recall@5'][c]:>11}{a_l['per_category_recall@5'].get(c, '-'):>11}")


if __name__ == "__main__":
    main()
