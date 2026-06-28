"""Embedding-dimension ablation: how low can 3-large go (Matryoshka truncation)?

3-large is 3072-dim = ~2x the storage of 3-small, which doesn't fit Neon's free
tier. But OpenAI 3-* embeddings are Matryoshka: truncating to the first N dims
and renormalizing == the natively N-dim output. If 3-large@1536 keeps the recall
gain, we keep the existing vector(1536) schema and productionize for FREE.

Near-free: derives truncated dims from the cached 3-large@3072 corpus vectors
(512-token chunks); embeds queries once at 3072 and truncates. No corpus re-embed.

  python -m sec_rag.eval.ablation_embed_dims
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sec_rag.config import EmbeddingConfig, Secrets, load_config
from sec_rag.eval.ablation_chunksize_local import _eval_arm, _normalize
from sec_rag.eval.metrics import hit_rate_at_k, mean_reciprocal_rank
from sec_rag.ingest.embed import Embedder
from sec_rag.ingest.financebench import load_questions

CACHE = Path("/tmp/sec_rag_large_emb.npz")
DIMS = [3072, 1536, 1024, 512, 256]


def _trunc(mat, d):
    # Truncating a unit-norm 3-* embedding to its first d dims + renormalizing is
    # exactly OpenAI's reduced-`dimensions` output (Matryoshka).
    return _normalize(mat[:, :d].astype(np.float32))


def main():
    if not CACHE.exists():
        raise SystemExit("3-large cache missing; run ablation_rerank_over_large first to build it")
    cfg = load_config("configs/v0.yaml")
    secrets = Secrets()
    ks = sorted(cfg.eval.recall_ks)
    questions = load_questions(cfg.eval.dataset)

    d = np.load(CACHE, allow_pickle=True)
    contents, V = list(d["contents"]), d["vecs"].astype(np.float32)
    large = Embedder(
        EmbeddingConfig(provider="openai", model="text-embedding-3-large", dim=3072, batch_size=128),
        secrets,
    )
    Qraw = np.asarray([large.embed_one(q.question) for q in questions], dtype=np.float32)

    arms = {}
    for dim in DIMS:
        ranks, by_cat = _eval_arm(contents, _trunc(V, dim), _trunc(Qraw, dim), questions, ks)
        arms[str(dim)] = {
            "recall_at_k": {f"recall@{k}": round(hit_rate_at_k(ranks, k), 4) for k in ks},
            "mrr": round(mean_reciprocal_rank(ranks), 4),
            "per_category_recall@5": {c: round(hit_rate_at_k(rs, 5), 4) for c, rs in sorted(by_cat.items())},
        }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "ablation_embed_dims",
        "match_mode": "fuzzy",
        "index": "local exact cosine",
        "embedding_model": "text-embedding-3-large (truncated dims)",
        "chunk_tokens": 512,
        "n_questions": len(questions),
        "n_chunks": len(contents),
        "note": "vector(1536) schema fits the Neon free tier; 3072 needs an upgrade",
        "arms": arms,
    }
    out = Path("eval_results")
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = out / f"ablation_embed_dims_{stamp}.json"
    p.write_text(json.dumps(report, indent=2))

    cols = [str(d) for d in DIMS]
    print(f"Wrote {p}\n(3-large truncated to N dims, 512-token chunks, exact cosine)\n")
    print(f"{'dim':24}" + "".join(f"{c:>9}" for c in cols))
    for k in ks:
        kk = f"recall@{k}"
        print(f"{kk:24}" + "".join(f"{arms[c]['recall_at_k'][kk]:>9}" for c in cols))
    print(f"{'mrr':24}" + "".join(f"{arms[c]['mrr']:>9}" for c in cols))
    for cat in sorted(arms[cols[0]]["per_category_recall@5"]):
        print(f"{'  ' + cat + ' @5':24}" + "".join(
            f"{arms[c]['per_category_recall@5'].get(cat, '-'):>9}" for c in cols))


if __name__ == "__main__":
    main()
