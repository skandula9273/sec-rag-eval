"""Chunk-size ablation on a LOCAL in-memory index (bypasses Neon's 512 MB cap).

Neon's free tier is maxed by the current 512-token corpus (468/512 MB), so a
second corpus doesn't fit. But an offline ablation doesn't need the production
store: we hold the vectors in a numpy array and do exact cosine locally.

Both arms use exact cosine over the SAME model (text-embedding-3-small); only the
chunk size differs:
  - 512 arm: reuse the embeddings already in prod `chunks` (same model + text =
    re-embedding). Also a sanity check — should reproduce the committed 0.44 and
    reveal whether Neon's approximate HNSW was costing any recall vs exact search.
  - 256 arm: re-chunk at 256 + embed locally (~$0.30). No Neon writes.

  python -m sec_rag.eval.ablation_chunksize_local
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sec_rag.config import Secrets, load_config
from sec_rag.db.pool import connect
from sec_rag.eval.metrics import evidence_match_rank, hit_rate_at_k, mean_reciprocal_rank
from sec_rag.ingest.chunk import chunk_document, tiktoken_encoder
from sec_rag.ingest.embed import Embedder
from sec_rag.ingest.financebench import load_questions, locate_pdf
from sec_rag.ingest.parse import extract_pages


def _normalize(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


def _load_512_from_neon(secrets) -> tuple[list[str], np.ndarray]:
    contents, vecs = [], []
    with connect(secrets) as conn, conn.cursor() as cur:
        cur.execute("SELECT content, embedding FROM chunks WHERE embedding IS NOT NULL")
        for content, emb in cur:
            contents.append(content)
            vecs.append(np.asarray(emb, dtype=np.float32))
    return contents, _normalize(np.vstack(vecs))


def _build_256_local(cfg, embedder, enc) -> tuple[list[str], np.ndarray]:
    doc_names = sorted({q.doc_name for q in load_questions(cfg.eval.dataset)})
    contents: list[str] = []
    for doc_name in doc_names:
        pdf = locate_pdf(doc_name, "data/")
        if pdf is None:
            continue
        chunks = chunk_document(
            extract_pages(pdf), enc, max_tokens=256, overlap_tokens=64,
            strategy=cfg.chunking.strategy,
        )
        contents.extend(c.content for c in chunks)
    embs = embedder.embed(contents)  # batched by cfg.embedding.batch_size
    return contents, _normalize(np.asarray(embs, dtype=np.float32))


def _eval_arm(contents, V, Q, questions, ks):
    sims = V @ Q.T  # (n_chunks, n_questions)
    top = max(ks)
    ranks, by_cat = [], defaultdict(list)
    for j, q in enumerate(questions):
        col = sims[:, j]
        idx = np.argpartition(-col, top)[:top]
        idx = idx[np.argsort(-col[idx])]
        cont = [contents[i] for i in idx]
        r = evidence_match_rank(cont, q.evidence_texts, mode="fuzzy")
        ranks.append(r)
        by_cat[q.question_type or "uncategorized"].append(r)
    return ranks, by_cat


def main():
    cfg = load_config("configs/v0.yaml")
    secrets = Secrets()
    embedder = Embedder(cfg.embedding, secrets)
    enc = tiktoken_encoder(cfg.chunking.encoder)
    ks = sorted(cfg.eval.recall_ks)
    questions = load_questions(cfg.eval.dataset)
    Q = _normalize(np.asarray([embedder.embed_one(q.question) for q in questions], dtype=np.float32))

    results, n_chunks = {}, {}
    # 512 arm (reuse prod embeddings), then free before building 256 to cap memory.
    c512, V512 = _load_512_from_neon(secrets)
    n_chunks["512"] = len(c512)
    results["512"] = _eval_arm(c512, V512, Q, questions, ks)
    del c512, V512

    c256, V256 = _build_256_local(cfg, embedder, enc)
    n_chunks["256"] = len(c256)
    results["256"] = _eval_arm(c256, V256, Q, questions, ks)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "ablation_chunksize_local",
        "match_mode": "fuzzy",
        "index": "local exact cosine",
        "embedding_model": cfg.embedding.model,
        "n_questions": len(questions),
        "n_chunks": n_chunks,
        "committed_baseline_512_neon": {"recall@5": 0.44, "recall@10": 0.54, "tables@5": 0.32},
        "arms": {},
    }
    for size, (ranks, by_cat) in results.items():
        report["arms"][size] = {
            "recall_at_k": {f"recall@{k}": round(hit_rate_at_k(ranks, k), 4) for k in ks},
            "mrr": round(mean_reciprocal_rank(ranks), 4),
            "per_category_recall@5": {c: round(hit_rate_at_k(rs, 5), 4) for c, rs in sorted(by_cat.items())},
        }

    out = Path("eval_results")
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = out / f"ablation_chunksize_local_{stamp}.json"
    p.write_text(json.dumps(report, indent=2))

    a512, a256 = report["arms"]["512"], report["arms"]["256"]
    print(f"Wrote {p}\n")
    print(f"{'metric':26}{'512 (exact)':>13}{'256 (exact)':>13}")
    print(f"{'n_chunks':26}{n_chunks['512']:>13}{n_chunks['256']:>13}")
    for k in ks:
        kk = f"recall@{k}"
        print(f"{kk:26}{a512['recall_at_k'][kk]:>13}{a256['recall_at_k'][kk]:>13}")
    print(f"{'mrr':26}{a512['mrr']:>13}{a256['mrr']:>13}")
    for c in sorted(a512["per_category_recall@5"]):
        print(f"{'  ' + c + ' @5':26}"
              f"{a512['per_category_recall@5'][c]:>13}"
              f"{a256['per_category_recall@5'].get(c, '-'):>13}")
    print("\n(512-exact vs committed 0.44 Neon-HNSW = the index sanity check)")


if __name__ == "__main__":
    main()
