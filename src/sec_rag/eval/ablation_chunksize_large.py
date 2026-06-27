"""Larger chunks over 3-large: does 768/1024 beat 512 (recall@5 0.57)?

The 256 test showed smaller chunks hurt; this checks the other direction, on top
of the winning 3-large embeddings. Local exact index. 512 arm loads the cached
3-large embeddings; 768/1024 re-chunk + embed locally.

HONEST CAVEAT: recall@k here is fuzzy >=0.5 overlap vs large gold spans, so bigger
chunks inflate it (they hold more of the span) -- part real retrieval, part metric
artifact. Bigger chunks also hurt citation granularity and raise generation cost,
so a recall rise here is NOT a clean win. Interpret accordingly.

  python -m sec_rag.eval.ablation_chunksize_large
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sec_rag.config import EmbeddingConfig, Secrets, load_config
from sec_rag.eval.ablation_chunksize_local import _eval_arm, _normalize
from sec_rag.eval.metrics import hit_rate_at_k, mean_reciprocal_rank
from sec_rag.ingest.chunk import chunk_document, tiktoken_encoder
from sec_rag.ingest.embed import Embedder
from sec_rag.ingest.financebench import load_questions, locate_pdf
from sec_rag.ingest.parse import extract_pages

LARGE_MODEL = "text-embedding-3-large"
LARGE_DIM = 3072
CACHE = Path("/tmp/sec_rag_large_emb.npz")
SIZES = [768, 1024]
OVERLAP = 64


def _chunk_texts(cfg, enc, size):
    out = []
    for dn in sorted({q.doc_name for q in load_questions(cfg.eval.dataset)}):
        pdf = locate_pdf(dn, "data/")
        if pdf is None:
            continue
        out.extend(
            c.content for c in chunk_document(
                extract_pages(pdf), enc, max_tokens=size, overlap_tokens=OVERLAP,
                strategy=cfg.chunking.strategy,
            )
        )
    return out


def _summary(ranks, by_cat, ks):
    return {
        "recall_at_k": {f"recall@{k}": round(hit_rate_at_k(ranks, k), 4) for k in ks},
        "mrr": round(mean_reciprocal_rank(ranks), 4),
        "per_category_recall@5": {c: round(hit_rate_at_k(rs, 5), 4) for c, rs in sorted(by_cat.items())},
    }


def main():
    cfg = load_config("configs/v0.yaml")
    secrets = Secrets()
    enc = tiktoken_encoder(cfg.chunking.encoder)
    ks = sorted(cfg.eval.recall_ks)
    questions = load_questions(cfg.eval.dataset)
    large = Embedder(
        EmbeddingConfig(provider="openai", model=LARGE_MODEL, dim=LARGE_DIM, batch_size=128), secrets
    )
    Q = _normalize(np.asarray([large.embed_one(q.question) for q in questions], dtype=np.float32))

    arms, nchunks = {}, {}
    if CACHE.exists():  # 512 arm from the cached 3-large embeddings
        d = np.load(CACHE, allow_pickle=True)
        c512, V = list(d["contents"]), d["vecs"]
        arms["512"] = _summary(*_eval_arm(c512, V, Q, questions, ks), ks)
        nchunks["512"] = len(c512)
        del V
    for size in SIZES:
        contents = _chunk_texts(cfg, enc, size)
        V = _normalize(np.asarray(large.embed(contents), dtype=np.float32))
        arms[str(size)] = _summary(*_eval_arm(contents, V, Q, questions, ks), ks)
        nchunks[str(size)] = len(contents)
        del V

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "ablation_chunksize_large",
        "match_mode": "fuzzy",
        "index": "local exact cosine",
        "embedding_model": LARGE_MODEL,
        "overlap_tokens": OVERLAP,
        "n_questions": len(questions),
        "n_chunks": nchunks,
        "caveat": "recall@k is partly inflated by larger chunks (fuzzy overlap vs large gold spans)",
        "arms": arms,
    }
    out = Path("eval_results")
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = out / f"ablation_chunksize_large_{stamp}.json"
    p.write_text(json.dumps(report, indent=2))

    cols = list(arms.keys())
    print(f"Wrote {p}\n(all arms: 3-large, exact cosine)\n")
    header = f"{'metric':24}" + "".join(f"{c:>10}" for c in cols)
    print(header)
    print(f"{'n_chunks':24}" + "".join(f"{nchunks[c]:>10}" for c in cols))
    for k in ks:
        kk = f"recall@{k}"
        print(f"{kk:24}" + "".join(f"{arms[c]['recall_at_k'][kk]:>10}" for c in cols))
    print(f"{'mrr':24}" + "".join(f"{arms[c]['mrr']:>10}" for c in cols))
    for cat in sorted(arms[cols[0]]["per_category_recall@5"]):
        print(f"{'  ' + cat + ' @5':24}" + "".join(
            f"{arms[c]['per_category_recall@5'].get(cat, '-'):>10}" for c in cols))


if __name__ == "__main__":
    main()
