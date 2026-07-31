"""Cross the two confounded variables: embedding model x chunk size x matcher.

The V0->V2 headline moved TWO variables at once (3-small->3-large AND 512->1024), so
the +0.20 recall@5 gain has never been decomposed, and the 3-small@1024 cell was never
run. This runs the full grid:

    {3-small, 3-large} x {512, 1024, 2048} x {strict, overlap, semantic}

by local exact-cosine retrieval (RETRIEVAL-ONLY — no generation, cheap in Anthropic
terms). recall@5/@10 + MRR per cell per matcher, plus a decomposition of the V0->V2
gain into embedding effect, chunk-size effect, and the interaction. Accuracy (the
chunk-invariant tiebreak) is measured separately by confound_accuracy.py.

Corpus embeddings are disk-cached (`/tmp/sec_rag_grid_cache`) so a TPM-throttled run is
resumable — re-run to continue from where embedding left off. 3-large/1024 reuses the
live Neon v2 embeddings (no re-embed).

  python -m sec_rag.eval.confound_grid
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sec_rag.config import EmbeddingConfig, Secrets, load_config
from sec_rag.db.pool import connect
from sec_rag.eval.matcher_study import KS, score_arm
from sec_rag.eval.metrics import load_matchers
from sec_rag.ingest.chunk import chunk_document, tiktoken_encoder
from sec_rag.ingest.embed import Embedder
from sec_rag.ingest.financebench import load_questions, locate_pdf
from sec_rag.ingest.parse import extract_pages

CHUNK_SIZES = [512, 1024, 2048]
MODELS = {"3-small": "text-embedding-3-small", "3-large": "text-embedding-3-large"}
CACHE = Path("/tmp/sec_rag_grid_cache")
TOP = max(KS)


def _normalize(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


def _corpus_items(questions, enc, max_tokens: int, strategy: str) -> list[list]:
    """Re-chunk the referenced FinanceBench docs -> [[content, doc_name, section], ...]."""
    items = []
    for doc_name in sorted({q.doc_name for q in questions}):
        pdf = locate_pdf(doc_name, "data/")
        if pdf is None:
            continue
        for c in chunk_document(extract_pages(pdf), enc, max_tokens=max_tokens,
                                overlap_tokens=64, strategy=strategy):
            items.append([c.content, doc_name, c.section])
    return items


def _neon_items() -> tuple[list[list], np.ndarray]:
    """Live v2 corpus (3-large@1536, 1024-tok): [[content, doc_name, section], ...] + vecs."""
    items, vecs = [], []
    with connect(Secrets()) as conn, conn.cursor() as cur:
        cur.execute("SELECT c.content, d.doc_name, c.section, c.embedding FROM chunks c "
                    "JOIN documents d ON d.id = c.doc_id WHERE c.embedding IS NOT NULL")
        for content, doc_name, section, emb in cur:
            items.append([content, doc_name, section])
            vecs.append(np.asarray(emb, dtype=np.float32))
    return items, _normalize(np.vstack(vecs))


def cell_corpus(model_name: str, model_id: str, chunk: int, questions, enc, strategy,
                secrets) -> tuple[list[list], np.ndarray]:
    """Load (items, normalized vecs) for one grid cell, from Neon / disk cache / fresh embed."""
    if model_name == "3-large" and chunk == 1024:
        return _neon_items()  # = production, no re-embed
    CACHE.mkdir(parents=True, exist_ok=True)
    cf, vf = CACHE / f"{model_name}_{chunk}.items.json", CACHE / f"{model_name}_{chunk}.vecs.npy"
    if cf.exists() and vf.exists():
        return json.loads(cf.read_text()), np.load(vf)
    items = _corpus_items(questions, enc, chunk, strategy)
    # 3-large has a tighter TPM ceiling than 3-small; pace its (large) corpus embed so
    # it stays under the limit instead of thrashing on 429-backoff. Query embeds and
    # 3-small are unthrottled.
    throttle = 6.0 if model_name == "3-large" else 0.0
    embedder = Embedder(EmbeddingConfig(provider="openai", model=model_id, dim=1536,
                                        batch_size=128), secrets, throttle_s=throttle)
    vecs = _normalize(np.asarray(embedder.embed([it[0] for it in items]), dtype=np.float32))
    cf.write_text(json.dumps(items))
    np.save(vf, vecs)
    return items, vecs


def _retrieve(items: list[list], V: np.ndarray, Q: np.ndarray, top: int) -> list[list[str]]:
    sims = V @ Q.T
    out = []
    for j in range(Q.shape[0]):
        col = sims[:, j]
        idx = np.argpartition(-col, top)[:top]
        idx = idx[np.argsort(-col[idx])]
        out.append([items[i][0] for i in idx])
    return out


def _r5(grid: dict, cell: str, matcher: str) -> float:
    return grid[cell][matcher]["recall_at_k"]["recall@5"]


def run() -> dict:
    cfg = load_config("configs/v0.yaml")
    secrets = Secrets()
    questions = load_questions(cfg.eval.dataset)
    enc = tiktoken_encoder(cfg.chunking.encoder)
    matchers = load_matchers("configs/matchers.yaml", secrets)

    grid: dict[str, dict] = {}
    n_chunks: dict[str, int] = {}
    for model_name, model_id in MODELS.items():
        q_embedder = Embedder(EmbeddingConfig(provider="openai", model=model_id, dim=1536,
                                              batch_size=128), secrets)
        Q = _normalize(np.asarray([q_embedder.embed_one(q.question) for q in questions],
                                  dtype=np.float32))
        for chunk in CHUNK_SIZES:
            cell = f"{model_name}/{chunk}"
            items, V = cell_corpus(model_name, model_id, chunk, questions, enc,
                                   cfg.chunking.strategy, secrets)
            n_chunks[cell] = len(items)
            retrieved = _retrieve(items, V, Q, TOP)
            grid[cell] = score_arm(retrieved, questions, matchers)
            del V

    # Decompose the V0 (3-small/512) -> V2 (3-large/1024) recall@5 gain, per matcher.
    decomp = {}
    for m in matchers:
        v0, v2 = _r5(grid, "3-small/512", m), _r5(grid, "3-large/1024", m)
        emb_at512 = _r5(grid, "3-large/512", m) - _r5(grid, "3-small/512", m)
        emb_at1024 = _r5(grid, "3-large/1024", m) - _r5(grid, "3-small/1024", m)
        chunk_small = _r5(grid, "3-small/1024", m) - _r5(grid, "3-small/512", m)
        chunk_large = _r5(grid, "3-large/1024", m) - _r5(grid, "3-large/512", m)
        decomp[m] = {
            "v0_recall@5": round(v0, 4),
            "v2_recall@5": round(v2, 4),
            "v0_to_v2_gain": round(v2 - v0, 4),
            "embedding_effect_@512": round(emb_at512, 4),
            "embedding_effect_@1024": round(emb_at1024, 4),
            "chunk_effect_3small_512to1024": round(chunk_small, 4),
            "chunk_effect_3large_512to1024": round(chunk_large, 4),
            # interaction: does the model change buy more at 1024 than at 512?
            "interaction": round(emb_at1024 - emb_at512, 4),
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "confound_grid",
        "index": "local exact cosine; 3-large/1024 = live Neon v2 embeddings",
        "models": list(MODELS),
        "chunk_sizes": CHUNK_SIZES,
        "matchers": list(matchers),
        "top_k_retrieved": TOP,
        "n_questions": len(questions),
        "n_chunks_per_cell": n_chunks,
        "grid": grid,
        "v0_to_v2_decomposition_recall@5": decomp,
        "note": "3-small@1024 was the never-run cell (fills the confound). Accuracy "
        "(chunk-invariant tiebreak) is in the confound_accuracy artifact.",
    }


def main() -> None:
    report = run()
    out = Path("eval_results")
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = out / f"confound_grid_{stamp}.json"
    p.write_text(json.dumps(report, indent=2))
    print(f"Wrote {p}\n")
    ms = report["matchers"]
    print(f"recall@5 grid  {'  '.join(f'{m:>9}' for m in ms)}")
    for model in MODELS:
        for chunk in CHUNK_SIZES:
            cell = f"{model}/{chunk}"
            vals = "  ".join(f"{report['grid'][cell][m]['recall_at_k']['recall@5']:>9}" for m in ms)
            print(f"  {cell:14}{vals}")
    print("\nV0->V2 recall@5 decomposition (per matcher):")
    for m, d in report["v0_to_v2_decomposition_recall@5"].items():
        print(f"  {m:9} gain {d['v0_to_v2_gain']:+.3f} = emb@512 {d['embedding_effect_@512']:+.3f} "
              f"+ chunk(3-large) {d['chunk_effect_3large_512to1024']:+.3f} + ...")


if __name__ == "__main__":
    main()
