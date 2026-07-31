"""The two headline ablations, re-scored under all three matchers.

Which conclusions survive matcher choice? Rebuilds three arms via local exact-cosine
retrieval (the same method the original ablations used) and scores each under strict
/ overlap / semantic:

  embedding lever : 3-small/512  ->  3-large/512   (chunk size fixed)
  chunk-size lever: 3-large/512  ->  3-large/1024  (model fixed)

The suspicion this tests: the chunk-size win may be partly a property of the OVERLAP
metric (a bigger chunk clears the 50%-token bar more easily), so it could shrink or
vanish under strict / semantic. If so, that is the finding — stated in the artifact.

Heavy: re-embeds the 512-token corpus with both models (~$2 OpenAI). The 3-large/1024
arm reuses the live v2 embeddings from Neon (no re-embed).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from sec_rag.config import EmbeddingConfig, Secrets, load_config
from sec_rag.db.pool import connect
from sec_rag.eval.matcher_study import KS, score_arm
from sec_rag.eval.metrics import load_matchers
from sec_rag.ingest.chunk import chunk_document, tiktoken_encoder
from sec_rag.ingest.embed import Embedder
from sec_rag.ingest.financebench import load_questions, locate_pdf
from sec_rag.ingest.parse import extract_pages

_SURVIVE_EPS = 0.02  # a recall@5 delta below this is treated as "not a real move"


def _normalize(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


def _corpus_contents(questions, enc, max_tokens: int, strategy: str) -> list[str]:
    """Re-chunk the FinanceBench docs the questions reference, at ``max_tokens``."""
    contents: list[str] = []
    for doc_name in sorted({q.doc_name for q in questions}):
        pdf = locate_pdf(doc_name, "data/")
        if pdf is None:
            continue
        chunks = chunk_document(
            extract_pages(pdf), enc, max_tokens=max_tokens, overlap_tokens=64, strategy=strategy
        )
        contents.extend(c.content for c in chunks)
    return contents


def _neon_corpus() -> tuple[list[str], np.ndarray]:
    """The live v2 corpus (3-large@1536, 1024-token chunks) — contents + embeddings."""
    contents, vecs = [], []
    with connect(Secrets()) as conn, conn.cursor() as cur:
        cur.execute("SELECT content, embedding FROM chunks WHERE embedding IS NOT NULL")
        for content, emb in cur:
            contents.append(content)
            vecs.append(np.asarray(emb, dtype=np.float32))
    return contents, _normalize(np.vstack(vecs))


def _retrieve(contents: list[str], V: np.ndarray, Q: np.ndarray, top: int) -> list[list[str]]:
    """Top-``top`` chunk contents per query by exact cosine (V, Q L2-normalized)."""
    sims = V @ Q.T  # (n_chunks, n_questions)
    out = []
    for j in range(Q.shape[0]):
        col = sims[:, j]
        idx = np.argpartition(-col, top)[:top]
        idx = idx[np.argsort(-col[idx])]
        out.append([contents[i] for i in idx])
    return out


def run_ablations(matchers_path: str) -> dict:
    cfg = load_config("configs/v0.yaml")  # 3-small / 512 baseline knobs
    secrets = Secrets()
    questions = load_questions(cfg.eval.dataset)
    enc = tiktoken_encoder(cfg.chunking.encoder)
    matchers = load_matchers(matchers_path, secrets)
    top = max(KS)

    def _embedder(model: str) -> Embedder:
        spec = EmbeddingConfig(provider="openai", model=model, dim=1536, batch_size=128)
        return Embedder(spec, secrets)

    small = _embedder("text-embedding-3-small")
    large = _embedder("text-embedding-3-large")

    def _embed_queries(embedder) -> np.ndarray:
        vecs = [embedder.embed_one(q.question) for q in questions]
        return _normalize(np.asarray(vecs, dtype=np.float32))

    Q_small = _embed_queries(small)
    Q_large = _embed_queries(large)

    # 512 corpus, embedded with each model (same chunk texts -> isolates the model).
    c512 = _corpus_contents(questions, enc, 512, cfg.chunking.strategy)
    Vs = _normalize(np.asarray(small.embed(c512), dtype=np.float32))
    arm_small512 = _retrieve(c512, Vs, Q_small, top)
    del Vs
    Vl = _normalize(np.asarray(large.embed(c512), dtype=np.float32))
    arm_large512 = _retrieve(c512, Vl, Q_large, top)
    del Vl
    # 1024 corpus = live v2 embeddings from Neon (3-large@1536), same query embeddings.
    c1024, Vl1024 = _neon_corpus()
    arm_large1024 = _retrieve(c1024, Vl1024, Q_large, top)
    del Vl1024

    arms = {
        "3-small/512": arm_small512,
        "3-large/512": arm_large512,
        "3-large/1024": arm_large1024,
    }
    n_chunks = {"512": len(c512), "1024": len(c1024)}
    scored = {name: score_arm(ret, questions, matchers) for name, ret in arms.items()}

    def r5(arm: str, matcher: str) -> float:
        return scored[arm][matcher]["recall_at_k"]["recall@5"]

    levers = {}
    for matcher in matchers:
        emb = round(r5("3-large/512", matcher) - r5("3-small/512", matcher), 4)
        chunk = round(r5("3-large/1024", matcher) - r5("3-large/512", matcher), 4)
        levers[matcher] = {
            "embedding_lever_recall@5_delta": emb,   # 3-small/512 -> 3-large/512
            "chunk_lever_recall@5_delta": chunk,     # 3-large/512 -> 3-large/1024
            "embedding_survives": emb >= _SURVIVE_EPS,
            "chunk_survives": chunk >= _SURVIVE_EPS,
        }

    def _deltas(key: str) -> str:
        return ", ".join(f"{m} {levers[m][key]:+.3f}" for m in matchers)

    emb_all = all(levers[m]["embedding_survives"] for m in matchers)
    chunk_matchers = [m for m in matchers if levers[m]["chunk_survives"]]
    if chunk_matchers == ["overlap"]:
        verdict = ("The chunk-size win is OVERLAP-ONLY -> largely a metric artifact (a "
                   "bigger chunk clears the 50%-token bar more easily), not a real "
                   "retrieval gain.")
    elif len(chunk_matchers) > 1:
        verdict = ("The chunk-size win holds under more than the lenient matcher -> "
                   "real, not just metric.")
    elif not chunk_matchers:
        verdict = "The chunk-size win survives NO matcher (>= +0.02 recall@5) -> not a real gain."
    else:
        verdict = f"The chunk-size win survives only under: {chunk_matchers}."
    summary = (
        f"Embedding lever (3-small->3-large @512), recall@5 delta by matcher: "
        f"{_deltas('embedding_lever_recall@5_delta')}; survives all = {emb_all}. || "
        f"Chunk lever (512->1024 @3-large), recall@5 delta by matcher: "
        f"{_deltas('chunk_lever_recall@5_delta')}; survives under "
        f"{chunk_matchers or 'no'} matcher(s). {verdict}"
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "matcher_study_ablations",
        "index": "local exact cosine; 3-large/1024 arm = live Neon v2 embeddings",
        "n_questions": len(questions),
        "n_chunks": n_chunks,
        "top_k_retrieved": top,
        "matcher_specs": (yaml.safe_load(Path(matchers_path).read_text()) or {}).get(
            "matchers", {}
        ),
        "arms_scored": scored,
        "levers_recall@5": levers,
        "summary": summary,
    }
