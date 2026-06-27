"""Cross-encoder reranking (V1.2).

Dense retrieval is a *bi-encoder*: it embeds the query and each chunk separately,
so it ranks by generic semantic similarity. The V1.1b diagnostics showed that
leaves table evidence stuck at ranks 6-20 even when it IS in the candidate set —
"FY2022 net sales" embeds like every chunk about sales. A *cross-encoder* reads
(query, chunk) TOGETHER and scores their joint relevance, so it can promote that
evidence into the top-k. We rerank the dense top-N candidates and return top-k.

Model: BAAI/bge-reranker-base (open, self-hosted, free; CPU is fine at eval
scale). Loaded once and cached — the load is the cost, not the scoring.

USE_TF=0 is set at import (before sentence_transformers loads): this env has
Keras 3 / TensorFlow, which transformers' TF backend cannot import. We only need
PyTorch, so disabling the TF backend keeps the import clean.

Reproduce (reranker is an optional ablation, not a core dependency):
    pip install "torch>=2.2,<2.4" "transformers==4.44.2" "sentence-transformers==3.0.1"
Pins matter: later transformers require torch>=2.4. To formalize, add these as a
[project.optional-dependencies] `rerank` extra in pyproject.toml.

Result: BGE-base rerank-over-dense did NOT beat the dense baseline on this corpus
(recall@5 0.44 -> 0.393). Kept as an off-by-default knob + documented negative
result; see docs/depth-round.md and eval_results/financebench_20260627T012428Z.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")  # torch-only; must precede the transformers import

from dataclasses import replace
from functools import lru_cache

from sec_rag.retrieve.dense import RetrievedChunk

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=2)
def _load_reranker(model_name: str):
    """Load + cache the cross-encoder. Imported lazily so the rest of the package
    (and the pure-logic tests) need no torch/sentence-transformers."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _apply_scores(
    chunks: list[RetrievedChunk], scores: list[float], top_k: int
) -> list[RetrievedChunk]:
    """Reorder chunks by score (desc), truncate to top_k, attach rerank_score.

    Pure logic, unit-tested without the model. retrieval_score is preserved;
    rerank_score is the cross-encoder score carried through to the citation.
    """
    order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
    return [replace(chunks[i], rerank_score=float(scores[i])) for i in order[:top_k]]


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    *,
    model_name: str = DEFAULT_RERANK_MODEL,
) -> list[RetrievedChunk]:
    """Rerank candidate chunks with the cross-encoder; return the top_k."""
    if not chunks:
        return []
    model = _load_reranker(model_name)
    scores = model.predict([(query, c.content) for c in chunks])
    return _apply_scores(chunks, list(scores), top_k)
