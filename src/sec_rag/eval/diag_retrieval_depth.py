"""Retrieval-depth diagnostic: where does the table evidence rank in dense top-100?

Result (2026-06-26, 50 metrics-generated questions): rank 1-5 = 32%, 6-20 = +26%,
21-100 = +10%, miss>100 = 32%. The table evidence survives parsing (see
diag_table_parse), so the tables gap is a RANKING problem (26% at rank 6-20, a
reranker can promote) plus a RECALL problem (32% absent from top-100, an
embedding/chunking lever). Drove the 2026-06-26 corrective amendment
(docs/design-doc.md): reranker next, embedding ablation for the deep-miss band.

Retrieval-only: OpenAI embeds + the DB, no Anthropic. Run from the repo root:
  python -m sec_rag.eval.diag_retrieval_depth
"""

from collections import Counter

from sec_rag.config import load_config
from sec_rag.eval.metrics import evidence_match_rank
from sec_rag.ingest.financebench import load_questions
from sec_rag.pipeline import QueryEngine
from sec_rag.retrieve.dense import dense_search

DEEP = 100


def bucket(r):
    if r is None:
        return "miss(>100)"
    if r <= 5:
        return "1-5 (hit)"
    if r <= 10:
        return "6-10"
    if r <= 20:
        return "11-20"
    if r <= 50:
        return "21-50"
    return "51-100"


def main():
    cfg = load_config("configs/v0.yaml")
    qs = [q for q in load_questions()
          if (q.question_type or "") == "metrics-generated" and q.evidence_texts]

    counts = Counter()
    ranks = []
    engine = QueryEngine(cfg)
    try:
        for q in qs:
            qvec = engine.embedder.embed_one(q.question)
            chunks = dense_search(engine.conn, qvec, DEEP)
            r = evidence_match_rank([c.content for c in chunks], q.evidence_texts, mode="fuzzy")
            counts[bucket(r)] += 1
            ranks.append(r)
    finally:
        engine.close()

    n = len(qs)
    print(f"metrics-generated questions: {n}  (dense top-{DEEP}, fuzzy)\n")
    cum = 0
    for b in ["1-5 (hit)", "6-10", "11-20", "21-50", "51-100", "miss(>100)"]:
        c = counts.get(b, 0)
        cum += c
        print(f"  {b:12} {c:3}  ({c/n:5.1%})   cumulative recall: {cum/n:5.1%}")

    in_top20 = sum(1 for r in ranks if r is not None and r <= 20)
    promotable = sum(1 for r in ranks if r is not None and 6 <= r <= 20)
    print(f"\n  evidence in dense top-20 (reranker reach): {in_top20}/{n} ({in_top20/n:.1%})")
    print(f"  rank 6-20 (reranker could promote into top-5): {promotable}/{n} ({promotable/n:.1%})")
    print(f"  not in top-100 (embedding/recall problem): {counts.get('miss(>100)', 0)}/{n}")


if __name__ == "__main__":
    main()
