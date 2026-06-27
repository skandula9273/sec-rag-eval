"""Diagnostic: does pdfplumber (text + tables) recover table evidence pypdf loses?

Result (2026-06-26): NO. Gold table evidence is recoverable in a 512-tok window
under pypdf for all sampled questions; pdfplumber does not improve it (sometimes
worse, because flattening cells reshapes text away from FinanceBench's gold
spans). This ruled out table extraction as a lever *before* a full re-ingest.
See the 2026-06-26 corrective amendment in docs/design-doc.md.

For N metrics-generated (table) questions, chunk each source doc under BOTH
parsers into identical 512-tok windows, then use the eval's own fuzzy matcher to
ask: is the gold evidence matchable in SOME window? That's the recall *ceiling*
for that parser.

Diagnostic-only dependency: `pip install pdfplumber` (NOT a project dependency —
pdfplumber was evaluated here and not adopted). Run from the repo root with the
FinanceBench PDFs under data/:  python -m sec_rag.eval.diag_table_parse
"""

import pdfplumber

from sec_rag.eval.metrics import _normalize, evidence_match_rank
from sec_rag.ingest.chunk import chunk_tokens, tiktoken_encoder
from sec_rag.ingest.financebench import load_questions, locate_pdf
from sec_rag.ingest.parse import extract_pages

N = 8
enc = tiktoken_encoder()


def windows(text):
    return chunk_tokens(text, enc, 512, 64)


def pdfplumber_text(path):
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            rows = []
            for tb in (page.extract_tables() or []):
                for row in tb:
                    rows.append(" | ".join((c or "").replace("\n", " ").strip() for c in row))
            parts.append(t + ("\n" + "\n".join(rows) if rows else ""))
    return "\n".join(parts)


def best_overlap(wins, evidence):
    evs = [set(_normalize(e).split()) for e in evidence if e and e.strip()]
    best = 0.0
    for w in wins:
        wt = set(_normalize(w).split())
        for et in evs:
            if et:
                best = max(best, len(et & wt) / len(et))
    return best


def main():
    qs = [q for q in load_questions()
          if (q.question_type or "") == "metrics-generated" and q.evidence_texts]
    picked = []
    for q in qs:
        if locate_pdf(q.doc_name, "data/"):
            picked.append(q)
        if len(picked) >= N:
            break

    cache = {}
    print(f"{'doc':30}{'pypdf':>8}{'pdfplmb':>9} | {'ov_py':>6}{'ov_pp':>7}")
    rec_py = rec_pp = 0
    for q in picked:
        pdf = locate_pdf(q.doc_name, "data/")
        if q.doc_name not in cache:
            cache[q.doc_name] = (windows("\n".join(extract_pages(pdf))), windows(pdfplumber_text(pdf)))
        wp, wpp = cache[q.doc_name]
        r_py = evidence_match_rank(wp, q.evidence_texts, mode="fuzzy")
        r_pp = evidence_match_rank(wpp, q.evidence_texts, mode="fuzzy")
        rec_py += r_py is not None
        rec_pp += r_pp is not None
        print(f"{q.doc_name[:30]:30}{('hit#'+str(r_py)) if r_py else 'MISS':>8}"
              f"{('hit#'+str(r_pp)) if r_pp else 'MISS':>9} | "
              f"{best_overlap(wp, q.evidence_texts):6.2f}{best_overlap(wpp, q.evidence_texts):7.2f}")
    print("\nEvidence recoverable in SOME 512-tok window (the recall ceiling):")
    print(f"  pypdf:      {rec_py}/{len(picked)}")
    print(f"  pdfplumber: {rec_pp}/{len(picked)}")


if __name__ == "__main__":
    main()
