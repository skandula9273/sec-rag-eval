"""Generation-latency A/B: verbose vs concise answer prompt.

The live-API p95 (~6.2 s) is generation-bound — a long answer is the tail, and
generation time scales with output tokens. Retrieval is already fast and pooled,
so the remaining lever is how much the model *writes*. This measures the concision
change to ``generate/answer._SYSTEM`` the only honest way: same questions, same
retrieved chunks, generate the answer under BOTH prompts back-to-back, and compare
output tokens + generation latency.

It also records the citation count and a refusal flag per answer, so we can confirm
the concise prompt did not drop citations or start guessing — the point is a
quality-neutral latency win, not a shorter-but-worse answer. (Faithfulness is
checked separately with eval/faithfulness_spotcheck.py on the concise prompt.)

Usage:
  python -m sec_rag.eval.bench_generation --n 20
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from sec_rag.config import load_config
from sec_rag.eval.measure_api_latency import QUESTIONS
from sec_rag.generate import answer as answer_mod
from sec_rag.generate.answer import generate_answer
from sec_rag.pipeline import QueryEngine

# The pre-change (verbose) system prompt, kept verbatim so this benchmark documents
# exactly what "before" was. "after" is the module's current answer_mod._SYSTEM.
_VERBOSE_SYSTEM = (
    "You answer questions about SEC filings using only the numbered sources "
    "provided. Cite every claim with the matching source number in square "
    "brackets, e.g. [1]. If the sources do not contain the answer, say so "
    "explicitly instead of guessing. Do not use outside knowledge."
)

_CITE_RE = re.compile(r"\[(\d+)\]")
_REFUSAL_RE = re.compile(r"\b(cannot|can't|could not|do not contain|don't contain)\b", re.I)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return float(s[idx])


def _stats(values: list[float]) -> dict:
    return {
        "p50": round(_percentile(values, 50), 1),
        "p95": round(_percentile(values, 95), 1),
        "max": round(max(values), 1) if values else 0.0,
        "mean": round(sum(values) / len(values), 1) if values else 0.0,
    }


def _gen(engine, question, chunks, system):
    t0 = time.perf_counter()
    ga = generate_answer(question, chunks, engine.cfg.generation, engine.secrets, system=system)
    ms = (time.perf_counter() - t0) * 1000
    return {
        "gen_ms": round(ms, 1),
        "tokens_out": ga.tokens_out,
        "cost_usd": ga.cost_usd,
        "n_citations": len(set(_CITE_RE.findall(ga.text))),
        "looks_like_refusal": bool(_REFUSAL_RE.search(ga.text)),
        "preview": ga.text[:160].replace("\n", " "),
    }


def run(config: str, n: int) -> dict:
    cfg = load_config(config)
    top_k = cfg.retrieval.top_k
    questions = QUESTIONS[:n]
    concise_system = answer_mod._SYSTEM  # the shipped (post-change) prompt

    engine = QueryEngine(cfg)
    rows = []
    try:
        for q in questions:
            chunks, _ = engine.retrieve(q, top_k=top_k)  # retrieve ONCE, share both arms
            verbose = _gen(engine, q, chunks, _VERBOSE_SYSTEM)
            concise = _gen(engine, q, chunks, concise_system)
            rows.append({"question": q, "verbose": verbose, "concise": concise})
    finally:
        engine.close()

    def arm(key, field):
        return [r[key][field] for r in rows]

    v_tok, c_tok = arm("verbose", "tokens_out"), arm("concise", "tokens_out")
    v_ms, c_ms = arm("verbose", "gen_ms"), arm("concise", "gen_ms")
    v_cost, c_cost = arm("verbose", "cost_usd"), arm("concise", "cost_usd")

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # Citations must survive the trim; count answers that lost all citations.
    lost_citations = sum(
        1 for r in rows if r["verbose"]["n_citations"] > 0 and r["concise"]["n_citations"] == 0
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "generation_prompt_ab",
        "config_file": config,
        "n": len(rows),
        "arms": {
            "verbose": {
                "tokens_out": _stats([float(x) for x in v_tok]),
                "gen_ms": _stats(v_ms),
                "mean_cost_usd": round(mean(v_cost), 6),
            },
            "concise": {
                "tokens_out": _stats([float(x) for x in c_tok]),
                "gen_ms": _stats(c_ms),
                "mean_cost_usd": round(mean(c_cost), 6),
            },
        },
        "delta": {
            "tokens_out_mean_pct": round(100 * (mean(c_tok) - mean(v_tok)) / mean(v_tok), 1),
            "gen_ms_p50_pct": round(
                100 * (_percentile(c_ms, 50) - _percentile(v_ms, 50)) / _percentile(v_ms, 50), 1
            ),
            "gen_ms_p95_pct": round(
                100 * (_percentile(c_ms, 95) - _percentile(v_ms, 95)) / _percentile(v_ms, 95), 1
            ),
            "cost_mean_pct": round(100 * (mean(c_cost) - mean(v_cost)) / mean(v_cost), 1),
        },
        "quality_guard": {
            "answers_that_lost_all_citations": lost_citations,
            "verbose_refusals": sum(r["verbose"]["looks_like_refusal"] for r in rows),
            "concise_refusals": sum(r["concise"]["looks_like_refusal"] for r in rows),
        },
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generation prompt A/B (verbose vs concise) -> JSON")
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args.config, args.n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"gen_bench_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))

    v, c, d = report["arms"]["verbose"], report["arms"]["concise"], report["delta"]
    print(f"Wrote {out_path}  (n={report['n']})")
    print(f"  tokens_out mean : {v['tokens_out']['mean']} -> {c['tokens_out']['mean']}  ({d['tokens_out_mean_pct']}%)")
    print(f"  gen_ms p50      : {v['gen_ms']['p50']} -> {c['gen_ms']['p50']}  ({d['gen_ms_p50_pct']}%)")
    print(f"  gen_ms p95      : {v['gen_ms']['p95']} -> {c['gen_ms']['p95']}  ({d['gen_ms_p95_pct']}%)")
    print(f"  cost mean       : ${v['mean_cost_usd']} -> ${c['mean_cost_usd']}  ({d['cost_mean_pct']}%)")
    print(f"  quality guard   : lost-all-citations={report['quality_guard']['answers_that_lost_all_citations']}  "
          f"refusals {report['quality_guard']['verbose_refusals']}->{report['quality_guard']['concise_refusals']}")


if __name__ == "__main__":
    main()
