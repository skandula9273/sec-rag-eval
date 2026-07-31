"""Connection-pool concurrency benchmark.

The old QueryEngine held ONE shared DB connection, so N concurrent retrievals
serialized on it (psycopg locks a connection for a query's duration). The pool
(db/pool.new_pool) lets them run in parallel. This measures the difference the
only honest way — an A/B on the SAME code path, varying only the pool size:

  arm "pool=1" reproduces the old single-connection behaviour (serialized);
  arm "pool=8" is the new default (parallel).

It fires N concurrent DB searches (dense_search over a pre-computed query vector,
so we isolate DB time, not the OpenAI embedding call) and reports wall-clock,
throughput, and the per-request latency distribution. Single-request latency is
unchanged by pooling (retrieval is generation-independent and already fast) — the
win is concurrent throughput and tail latency, which is exactly the documented
"single shared connection serializes queries" debt.

Usage:
  python -m sec_rag.eval.bench_pool_concurrency --concurrency 24 --top-k 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from sec_rag.config import load_config
from sec_rag.pipeline import QueryEngine
from sec_rag.retrieve.dense import dense_search

_QUERY = "What was total revenue and net income for the fiscal year?"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return float(s[idx])


def _run_arm(cfg, qvec: list[float], pool_size: int, concurrency: int, top_k: int) -> dict:
    # Force min=max=pool_size so all connections are pre-warmed and the pool size
    # is the only variable (min=max avoids first-borrow connect latency skew).
    os.environ["SEC_RAG_POOL_MIN"] = str(pool_size)
    os.environ["SEC_RAG_POOL_MAX"] = str(pool_size)
    eng = QueryEngine(cfg)
    try:
        # Warm: one search per connection so the DB plan/caches are hot for all.
        for _ in range(pool_size):
            with eng.connection() as conn:
                dense_search(conn, qvec, top_k)

        latencies: list[float] = []

        def one() -> float:
            t0 = time.perf_counter()
            with eng.connection() as conn:
                dense_search(conn, qvec, top_k)
            return (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            latencies = list(ex.map(lambda _: one(), range(concurrency)))
        wall_ms = (time.perf_counter() - t0) * 1000
    finally:
        eng.close()

    return {
        "pool_size": pool_size,
        "concurrency": concurrency,
        "wall_ms": round(wall_ms, 1),
        "throughput_qps": round(concurrency / (wall_ms / 1000), 1),
        "per_request_ms": {
            "p50": round(_percentile(latencies, 50), 1),
            "p95": round(_percentile(latencies, 95), 1),
            "max": round(max(latencies), 1),
            "mean": round(sum(latencies) / len(latencies), 1),
        },
    }


def run(config: str, concurrency: int, top_k: int) -> dict:
    cfg = load_config(config)
    # Embed the shared query once (isolate DB latency from the OpenAI call).
    warm = QueryEngine(cfg)
    qvec = warm.embedder.embed_one(_QUERY)
    warm.close()

    serialized = _run_arm(cfg, qvec, 1, concurrency, top_k)
    pooled = _run_arm(cfg, qvec, 8, concurrency, top_k)

    speedup = round(serialized["wall_ms"] / pooled["wall_ms"], 2) if pooled["wall_ms"] else None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "pool_concurrency_bench",
        "config_file": config,
        "note": "dense_search over a pre-embedded query vector; DB time isolated. "
        "pool=1 reproduces the old single shared connection (serialized); pool=8 is "
        "the new default (parallel). Same code path, only the pool size differs.",
        "top_k": top_k,
        "arms": {"serialized_pool1": serialized, "pooled_pool8": pooled},
        "wall_speedup_pool8_vs_pool1": speedup,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pool concurrency benchmark -> JSON")
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args.config, args.concurrency, args.top_k)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"pool_bench_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))

    s, p = report["arms"]["serialized_pool1"], report["arms"]["pooled_pool8"]
    print(f"Wrote {out_path}")
    print(f"  concurrency = {s['concurrency']}  top_k = {report['top_k']}")
    for label, a in (("pool=1 (old)", s), ("pool=8 (new)", p)):
        print(f"  {label}: wall {a['wall_ms']} ms  {a['throughput_qps']} qps  "
              f"p95 {a['per_request_ms']['p95']} ms")
    print(f"  wall speedup (pool8 vs pool1) = {report['wall_speedup_pool8_vs_pool1']}x")


if __name__ == "__main__":
    main()
