"""Live-API latency + cost measurement (the deployed Cloud Run service).

CLAUDE.md long asserted the live API was "~$0.005-6 and faster than the eval's
15.3 s" without a committed measurement — a violation of the repo's own
"if it can't be rerun, it doesn't count" rule. This script measures it for real.

It hits the deployed ``POST /query`` (the non-streaming path a user's answer comes
from: server keys, judge OFF, top_k as sent) with a fixed set of single-company
questions, and records BOTH:

  * client wall-clock per request (includes network RTT to the region — the true
    user-facing latency), and
  * the server's own ``metrics.latency_ms`` breakdown (retrieval + generation),
    which is apples-to-apples with the eval's server-side numbers, plus the
    per-request ``cost_usd`` the API reports.

The FIRST request is reported separately as a cold start (Cloud Run scales to
zero); warm percentiles exclude it. Optionally (``--stream``) it also measures
time-to-first-token on ``POST /query/stream`` — the metric the streaming path
exists for.

This is a CLIENT-SIDE measurement from wherever it is run; the region and network
path are recorded in the output so the number is interpretable, not absolute.

Usage:
  python -m sec_rag.eval.measure_api_latency --n 20 --top-k 5 --stream
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BASE = "https://sec-rag-api-200217758117.us-east1.run.app"

# Single-company, corpus-answerable questions — varied companies/metrics so
# generation length (the latency driver) is realistic, not a pathological refusal.
QUESTIONS = [
    "What was Costco's total revenue in FY2021?",
    "What was Boeing's net property, plant and equipment at year end FY2018?",
    "Did AMD report customer concentration in FY2022?",
    "What drove AMD's revenue change in FY2022?",
    "What was Nike's total current assets at the end of FY2019?",
    "What was Amcor's net accounts receivable at year end FY2020?",
    "Does 3M maintain a stable dividend distribution trend?",
    "What was Block's total net revenue in FY2020?",
    "What was Coca-Cola's net income in FY2017?",
    "What are AMD's major products and services as of FY2022?",
    "What was Microsoft's total revenue in FY2016?",
    "What drove 3M's operating margin change in FY2022?",
    "What was CVS Health's total revenue in FY2018?",
    "What was Corning's effective tax rate in FY2020?",
    "What was American Water Works' total current liabilities in FY2022?",
    "How did Boeing's effective tax rate in FY2022 compare to FY2021?",
    "What was Costco's total assets at the end of FY2021?",
    "What were Best Buy's domestic segment store counts in FY2023?",
    "How does Amazon describe its working capital management?",
    "What was Coca-Cola's total current assets in FY2017?",
]


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
        "p99": round(_percentile(values, 99), 1),
        "mean": round(sum(values) / len(values), 1) if values else 0.0,
        "min": round(min(values), 1) if values else 0.0,
        "max": round(max(values), 1) if values else 0.0,
    }


def _post_query(base: str, query: str, top_k: int | None) -> tuple[int, float, dict]:
    """POST /query. Returns (http_status, client_wall_ms, parsed_body)."""
    body = {"query": query}
    if top_k is not None:
        body["top_k"] = top_k
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base}/query", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            wall = (time.perf_counter() - t0) * 1000
            return resp.status, wall, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        wall = (time.perf_counter() - t0) * 1000
        return e.code, wall, {"error": e.read().decode()[:200]}


def _ttft_stream(base: str, query: str, top_k: int | None) -> tuple[int, float, float]:
    """POST /query/stream. Returns (status, time_to_first_token_ms, total_ms)."""
    body = {"query": query}
    if top_k is not None:
        body["top_k"] = top_k
    req = urllib.request.Request(
        f"{base}/query/stream", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace")
                if ttft is None and '"type": "token"' in line.replace('"type":"token"', '"type": "token"'):
                    ttft = (time.perf_counter() - t0) * 1000
            total = (time.perf_counter() - t0) * 1000
            return resp.status, round(ttft or total, 1), round(total, 1)
    except urllib.error.HTTPError as e:
        return e.code, 0.0, round((time.perf_counter() - t0) * 1000, 1)


def run(base: str, n: int, top_k: int | None, do_stream: bool) -> dict:
    questions = (QUESTIONS * ((n // len(QUESTIONS)) + 1))[:n]

    rows: list[dict] = []
    for i, q in enumerate(questions):
        status, wall_ms, body = _post_query(base, q, top_k)
        m = body.get("metrics", {}) if isinstance(body, dict) else {}
        rows.append({
            "i": i,
            "cold": i == 0,
            "http": status,
            "client_wall_ms": round(wall_ms, 1),
            "server_latency_ms": m.get("latency_ms"),
            "retrieval_ms": m.get("retrieval_ms"),
            "generation_ms": m.get("generation_ms"),
            "cost_usd": m.get("cost_usd"),
            "tokens_in": m.get("tokens_in"),
            "tokens_out": m.get("tokens_out"),
            "model": body.get("model") if isinstance(body, dict) else None,
        })

    ok = [r for r in rows if r["http"] == 200]
    warm = [r for r in ok if not r["cold"]]
    cold = next((r for r in rows if r["cold"]), None)
    costs = [r["cost_usd"] for r in warm if r["cost_usd"] is not None]
    server_ms = [r["server_latency_ms"] for r in warm if r["server_latency_ms"] is not None]

    stream_rows: list[dict] = []
    if do_stream:
        for q in QUESTIONS[:8]:
            st, ttft, total = _ttft_stream(base, q, top_k)
            stream_rows.append({"http": st, "ttft_ms": ttft, "total_ms": total})

    ttfts = [s["ttft_ms"] for s in stream_rows if s["http"] == 200]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "live_api_latency",
        "base_url": base,
        "endpoint": "/query (non-stream, server keys, faithfulness judge OFF)",
        "region": "us-east1 (Cloud Run); measured client-side — includes network RTT",
        "top_k_sent": top_k,
        "n_requests": len(rows),
        "n_ok": len(ok),
        "model": next((r["model"] for r in ok if r["model"]), None),
        "cold_start": {
            "client_wall_ms": cold["client_wall_ms"] if cold else None,
            "server_latency_ms": cold["server_latency_ms"] if cold else None,
        },
        "warm_client_wall_ms": _stats([r["client_wall_ms"] for r in warm]),
        "warm_server_latency_ms": _stats([float(x) for x in server_ms]),
        "cost_per_query_usd": {
            "mean": round(sum(costs) / len(costs), 6) if costs else None,
            "min": round(min(costs), 6) if costs else None,
            "max": round(max(costs), 6) if costs else None,
        },
        "stream_ttft_ms": _stats([float(x) for x in ttfts]) if ttfts else None,
        "rows": rows,
        "stream_rows": stream_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure live API latency -> JSON")
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--n", type=int, default=20, help="total requests (first = cold start)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--stream", action="store_true", help="also measure /query/stream TTFT")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args.base_url, args.n, args.top_k, args.stream)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"api_latency_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote {out_path}")
    print(f"  n_ok = {report['n_ok']}/{report['n_requests']}  model = {report['model']}")
    print(f"  cold start (client)      = {report['cold_start']['client_wall_ms']} ms")
    w = report["warm_client_wall_ms"]
    print(f"  warm client p50/p95/p99  = {w['p50']}/{w['p95']}/{w['p99']} ms")
    s = report["warm_server_latency_ms"]
    print(f"  warm server p50/p95/p99  = {s['p50']}/{s['p95']}/{s['p99']} ms")
    print(f"  cost/query (mean)        = ${report['cost_per_query_usd']['mean']}")
    if report["stream_ttft_ms"]:
        print(f"  stream TTFT p50/p95      = {report['stream_ttft_ms']['p50']}/{report['stream_ttft_ms']['p95']} ms")


if __name__ == "__main__":
    main()
