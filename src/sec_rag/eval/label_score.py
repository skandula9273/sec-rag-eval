"""Score each automatic matcher against human labels.

Joins the committed sample (per-pair matcher verdicts) to the human labels and, per
matcher, reports Cohen's kappa (+ an approximate 95% CI), precision, recall, and the
confusion matrix, treating the human label as ground truth and a matcher "match" as
the positive class. Emits one JSON to eval_results/.

The kappa/precision/recall functions are pure and unit-tested; the runner just reads
files. Nothing here labels anything — it needs metric_validity_labels.jsonl (yours).

  python -m sec_rag.eval.label_score
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def confusion(preds: list[bool], labels: list[bool]) -> dict[str, int]:
    """TP/FP/FN/TN with human ``labels`` as truth and matcher ``preds`` as positive."""
    tp = sum(1 for p, y in zip(preds, labels, strict=True) if p and y)
    fp = sum(1 for p, y in zip(preds, labels, strict=True) if p and not y)
    fn = sum(1 for p, y in zip(preds, labels, strict=True) if not p and y)
    tn = sum(1 for p, y in zip(preds, labels, strict=True) if not p and not y)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall(preds: list[bool], labels: list[bool]) -> tuple[float | None, float | None]:
    c = confusion(preds, labels)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else None
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None
    return prec, rec


def cohen_kappa(preds: list[bool], labels: list[bool]) -> float:
    """Cohen's kappa between two binary raters (matcher vs human)."""
    n = len(preds)
    if n == 0:
        return 0.0
    po = sum(1 for p, y in zip(preds, labels, strict=True) if p == y) / n
    p_m = sum(preds) / n
    p_h = sum(labels) / n
    pe = p_m * p_h + (1 - p_m) * (1 - p_h)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def kappa_ci(preds: list[bool], labels: list[bool], z: float = 1.96) -> tuple[float, float, float]:
    """(kappa, lo, hi) with a first-order large-sample SE — indicative at n~50, not exact."""
    n = len(preds)
    k = cohen_kappa(preds, labels)
    if n == 0:
        return k, k, k
    po = sum(1 for p, y in zip(preds, labels, strict=True) if p == y) / n
    p_m, p_h = sum(preds) / n, sum(labels) / n
    pe = p_m * p_h + (1 - p_m) * (1 - p_h)
    if pe >= 1.0 or po >= 1.0:
        return k, k, k
    se = math.sqrt(po * (1 - po) / n) / (1 - pe)
    return k, k - z * se, k + z * se


def _load_sample(path: str) -> dict[str, dict]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    return {r["pair_id"]: r for r in rows if "_meta" not in r}


def _load_labels(path: str) -> dict[str, int]:
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["pair_id"]] = int(r["supports"])
    return out


def score(sample_path: str, labels_path: str) -> dict:
    sample = _load_sample(sample_path)
    labels = _load_labels(labels_path)
    joined = [(pid, sample[pid], labels[pid]) for pid in labels if pid in sample]
    if not joined:
        raise SystemExit(
            f"No labeled pairs found. Label first: python -m sec_rag.eval.label_cli "
            f"(sample={sample_path}, labels={labels_path})"
        )

    matcher_names = sorted(next(iter(sample.values()))["matchers"].keys())
    human = [bool(y) for _, _, y in joined]
    per_matcher = {}
    for name in matcher_names:
        preds = [bool(row["matchers"][name]) for _, row, _ in joined]
        k, lo, hi = kappa_ci(preds, human)
        prec, rec = precision_recall(preds, human)
        per_matcher[name] = {
            "cohen_kappa": round(k, 4),
            "kappa_95ci": [round(lo, 4), round(hi, 4)],
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "confusion": confusion(preds, human),
        }

    ranked = sorted(per_matcher, key=lambda m: per_matcher[m]["cohen_kappa"], reverse=True)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "metric_validity_scores",
        "n_labeled": len(joined),
        "human_yes_rate": round(sum(human) / len(human), 4),
        "sample_note": "disagreement-oversampled: metrics are on the CONTESTED pairs, "
        "not a population rate (see docs/metric-validity.md).",
        "best_agreement_matcher": ranked[0],
        "kappa_ranking": ranked,
        "by_matcher": per_matcher,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score matchers vs human labels -> JSON")
    ap.add_argument("--sample", default="eval_results/metric_validity_sample.jsonl")
    ap.add_argument("--labels", default="eval_results/metric_validity_labels.jsonl")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = score(args.sample, args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = out_dir / f"metric_validity_scores_{stamp}.json"
    p.write_text(json.dumps(report, indent=2))

    print(f"Wrote {p}  (n_labeled={report['n_labeled']})\n")
    print(f"{'matcher':12}{'kappa':>8}{'95% CI':>18}{'prec':>8}{'recall':>8}")
    for m in report["kappa_ranking"]:
        r = report["by_matcher"][m]
        ci = f"[{r['kappa_95ci'][0]}, {r['kappa_95ci'][1]}]"
        print(f"{m:12}{r['cohen_kappa']:>8}{ci:>18}{str(r['precision']):>8}{str(r['recall']):>8}")
    print(f"\nbest agreement with human: {report['best_agreement_matcher']}")


if __name__ == "__main__":
    main()
