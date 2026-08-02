"""Numbers-consistency checker — README claims vs. committed artifacts.

The failure mode this kills: the README asserts a number (recall@5 0.64), the
truth lives in `eval_results/*.json`, and the two drift apart silently between
runs. This tool makes that class of bug fail loudly instead of being found by
reading.

How it works (three checks, run over `configs/claims.yaml`):

1. **Sourced claims** — each has an `anchor` regex that captures the number *from
   the current README text*, an `artifact` file, and a `path` (list of keys) into
   that file. The tool extracts the README value and the artifact value and fails
   if they differ by more than `tol`. Because the value is read from the README
   *text*, editing the README without touching the artifact fails here; committing
   a new artifact whose value no longer matches the (unchanged) README also fails.
   A claim may instead give `values:` (named artifact lookups) + `expr:` for
   derived numbers (e.g. a percentage of two components), evaluated with a
   restricted arithmetic evaluator — no artifact, no `eval`.

2. **Unsourced claims** — numbers with no committed artifact (corpus stats, live
   observations, design-doc targets, the test count). Recorded on purpose, with a
   `reason`, so "this has no artifact" is *expressible and reviewed*, not
   indistinguishable from an un-audited number. Listed, never failed.

3. **Coverage scan** — finds every claim-shaped number in the README (decimals,
   `$`-amounts, percentages) that is NOT inside a registered anchor, NOT an
   allow-listed constant, and NOT on an ignored line. Those are *unmapped* — a new
   number nobody vouched for. Under `--strict` (CI + `make check-numbers`) an
   unmapped number fails the run, so you cannot ship a number without registering
   it as sourced, unsourced, or an allowed constant.

Deliberate-run handling: the manifest records WHICH artifact each claim cites, so
"this deliberately cites the 2026-06-29 run" is a line in `configs/claims.yaml`,
not a guess — an older run is a valid citation, staleness is a mismatch, and the
two are distinguishable.

Exit codes: 0 = clean · 2 = a sourced claim mismatched (always fatal) · 3 =
unmapped numbers under --strict. Every problem is reported, not just the first.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

# Claim-shaped numbers the coverage scan cares about: decimals, $-amounts, and
# percentages. Bare integers (150 questions, 84 filings, seed 13) are structural,
# not result-shaped, so they are not auto-scanned — register them explicitly if
# they matter.
_CLAIM_NUMBER = re.compile(r"\$\s?\d+(?:\.\d+)?|\d+\.\d+%?|\b\d{1,3}%")

_ARITH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str, names: dict[str, float]) -> float:
    """Evaluate an arithmetic expression over `names` — +,-,*,/ and parens only.

    AST-walked, not `eval`: any node other than a number, a registered name, or a
    basic arithmetic op raises. So a manifest cannot smuggle in code.
    """

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
            return _ARITH_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_OPS:
            return _ARITH_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            raise ValueError(f"unknown name in expr: {node.id!r}")
        raise ValueError(f"disallowed expression element: {ast.dump(node)}")

    return ev(ast.parse(expr, mode="eval"))


def _parse_number(s: str) -> float:
    """First numeric token in `s`, tolerant of $, %, commas, ~, <, >, and both
    ASCII '-' and Unicode '−' signs."""
    cleaned = s.replace(",", "").replace("−", "-")
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not m:
        raise ValueError(f"no number found in {s!r}")
    return float(m.group())


def _resolve_path(obj, keys: list):
    """Walk a list-of-keys path into a loaded JSON/YAML object. Ints index lists.

    Keys are a *list* (not a dotted string) so keys containing dots/slashes/@ —
    `recall@5`, `3-large/1024`, `metrics-generated` — need no escaping.
    """
    cur = obj
    for k in keys:
        if isinstance(cur, list):
            cur = cur[int(k)]
        elif isinstance(cur, dict):
            if k not in cur:
                raise KeyError(f"missing key {k!r} (path so far: {keys})")
            cur = cur[k]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} for key {k!r}")
    return cur


@dataclass
class Manifest:
    readme_path: Path
    artifacts_dir: Path
    claims: list
    unsourced: list
    allow: list
    ignore: list[re.Pattern] = field(default_factory=list)


def load_manifest(path: Path) -> Manifest:
    raw = yaml.safe_load(path.read_text())
    return Manifest(
        readme_path=REPO_ROOT / raw.get("readme", "README.md"),
        artifacts_dir=REPO_ROOT / raw.get("artifacts_dir", "eval_results"),
        claims=raw.get("claims", []) or [],
        unsourced=raw.get("unsourced", []) or [],
        allow=raw.get("allow", []) or [],
        ignore=[re.compile(p) for p in (raw.get("ignore", []) or [])],
    )


_artifact_cache: dict[Path, object] = {}


def _load_artifact(name: str, m: Manifest):
    # A bare name lives under artifacts_dir; a path with a slash is repo-relative
    # (so a claim can cite configs/ci_eval.yaml, not only eval_results/*.json).
    p = (REPO_ROOT / name) if "/" in name else (m.artifacts_dir / name)
    if p not in _artifact_cache:
        text = p.read_text()
        _artifact_cache[p] = (
            yaml.safe_load(text) if p.suffix in (".yaml", ".yml") else json.loads(text)
        )
    return _artifact_cache[p]


def _find_anchor(readme: str, pattern: str) -> re.Match:
    matches = list(re.finditer(pattern, readme))
    if not matches:
        raise LookupError("anchor not found in README")
    if len(matches) > 1:
        raise LookupError(f"anchor is ambiguous — matched {len(matches)}x")
    return matches[0]


def _expected_value(claim: dict, m: Manifest) -> float:
    """The artifact-side value for a claim: a single `path`, or `expr` over `values`."""
    if "expr" in claim:
        names = {
            alias: float(_resolve_path(_load_artifact(art, m), path))
            for alias, (art, path) in claim["values"].items()
        }
        return _safe_eval(claim["expr"], names)
    return float(_resolve_path(_load_artifact(claim["artifact"], m), claim["path"]))


@dataclass
class Result:
    id: str
    status: str  # ok | mismatch | error
    readme: float | None = None
    expected: float | None = None
    tol: float | None = None
    detail: str = ""


def check(manifest_path: Path) -> tuple[list[Result], list[dict], list[dict]]:
    m = load_manifest(manifest_path)
    readme = m.readme_path.read_text()

    results: list[Result] = []
    covered_spans: list[tuple[int, int]] = []

    for claim in m.claims:
        cid = claim["id"]
        try:
            match = _find_anchor(readme, claim["anchor"])
            covered_spans.append(match.span())
            readme_val = _parse_number(match.group(1))
            expected = _expected_value(claim, m)
            tol = float(claim.get("tol", 0.005))
            if abs(readme_val - expected) <= tol:
                results.append(Result(cid, "ok", readme_val, expected, tol))
            else:
                src = claim.get("artifact") or "+".join(
                    v[0] for v in claim.get("values", {}).values()
                )
                results.append(
                    Result(cid, "mismatch", readme_val, expected, tol,
                           detail=f"{claim.get('desc','')}  [{src}]")
                )
        except Exception as e:  # noqa: BLE001 — surface any manifest/artifact error as a failure
            results.append(Result(cid, "error", detail=f"{type(e).__name__}: {e}"))

    # Unsourced anchors still consume README coverage (they are acknowledged text).
    for u in m.unsourced:
        for mt in re.finditer(u["anchor"], readme):
            covered_spans.append(mt.span())

    unmapped = _coverage_scan(readme, m, covered_spans)
    return results, m.unsourced, unmapped


def _coverage_scan(readme: str, m: Manifest, covered: list[tuple[int, int]]) -> list[dict]:
    allow_values = [(float(a["value"]), float(a.get("tol", 0.0005))) for a in m.allow]
    lines = readme.splitlines(keepends=True)
    line_starts, pos = [], 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)

    def line_of(idx: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo

    unmapped = []
    for tok in _CLAIM_NUMBER.finditer(readme):
        s, e = tok.span()
        if any(cs <= s and e <= ce for cs, ce in covered):
            continue
        line_no = line_of(s)
        line_text = lines[line_no]
        if any(p.search(line_text) for p in m.ignore):
            continue
        try:
            val = _parse_number(tok.group())
        except ValueError:
            continue
        if any(abs(val - av) <= at for av, at in allow_values):
            continue
        unmapped.append(
            {"value": tok.group().strip(), "line": line_no + 1, "text": line_text.strip()[:90]}
        )
    return unmapped


def _report(results: list[Result], unsourced: list[dict], unmapped: list[dict],
            strict: bool) -> int:
    ok = [r for r in results if r.status == "ok"]
    bad = [r for r in results if r.status != "ok"]

    print(f"\nNumbers-consistency check — {len(results)} sourced claims\n" + "=" * 60)
    for r in ok:
        print(f"  ✓ {r.id:<34} README {r.readme} == artifact {r.expected} (±{r.tol})")

    if bad:
        print("\nMISMATCHES / ERRORS (fatal):")
        for r in bad:
            if r.status == "mismatch":
                delta = abs(r.readme - r.expected)
                print(f"  ✗ {r.id:<32} README {r.readme}  !=  artifact {r.expected} "
                      f"(Δ {delta:.4g} > tol {r.tol})\n      {r.detail}")
            else:
                print(f"  ✗ {r.id:<32} could not verify — {r.detail}")

    print(f"\nUNSOURCED (no committed artifact — acknowledged, {len(unsourced)}):")
    for u in unsourced:
        print(f"  • {u['id']:<30} {u.get('desc','')}  — {u.get('reason','')}")

    print(f"\nUNMAPPED claim-shaped numbers (not registered, {len(unmapped)}):")
    for u in unmapped:
        print(f"  ? line {u['line']:<4} {u['value']:<10} {u['text']}")
    if unmapped:
        print("    → add each to configs/claims.yaml as a claim, an `unsourced` entry, "
              "an `allow` constant, or an `ignore` line.")

    n_bad = len(bad)
    print("\n" + "=" * 60)
    print(f"summary: {len(ok)} ok · {n_bad} mismatch/error · "
          f"{len(unsourced)} unsourced · {len(unmapped)} unmapped")

    if n_bad:
        return 2
    if unmapped and strict:
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check README numbers against committed artifacts.")
    ap.add_argument("--manifest", default=str(REPO_ROOT / "configs" / "claims.yaml"))
    ap.add_argument("--strict", action="store_true",
                    help="fail (exit 3) when unmapped numbers exist (CI / make check-numbers)")
    args = ap.parse_args(argv)
    results, unsourced, unmapped = check(Path(args.manifest))
    return _report(results, unsourced, unmapped, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
