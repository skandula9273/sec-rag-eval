"""Tests for the README-vs-artifact numbers checker.

Pure-logic + a hermetic end-to-end: a fixture README, artifact, and manifest in a
tmp dir prove the checker (a) passes when they agree, (b) FAILS with a mismatch
when the README number drifts from the artifact, and (c) flags a stray unmapped
number. No network, no real files.
"""

import json

import pytest
import yaml

from sec_rag.eval import check_numbers as cn

# ---- pure helpers ---------------------------------------------------------

def test_safe_eval_arithmetic():
    assert cn._safe_eval("a / b * 100", {"a": 0.1267, "b": 0.2067}) == pytest.approx(61.3, abs=0.1)
    assert cn._safe_eval("(x + y) / z", {"x": 1, "y": 2, "z": 3}) == 1.0


def test_safe_eval_rejects_non_arithmetic():
    # names not provided, attribute access, and calls must all raise — no code exec.
    with pytest.raises(ValueError):
        cn._safe_eval("ms / 1000", {})            # unknown name
    with pytest.raises(ValueError):
        cn._safe_eval("__import__('os')", {})     # call node disallowed


@pytest.mark.parametrize(
    "s,expected",
    [("0.64", 0.64), ("$0.0045", 0.0045), ("~$0.017", 0.017), ("80%", 80.0),
     ("15,192", 15192.0), ("−0.05", -0.05), ("+0.127", 0.127)],
)
def test_parse_number(s, expected):
    assert cn._parse_number(s) == pytest.approx(expected)


def test_resolve_path_dict_list_and_missing():
    obj = {"a": {"b": [10, {"c": 0.5}]}}
    assert cn._resolve_path(obj, ["a", "b", 1, "c"]) == 0.5
    assert cn._resolve_path(obj, ["a", "b", 0]) == 10
    with pytest.raises(KeyError):
        cn._resolve_path(obj, ["a", "nope"])


# ---- end-to-end fixture ---------------------------------------------------

def _fixture(tmp_path, readme_body, manifest_extra=""):
    (tmp_path / "art.json").write_text(json.dumps({"recall_at_k": {"recall@5": 0.64}}))
    (tmp_path / "README.md").write_text(readme_body)
    manifest = {
        "readme": str(tmp_path / "README.md"),
        "artifacts_dir": str(tmp_path),
        "claims": [{
            "id": "recall5",
            "anchor": r"recall@5 \*\*(0\.\d+)\*\*",
            "artifact": "art.json",
            "path": ["recall_at_k", "recall@5"],
            "tol": 0.005,
        }],
        "allow": [],
        "ignore": [],
    }
    mpath = tmp_path / "claims.yaml"
    mpath.write_text(yaml.safe_dump(manifest) + manifest_extra)
    return mpath


def test_end_to_end_pass(tmp_path):
    mpath = _fixture(tmp_path, "recall@5 **0.64** on the benchmark.\n")
    results, unsourced, unmapped = cn.check(mpath)
    assert [r.status for r in results] == ["ok"]
    assert unmapped == []


def test_end_to_end_catches_drift(tmp_path):
    # README says 0.70, artifact still says 0.64 -> must be flagged mismatch, not ok.
    mpath = _fixture(tmp_path, "recall@5 **0.70** on the benchmark.\n")
    results, _, _ = cn.check(mpath)
    assert results[0].status == "mismatch"
    assert results[0].readme == 0.70 and results[0].expected == 0.64


def test_end_to_end_flags_unmapped_number(tmp_path):
    # A second, unregistered claim-shaped number must surface as unmapped.
    mpath = _fixture(tmp_path, "recall@5 **0.64**, and faithfulness 0.91 too.\n")
    _, _, unmapped = cn.check(mpath)
    assert any(u["value"] == "0.91" for u in unmapped)


def test_ambiguous_anchor_is_an_error(tmp_path):
    # Same anchor matching twice is unsafe (which number did we check?) -> error.
    mpath = _fixture(tmp_path, "recall@5 **0.64** ... recall@5 **0.64** again.\n")
    results, _, _ = cn.check(mpath)
    assert results[0].status == "error"


def test_real_manifest_is_consistent():
    """The committed manifest must be green against the committed README/artifacts —
    this is the regression guard that runs in CI via `make check-numbers`."""
    results, _, unmapped = cn.check(cn.REPO_ROOT / "configs" / "claims.yaml")
    bad = [(r.id, r.detail) for r in results if r.status != "ok"]
    assert not bad, f"README drifted from artifacts: {bad}"
    assert unmapped == [], f"unregistered numbers in README: {unmapped}"
