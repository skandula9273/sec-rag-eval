"""Tests for fatal-vs-transient error classification and the eval runner's fail-fast.

The behaviour under test is the fix for the "eval runner swallows infra failures"
debt: an account/billing/auth outage (Anthropic credit-out, OpenAI quota-out, a bad
key) must ABORT the run — not be retried per-question into a misleading partial that
still emits an aggregate. See src/sec_rag/eval/errors.py for the why.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import sec_rag.eval.run_financebench as rf
from sec_rag.eval.errors import fatal_reason
from sec_rag.ingest.financebench import Question

# --- fatal_reason: the classifier ------------------------------------------------

class _StatusError(Exception):
    """Mimics an SDK APIStatusError: carries a status_code attribute."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


# The exact Anthropic credit-depletion string, copied from a committed partial run
# (eval_results/*.json) — the failure this whole change exists to catch.
_ANTHROPIC_CREDIT = (
    "BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'Your credit balance is too low to access "
    "the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}}"
)
# OpenAI quota-out: a 429 that is identical to a transient rate limit but for the code.
_OPENAI_QUOTA = (
    "Error code: 429 - {'error': {'message': 'You exceeded your current quota, please "
    "check your plan and billing details.', 'type': 'insufficient_quota', 'code': "
    "'insufficient_quota'}}"
)


def test_anthropic_credit_depletion_is_fatal():
    # A 400 (not 401/402) -> must be caught by the message marker, not the status.
    assert fatal_reason(_StatusError(_ANTHROPIC_CREDIT, status_code=400)) is not None


def test_openai_insufficient_quota_is_fatal():
    # A 429 that IS fatal — distinguished from a transient 429 only by the marker.
    assert fatal_reason(_StatusError(_OPENAI_QUOTA, status_code=429)) is not None


def test_auth_401_is_fatal():
    assert fatal_reason(_StatusError("Unauthorized", status_code=401)) is not None


def test_permission_403_is_fatal():
    assert fatal_reason(_StatusError("Forbidden", status_code=403)) is not None


def test_plain_rate_limit_429_is_transient():
    # A 429 with NO billing marker is a real rate limit -> retryable, NOT fatal.
    assert fatal_reason(_StatusError("Error code: 429 - rate limit exceeded", 429)) is None


def test_transient_connection_error_is_not_fatal():
    assert fatal_reason(ConnectionError("connection reset by peer")) is None
    assert fatal_reason(TimeoutError("read timed out")) is None
    assert fatal_reason(_StatusError("Internal server error", status_code=500)) is None


def test_generic_400_bad_request_is_not_fatal():
    # A non-billing 400 stays transient (surfaced via n_errors), so we don't abort a
    # run on a one-off malformed request that a retry might clear.
    assert fatal_reason(_StatusError("Error code: 400 - bad request", 400)) is None


def test_fatal_reason_walks_the_cause_chain():
    # The engine may re-raise an SDK error wrapped in something else.
    inner = _StatusError(_ANTHROPIC_CREDIT, status_code=400)
    try:
        try:
            raise inner
        except Exception as e:
            raise ValueError("generation failed") from e
    except ValueError as wrapped:
        assert fatal_reason(wrapped) is not None


# --- the runner: fail-fast + completeness flag -----------------------------------

class _Dump(dict):
    """A stand-in for a pydantic sub-config: only .model_dump() is read by run()."""

    def model_dump(self):
        return dict(self)


def _fake_cfg():
    """The minimal shape run() reads — no files, no pydantic, no network."""
    return SimpleNamespace(
        eval=SimpleNamespace(recall_ks=[5, 10], dataset="fake", seed=13, faithfulness=False),
        generation=SimpleNamespace(model="claude-haiku-fake"),
        chunking=_Dump(strategy="token", size=512),
        embedding=SimpleNamespace(model="text-embedding-fake"),
        retrieval=_Dump(method="dense"),
    )


def _questions(n: int) -> list[Question]:
    return [
        Question(id=f"q{i}", question=f"question {i}?", answer=f"a{i}",
                 doc_name="DOC", evidence_texts=[f"evidence {i}"], question_type="metrics")
        for i in range(n)
    ]


class _Chunk:
    def __init__(self, content: str):
        self.content = content


def _install_fake_engine(monkeypatch, fatal_on_id: str | None):
    """Monkeypatch QueryEngine so retrieve() records which questions it saw and raises
    a fatal Anthropic credit error on `fatal_on_id`. Returns the shared `seen` list so
    a test can assert the run short-circuited (never reached later questions)."""
    seen: list[str] = []

    class _FakeEngine:
        def __init__(self, cfg):
            pass

        def retrieve(self, question: str, top_k: int = 10):
            qid = question.split()[1].rstrip("?")  # "question 2?" -> "2" (matches q{i})
            seen.append(f"q{qid}")
            if fatal_on_id is not None and f"q{qid}" == fatal_on_id:
                raise _StatusError(_ANTHROPIC_CREDIT, status_code=400)
            return [_Chunk(f"evidence {qid} lives here")], 1.0

        def close(self):
            pass

    monkeypatch.setattr(rf, "QueryEngine", _FakeEngine)
    monkeypatch.setattr(rf, "load_questions", lambda dataset: _questions(3))
    return seen


def test_runner_aborts_on_fatal_and_marks_incomplete(monkeypatch):
    seen = _install_fake_engine(monkeypatch, fatal_on_id="q1")
    report = rf.run(_fake_cfg(), match_mode="substring", retrieval_only=True)

    assert report["complete"] is False
    assert report["aborted"] is not None
    assert report["aborted"]["id"] == "q1"
    assert "credit balance" in report["aborted"]["error"].lower()
    # Only q0 scored; the fatal q1 aborted the loop, so q2 was NEVER attempted.
    assert report["n_scored"] == 1
    assert report["n_questions"] == 3
    assert seen == ["q0", "q1"]  # short-circuit: did not grind through q2


def test_runner_complete_on_clean_run(monkeypatch):
    _install_fake_engine(monkeypatch, fatal_on_id=None)
    report = rf.run(_fake_cfg(), match_mode="substring", retrieval_only=True)

    assert report["complete"] is True
    assert report["aborted"] is None
    assert report["n_scored"] == 3
    assert report["n_errors"] == 0


def test_main_exits_nonzero_on_incomplete(monkeypatch, tmp_path):
    _install_fake_engine(monkeypatch, fatal_on_id="q0")
    monkeypatch.setattr(rf, "load_config", lambda path: _fake_cfg())
    monkeypatch.setattr(
        rf.sys, "argv",
        ["run_financebench", "--no-generate", "--out-dir", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc:
        rf.main()
    assert exc.value.code == 1
    # The (partial) artifact is still written for debugging, flagged not-citable.
    written = list(tmp_path.glob("financebench_*.json"))
    assert len(written) == 1
