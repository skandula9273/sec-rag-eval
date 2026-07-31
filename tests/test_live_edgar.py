"""Offline tests for the live-EDGAR eval harness.

These validate the harness logic (question-set integrity, filing-drift handling, the
fail-fast abort, completeness) WITHOUT touching EDGAR, OpenAI, or Anthropic — the
LiveEngine and the S2 scorer are monkeypatched. The live accuracy NUMBER itself needs
real API calls and is produced by running the module; this file guards the plumbing.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import sec_rag.eval.live_edgar as le
from sec_rag.eval.answer_accuracy import AnswerScore

_VALID_FORMS = {"10-K", "10-Q", "8-K"}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --- the committed question set --------------------------------------------------

def test_question_set_integrity():
    qs = le.load_live_questions()
    assert 20 <= len(qs) <= 30, f"expected 20-30 hand-verified questions, got {len(qs)}"
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids)), "question ids must be unique"
    for q in qs:
        for key in ("id", "ticker", "form", "question", "gold", "gold_filing_date"):
            assert q.get(key), f"{q.get('id')} missing {key}"
        assert q["form"] in _VALID_FORMS, f"{q['id']} bad form {q['form']}"
        assert _DATE.match(q["gold_filing_date"]), f"{q['id']} bad date {q['gold_filing_date']}"
    # Covers all three filing types (the product auto-detects among them).
    assert _VALID_FORMS <= {q["form"] for q in qs}


# --- run(): drift, abort, completeness -------------------------------------------

class _StatusError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


_CREDIT = ("BadRequestError: Error code: 400 - 'Your credit balance is too low to "
           "access the Anthropic API.'")


def _fake_cfg():
    return SimpleNamespace(
        embedding=SimpleNamespace(model="text-embedding-fake"),
        generation=SimpleNamespace(model="claude-haiku-fake"),
        retrieval=SimpleNamespace(top_k=5),
        chunking=SimpleNamespace(model_dump=lambda: {"strategy": "section_then_token"}),
    )


def _done(answer: str, filing_date: str):
    resp = SimpleNamespace(
        answer=answer,
        citations=[SimpleNamespace(filing_date=filing_date)],
        metrics=SimpleNamespace(latency_ms=1234, cost_usd=0.002, chunks_retrieved=5),
    )
    return {"type": "done", "response": resp}


def _install(monkeypatch, questions, stream_impl):
    """Patch load_live_questions + LiveEngine + score_answer for an offline run."""
    monkeypatch.setattr(le, "load_live_questions", lambda *a, **k: list(questions))

    class _FakeEngine:
        def __init__(self, cfg, secrets=None):
            pass

        def stream(self, ticker, question, form="auto"):
            yield from stream_impl(ticker, question, form)

        def close(self):
            pass

    monkeypatch.setattr(le, "LiveEngine", _FakeEngine)
    # Stub the S2 scorer (would call Anthropic): correct iff the answer equals the gold.
    def _fake_score(question, gold, generated, *, judge_model, secrets=None):
        return AnswerScore(refused=False, numeric=None, llm_correct=(generated == gold))
    monkeypatch.setattr(le, "score_answer", _fake_score)


def _q(id_, form, gold, gold_date):
    return {"id": id_, "ticker": "X", "form": form, "question": f"q {id_}?",
            "gold": gold, "gold_filing_date": gold_date}


def test_run_scores_and_flags_filing_drift(monkeypatch):
    questions = [
        _q("a", "10-K", "right", "2025-10-31"),   # correct, no drift
        _q("b", "10-K", "right", "2025-10-31"),   # wrong answer, no drift
        _q("c", "10-Q", "right", "2026-04-29"),   # correct BUT filing rotated -> drift
    ]

    def stream_impl(ticker, question, form):
        if question.startswith("q a"):
            yield _done("right", "2025-10-31")
        elif question.startswith("q b"):
            yield _done("wrong", "2025-10-31")
        else:  # c: live filing is NEWER than the verified gold date -> drift
            yield _done("right", "2026-07-30")

    _install(monkeypatch, questions, stream_impl)
    report = le.run(_fake_cfg())

    assert report["complete"] is True
    assert report["n_scored"] == 3
    assert report["filing_drift_count"] == 1
    # 'fresh' excludes the drifted row: 2 scored (a,b), 1 correct -> 0.5
    assert report["accuracy_fresh"]["llm"]["accuracy"] == 0.5
    assert report["accuracy_fresh"]["n"] == 2
    # 'all' includes the drifted (correct) row: 3 scored, 2 correct
    assert report["accuracy_all"]["llm"]["accuracy"] == round(2 / 3, 4)
    # per-form only over fresh rows -> 10-Q dropped (its only row drifted)
    assert set(report["per_form"]) == {"10-K"}


def test_run_aborts_on_fatal_billing(monkeypatch):
    questions = [
        _q("a", "10-K", "right", "2025-10-31"),
        _q("b", "10-K", "right", "2025-10-31"),
    ]
    seen = []

    def stream_impl(ticker, question, form):
        seen.append(question)
        if question.startswith("q a"):
            yield _done("right", "2025-10-31")
        else:
            raise _StatusError(_CREDIT, status_code=400)

    _install(monkeypatch, questions, stream_impl)
    report = le.run(_fake_cfg())

    assert report["complete"] is False
    assert report["aborted"] is not None and report["aborted"]["id"] == "b"
    assert "credit balance" in report["aborted"]["error"].lower()
    assert report["n_scored"] == 1  # only 'a' scored before the abort
