"""Answer accuracy — does the generated answer match FinanceBench's gold answer?

FinanceBench ships a gold ``answer`` per question. Retrieval recall says nothing
about whether the final answer is *right*; this module scores that, so the project
can report a number comparable to published FinanceBench baselines (which are
LLM-graded). Two scorers, deliberately different in strictness:

1. ``numeric_match`` — a DETERMINISTIC exact/numeric matcher. Only applies when the
   gold answer reduces to a single figure (most ``metrics-generated`` questions);
   returns ``None`` ("not applicable") for sentence / multi-number / yes-no golds.

2. ``grade_correctness`` — an LLM judge (model configurable + recorded) that grades
   correct/incorrect leniently on format but strictly on the value/conclusion.

Refusals are detected separately (``is_refusal``) and never counted as wrong — an
unanswered question is not an incorrect one.

Numeric normalization rules (write them down so the number is auditable):
  * currency symbols ($ € £ ¥) are stripped.
  * thousands separators (commas) removed: "1,577" -> 1577.
  * scale words/suffixes multiply: k/thousand ->1e3, m/mm/mn/million ->1e6,
    b/bn/billion ->1e9, t/tn/trillion ->1e12. So "$1.2B" and "1,200 million" both
    normalize to 1.2e9.
  * accounting negatives: a value in parentheses "(1,234)" is negative, as is a
    leading "-". Sign must match for a numeric hit.
  * percentages ("16%", "16 percent") are kept as their own kind and compared ONLY
    to other percentages — never scale-bridged (so 1.9% never matches 1900).
  * citation markers like "[1]" are stripped from the generated answer before
    number extraction (they are references, not figures).
  * SCALE BRIDGING: FinanceBench stores figures bare when the question says
    "in USD millions" (gold "$1577.00" = $1,577 million), while the model answers
    "$1,577 million". So a gold number matches a candidate if the candidate equals
    gold x f for f in {1e-9,1e-6,1e-3,1,1e3,1e6,1e9}, within ``rel_tol`` (default
    1%). This bridges thousand/million/billion conventions; it can (rarely)
    false-match two figures exactly a power of 1000 apart — a documented limitation,
    which is why the LLM grader is the primary accuracy number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sec_rag.config import Secrets

_SCALE = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12,
}
# A numeric token: optional leading -/$/( , the digits (with commas/decimal), an
# optional scale word or %, and an optional trailing ) for accounting negatives.
_NUM = re.compile(
    r"""(?P<open>[-(])?\s*\$?\s*
        (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)   # 1,577.00 or 24.26 or 0
        \s*(?P<scale>billion|trillion|million|thousand|bn|tn|mn|mm|[kbmt]\b|%|percent)?
        \s*(?P<close>\))?""",
    re.I | re.X,
)
_CITE = re.compile(r"\[\d+\]")
_BRIDGE = (1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9)


def extract_numbers(text: str) -> list[tuple[float, bool]]:
    """All numeric values in ``text`` as (value, is_percent). Scale applied, sign
    from parens/minus, citation markers removed first."""
    text = _CITE.sub(" ", text or "")
    out: list[tuple[float, bool]] = []
    for m in _NUM.finditer(text):
        raw = m.group("num").replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        scale = (m.group("scale") or "").lower()
        is_pct = scale in ("%", "percent")
        if scale and not is_pct:
            val *= _SCALE.get(scale.rstrip("."), 1.0)
        negative = m.group("open") == "-" or (m.group("open") == "(" and m.group("close") == ")")
        if negative:
            val = -val
        out.append((val, is_pct))
    return out


def _parse_gold(gold: str) -> tuple[str, float | None]:
    """Gold -> ('num'|'pct'|'na', value). 'na' when it is not a single clean figure
    (a sentence, a yes/no, or several numbers) — the numeric matcher abstains there."""
    nums = extract_numbers(gold or "")
    if len(nums) != 1:
        return ("na", None)
    val, is_pct = nums[0]
    return ("pct" if is_pct else "num", val)


def numeric_match(gold: str, generated: str, *, rel_tol: float = 0.01) -> bool | None:
    """True/False if the gold is a single figure and the generated answer does/doesn't
    contain a matching number; None if the gold isn't a single figure (use the LLM grader).
    """
    kind, gval = _parse_gold(gold)
    if kind == "na" or gval is None:
        return None
    cands = extract_numbers(generated or "")
    if kind == "pct":
        pcts = [v for v, p in cands if p]
        return any(abs(v - gval) <= max(rel_tol * abs(gval), 0.1) for v in pcts)
    # numeric (non-percent): sign must match; bridge power-of-1000 scale conventions.
    nums = [v for v, p in cands if not p]
    if gval == 0.0:
        return any(abs(v) <= 0.5 for v in nums)
    for f in _BRIDGE:
        target = gval * f
        if any(v * target > 0 and abs(v - target) <= rel_tol * abs(target) for v in nums):
            return True
    return False


# Assistant-refusal patterns (grounded refusal from the concise answer prompt +
# fallbacks). Anchored on the assistant declining, not on a real answer that merely
# contains "cannot" (e.g. "the company cannot pay dividends").
_REFUSAL_RE = re.compile(
    r"\b(i cannot answer|i can't answer|cannot be answered|cannot be determined "
    r"from the (?:provided )?sources|the sources (?:do not|don't|does not) "
    r"(?:contain|provide|include)|not (?:available|found|provided|present) in the "
    r"(?:provided )?sources|unable to (?:answer|determine|find))",
    re.I,
)


def is_refusal(generated: str) -> bool:
    """The answer declined to answer (grounded refusal) rather than asserting a fact."""
    return bool(_REFUSAL_RE.search(generated or ""))


@dataclass
class AnswerScore:
    refused: bool
    numeric: bool | None      # None = numeric matcher not applicable (non-figure gold)
    llm_correct: bool | None  # None only if refused (no correctness call made)


_JUDGE_SYSTEM = (
    "You grade a financial-QA answer against a known-correct GOLD answer. Reply with "
    "exactly one word: correct or incorrect. Mark 'correct' if the ANSWER conveys the "
    "same key figure or conclusion as the GOLD, allowing differences in wording, "
    "rounding (~1%), units and scale ($1.2B = 1,200 million), sign conventions stated "
    "the same way, and extra explanation. Mark 'incorrect' if the key figure or "
    "conclusion differs from, contradicts, or is missing versus the GOLD. Do not "
    "reward a confident wrong number."
)


def grade_correctness(
    question: str, gold: str, generated: str, *, model: str, secrets: Secrets | None = None
) -> bool:
    """LLM correctness: True if the answer matches the gold. One judge call, ``model``
    is recorded by the caller. Assumes the answer is not a refusal (check is_refusal first)."""
    secrets = secrets or Secrets()
    secrets.require("anthropic_api_key")
    from anthropic import Anthropic

    client = Anthropic(api_key=secrets.anthropic_api_key)
    user = f"QUESTION:\n{question}\n\nGOLD:\n{gold}\n\nANSWER:\n{generated}"
    msg = client.messages.create(
        model=model, max_tokens=5, temperature=0.0,
        system=_JUDGE_SYSTEM, messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip().lower()
    return text.startswith("correct")


def score_answer(
    question: str, gold: str, generated: str, *, judge_model: str, secrets: Secrets | None = None
) -> AnswerScore:
    """Refusal + numeric + LLM correctness for one (question, gold, answer)."""
    if is_refusal(generated):
        return AnswerScore(refused=True, numeric=None, llm_correct=None)
    numeric = numeric_match(gold, generated)
    llm = grade_correctness(question, gold, generated, model=judge_model, secrets=secrets)
    return AnswerScore(refused=False, numeric=numeric, llm_correct=llm)
