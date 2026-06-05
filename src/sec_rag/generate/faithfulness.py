"""Faithfulness scoring (V0 lightweight judge).

Why not RAGAS: the ``ragas`` package (design-doc's named tool) is built for
LangChain 0.x and does not import under the current LangChain 1.x ecosystem —
pinning it back breaks openai/anthropic and violates the reproducibility rule.
See the 2026-06-04 design-doc amendment.

What this does instead: the same *definition* RAGAS uses for faithfulness —
the fraction of the answer's factual claims that are supported by the retrieved
context — computed with one judge call (Claude Haiku, temperature 0 for
reproducibility). The judge decomposes the answer into atomic claims and marks
each as supported / not supported by the sources; the score is supported/total.

Score is in [0, 1]. An answer with no factual claims (e.g. "I can't find this
in the sources") scores 1.0 — it asserts nothing unsupported, which is exactly
the grounded refusal behaviour we want to reward, not penalise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sec_rag.config import GenerationConfig, Secrets
from sec_rag.retrieve.dense import RetrievedChunk

_JUDGE_SYSTEM = (
    "You are a strict grading assistant. You are given an ANSWER and the SOURCES "
    "it was supposed to be based on. Break the ANSWER into atomic factual claims. "
    "For each claim, decide if it is directly supported by the SOURCES. Ignore "
    "hedging, refusals, or statements that assert no fact (e.g. 'I cannot find "
    "this'). Respond with ONLY a JSON object of the form "
    '{"claims": <int>, "supported": <int>} where claims is the number of atomic '
    "factual claims and supported is how many are backed by the SOURCES. If the "
    "answer makes no factual claims, return {\"claims\": 0, \"supported\": 0}."
)

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class FaithfulnessResult:
    score: float  # in [0, 1]
    n_claims: int
    n_supported: int


def _sources_block(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {c.content}" for i, c in enumerate(chunks, start=1))


def score_faithfulness(
    answer: str,
    chunks: list[RetrievedChunk],
    cfg: GenerationConfig,
    secrets: Secrets | None = None,
) -> FaithfulnessResult:
    """Grade how much of ``answer`` is grounded in ``chunks``. One judge call."""
    secrets = secrets or Secrets()
    secrets.require("anthropic_api_key")
    from anthropic import Anthropic

    client = Anthropic(api_key=secrets.anthropic_api_key)
    user = f"SOURCES:\n{_sources_block(chunks)}\n\nANSWER:\n{answer}"
    msg = client.messages.create(
        model=cfg.model,
        max_tokens=200,
        temperature=0.0,  # deterministic judge for reproducible eval
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    # Parse the JSON the judge was asked to return; fall back safely if it didn't.
    m = _JSON_RE.search(text)
    if not m:
        return FaithfulnessResult(score=0.0, n_claims=0, n_supported=0)
    try:
        data = json.loads(m.group(0))
        claims = int(data.get("claims", 0))
        supported = int(data.get("supported", 0))
    except (ValueError, TypeError):
        return FaithfulnessResult(score=0.0, n_claims=0, n_supported=0)

    supported = max(0, min(supported, claims))  # guard against a bad judge count
    # No factual claims -> nothing unsupported -> perfectly faithful (1.0).
    score = 1.0 if claims == 0 else supported / claims
    return FaithfulnessResult(score=round(score, 4), n_claims=claims, n_supported=supported)
