"""Answer generation with Claude Haiku (V0).

Builds a grounded prompt from the retrieved chunks, numbered so the model can
cite them as ``[1]``, ``[2]``. The cited markers are parsed back out so the UI
can distinguish chunks the answer actually used from chunks that were merely
retrieved (the cited-vs-retrieved distinction in the design doc).

Pricing note: cost_usd is computed from token usage times the rates in
``PRICING``. These rates are NOT confirmed against current Anthropic pricing —
set them from the live pricing page before trusting cost numbers in the writeup.
Until then cost is directional, and that is stated wherever it is shown.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from sec_rag.config import GenerationConfig, Secrets
from sec_rag.retrieve.dense import RetrievedChunk

# USD per token (NOT per million). Confirmed 2026-06-04 against Anthropic's
# pricing page: Claude Haiku 4.5 = $1.00 / 1M input tokens, $5.00 / 1M output
# tokens -> divide by 1e6 for the per-token rate _cost() multiplies by.
PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 1.00 / 1_000_000, "output": 5.00 / 1_000_000},
}

_SYSTEM = (
    "You answer questions about SEC filings using only the numbered sources "
    "provided. Cite every claim with the matching source number in square "
    "brackets, e.g. [1]. If the sources do not contain the answer, say so "
    "explicitly instead of guessing. Do not use outside knowledge."
)

_CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class GeneratedAnswer:
    text: str
    cited_indices: list[int]  # 1-based source numbers referenced in the answer
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cost_is_estimate: bool  # True if the model has no confirmed rate in PRICING
    model: str


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        head = f"[{i}] {c.doc_name}"
        if c.section:
            head += f" — {c.section}"
        if c.page is not None:
            head += f" (p.{c.page})"
        blocks.append(f"{head}\n{c.content}")
    sources = "\n\n".join(blocks)
    return f"Question: {question}\n\nSources:\n{sources}\n\nAnswer with citations:"


def _cost(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return tokens_in * rates["input"] + tokens_out * rates["output"]


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    cfg: GenerationConfig,
    secrets: Secrets | None = None,
) -> GeneratedAnswer:
    if cfg.provider != "anthropic":
        raise NotImplementedError(f"V0 generates via Anthropic, got {cfg.provider!r}")
    secrets = secrets or Secrets()
    secrets.require("anthropic_api_key")
    from anthropic import Anthropic

    client = Anthropic(api_key=secrets.anthropic_api_key)
    msg = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=_SYSTEM,
        messages=[{"role": "user", "content": _build_prompt(question, chunks)}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    cited = sorted({int(m) for m in _CITE_RE.findall(text)})
    tokens_in = msg.usage.input_tokens
    tokens_out = msg.usage.output_tokens
    return _assemble(cfg.model, text, tokens_in, tokens_out)


def _assemble(model: str, text: str, tokens_in: int, tokens_out: int) -> GeneratedAnswer:
    return GeneratedAnswer(
        text=text,
        cited_indices=sorted({int(m) for m in _CITE_RE.findall(text)}),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=_cost(model, tokens_in, tokens_out),
        # Honest by construction: estimate iff this model has no confirmed rate.
        cost_is_estimate=model not in PRICING,
        model=model,
    )


def generate_answer_stream(
    question: str,
    chunks: list[RetrievedChunk],
    cfg: GenerationConfig,
    secrets: Secrets | None = None,
) -> Iterator[str | GeneratedAnswer]:
    """Stream the answer for low time-to-first-token.

    Yields text deltas (str) as the model produces them, then ONE final
    ``GeneratedAnswer`` (citations + usage + cost, built from the final message).
    Same prompt/system/model as ``generate_answer`` — only the transport differs,
    so the streamed answer is the answer ``/query`` would return.
    """
    if cfg.provider != "anthropic":
        raise NotImplementedError(f"V0 generates via Anthropic, got {cfg.provider!r}")
    secrets = secrets or Secrets()
    secrets.require("anthropic_api_key")
    from anthropic import Anthropic

    client = Anthropic(api_key=secrets.anthropic_api_key)
    with client.messages.stream(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=_SYSTEM,
        messages=[{"role": "user", "content": _build_prompt(question, chunks)}],
    ) as stream:
        for delta in stream.text_stream:
            yield delta
        final = stream.get_final_message()
    text = "".join(b.text for b in final.content if getattr(b, "type", None) == "text")
    yield _assemble(cfg.model, text, final.usage.input_tokens, final.usage.output_tokens)
