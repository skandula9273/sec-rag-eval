"""Cost computation — confirmed Haiku 4.5 rates.

Rates confirmed 2026-06-04: $1.00 / 1M input tokens, $5.00 / 1M output tokens.
_cost is pure (no API), so this runs without a key.
"""

from sec_rag.generate.answer import PRICING, _cost


def test_haiku_rate_is_per_token_not_per_million():
    r = PRICING["claude-haiku-4-5"]
    assert r["input"] == 1.00 / 1_000_000
    assert r["output"] == 5.00 / 1_000_000


def test_cost_known_model():
    # 1000 in * $1/M + 200 out * $5/M = 0.001 + 0.001 = 0.002
    assert abs(_cost("claude-haiku-4-5", 1000, 200) - 0.002) < 1e-12


def test_cost_zero_tokens():
    assert _cost("claude-haiku-4-5", 0, 0) == 0.0


def test_cost_unknown_model_is_zero():
    # An unpriced model returns 0 and (elsewhere) is flagged cost_is_estimate=True.
    assert _cost("some-future-model", 1000, 200) == 0.0
    assert "some-future-model" not in PRICING
