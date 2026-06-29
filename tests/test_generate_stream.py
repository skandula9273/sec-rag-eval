"""Streaming generation yields text deltas then one final GeneratedAnswer (mocked)."""

from sec_rag.config import GenerationConfig
from sec_rag.generate.answer import GeneratedAnswer, generate_answer_stream


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Usage:
    input_tokens = 10
    output_tokens = 5


class _Final:
    content = [_Block("3M net sales were $32,765M [1].")]
    usage = _Usage()


class _Stream:
    def __init__(self, deltas):
        self._deltas = deltas

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def text_stream(self):
        return iter(self._deltas)

    def get_final_message(self):
        return _Final()


class _Client:
    class messages:  # noqa: N801 - mimic the SDK's client.messages.stream(...)
        @staticmethod
        def stream(**kwargs):
            return _Stream(["3M ", "net sales ", "[1]."])


class _Secrets:
    anthropic_api_key = "x"

    def require(self, *names):
        pass


def test_stream_yields_deltas_then_final(monkeypatch):
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key=None: _Client())
    cfg = GenerationConfig(provider="anthropic", model="claude-haiku-4-5")

    out = list(generate_answer_stream("q", [], cfg, secrets=_Secrets()))
    deltas = [x for x in out if isinstance(x, str)]
    finals = [x for x in out if isinstance(x, GeneratedAnswer)]

    assert deltas == ["3M ", "net sales ", "[1]."]  # streamed in order
    assert len(finals) == 1  # exactly one final frame, last
    fa = finals[0]
    assert fa.cited_indices == [1]
    assert fa.tokens_in == 10 and fa.tokens_out == 5
    assert fa.cost_usd > 0 and fa.cost_is_estimate is False
    assert "32,765" in fa.text
