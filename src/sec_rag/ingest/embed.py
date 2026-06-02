"""Embeddings (OpenAI text-embedding-3-small in V0).

The embedding model is an ablation variable (V2 compares 3-small / 3-large /
Voyage finance-2 / open BGE), so this wrapper takes the model + dimension from
config rather than hardcoding them. Output order is keyed off the response
``index`` field, not list position, so a batched response can't silently
misalign embeddings with their input text.
"""

from __future__ import annotations

from sec_rag.config import EmbeddingConfig, Secrets


class Embedder:
    def __init__(self, cfg: EmbeddingConfig, secrets: Secrets | None = None):
        if cfg.provider != "openai":
            raise NotImplementedError(f"V0 embeds via OpenAI, got provider={cfg.provider!r}")
        secrets = secrets or Secrets()
        secrets.require("openai_api_key")
        from openai import OpenAI

        self.client = OpenAI(api_key=secrets.openai_api_key)
        self.model = cfg.model
        self.dim = cfg.dim
        self.batch_size = cfg.batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, preserving input order. Empty input -> empty output."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)
            for d in ordered:
                if len(d.embedding) != self.dim:
                    raise ValueError(
                        f"embedding dim {len(d.embedding)} != configured {self.dim}; "
                        "update embedding.dim and db/schema.sql vector(N) together."
                    )
                vectors.append(d.embedding)
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
