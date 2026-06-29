"""Embeddings (OpenAI text-embedding-3-small in V0).

The embedding model is an ablation variable (V2 compares 3-small / 3-large /
Voyage finance-2 / open BGE), so this wrapper takes the model + dimension from
config rather than hardcoding them. Output order is keyed off the response
``index`` field, not list position, so a batched response can't silently
misalign embeddings with their input text.
"""

from __future__ import annotations

import time

from sec_rag.config import EmbeddingConfig, Secrets

# Transient OpenAI failures worth retrying: per-minute rate limits (429) and
# network/timeout blips. A large corpus easily exceeds the 1M tokens/min limit,
# so embedding back-to-back batches 429s without this — exponential backoff
# self-paces the run under the TPM ceiling instead of crashing the ingest.
_MAX_RETRIES = 6


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

    def _create(self, batch: list[str]):
        """One embeddings call with bounded backoff on transient errors.

        A 429 with ``insufficient_quota`` is a BILLING failure (out of credits),
        not a transient rate limit — retrying just burns backoff cycles before the
        same failure and hides the real cause. Fail fast on it (same rule as the
        eval runner: infra/billing errors are fatal); retry only true transients.
        """
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        kwargs = {"model": self.model, "input": batch}
        # text-embedding-3-* support Matryoshka truncation via `dimensions`: this
        # is how 3-large is requested at 1536-d (same recall as 3072, but fits the
        # vector(1536) schema + Neon free tier). No-op for 3-small (native 1536).
        if self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dim
        delay = 2.0
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self.client.embeddings.create(**kwargs)
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if "insufficient_quota" in str(e):  # billing, not transient -> fail fast
                    raise
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, preserving input order. Empty input -> empty output."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self._create(batch)
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
