"""Typed configuration.

Two sources, kept separate on purpose:

* Secrets (API keys, DB URL) come from the environment / .env  -> ``Secrets``.
* Ablation knobs (chunking, embedding, retrieval, generation, eval) come from a
  YAML file under ``configs/`` -> ``Config``.

Keeping them apart means a config file can be committed and diffed (it is the
record of an ablation) while secrets never touch the repo.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Loaded from environment / .env. Never serialized into eval output."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    database_url: str = ""
    # V1+, optional.
    finnhub_api_key: str = ""

    def require(self, *names: str) -> None:
        """Raise a clear error naming any missing secret, before a network call."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                f"Missing required secret(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )


class ChunkingConfig(BaseModel):
    strategy: str = "token"  # "token" | "section_then_token"
    max_tokens: int = 512
    overlap_tokens: int = 64
    encoder: str = "cl100k_base"


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dim: int = 1536
    batch_size: int = 128


class RetrievalConfig(BaseModel):
    method: str = "dense"  # "dense" (V0) | "hybrid" (V1) | "lexical" (FTS-only)
    top_k: int = 5
    distance: str = "cosine"
    # Hybrid knobs (used only when method == "hybrid"). Defaults are inert for
    # dense, so v0.yaml behaviour is unchanged.
    candidates: int = 20  # depth per retriever before fusion
    k_rrf: int = 60       # RRF damping constant
    # Weighted RRF: dense contributions are scaled by dense_weight, lexical by
    # (1 - dense_weight). 0.5 = balanced (the original equal-weight RRF ordering);
    # 1.0 = dense-only; 0.0 = lexical-only. Ablation lever for the fusion sweep.
    dense_weight: float = 0.5
    # Cross-encoder reranker (V1.2). When on, fetch ``candidates`` from the base
    # retriever, rerank with rerank_model, return top_k. Inert when off.
    rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"


class GenerationConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"
    max_tokens: int = 1024
    temperature: float = 0.0


class EvalConfig(BaseModel):
    dataset: str = "PatronusAI/financebench"
    recall_ks: list[int] = Field(default_factory=lambda: [5, 10])
    seed: int = 13
    faithfulness: bool = False


class Config(BaseModel):
    """The full ablation config for one run. Validated from YAML."""

    corpus: str = "financebench"
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    eval: EvalConfig = EvalConfig()


def load_config(path: str | Path) -> Config:
    """Read and validate a YAML config file into a ``Config``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    return Config.model_validate(data)
