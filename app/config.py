"""Runtime settings, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    docs_dir: Path = field(default_factory=lambda: Path(os.getenv("DOCS_DIR", ROOT / "data" / "docs")))
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("INDEX_DIR", ROOT / "data" / "index")))
    # "openai" (text-embedding-3-small) or "local" (WordLlama, bundled, runs offline).
    # "auto" picks openai when OPENAI_API_KEY is set, else local.
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "auto"))
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    openai_chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    use_llm: bool = field(default_factory=lambda: _bool("USE_LLM", True))
    chunk_tokens: int = field(default_factory=lambda: int(os.getenv("CHUNK_TOKENS", "600")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "75")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    rate_limit_ask: str = field(default_factory=lambda: os.getenv("RATE_LIMIT_ASK", "10/minute"))
    rate_limit_ingest: str = field(default_factory=lambda: os.getenv("RATE_LIMIT_INGEST", "3/minute"))
    admin_token: str | None = field(default_factory=lambda: os.getenv("ADMIN_TOKEN") or None)

    @property
    def openai_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY") or None

    @property
    def resolved_backend(self) -> str:
        if self.embedding_backend == "auto":
            return "openai" if self.openai_key else "local"
        return self.embedding_backend


def get_settings() -> Settings:
    return Settings()
