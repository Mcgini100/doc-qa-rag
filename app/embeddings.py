"""Embedding backends. All return L2-normalised float32 matrices (cosine = dot product)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


class Embedder:
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class LocalEmbedder(Embedder):
    """WordLlama (256-d). Weights ship inside the pip wheel, so it runs offline and in CI.
    Fast and small, but much weaker than a transformer embedding model."""

    name = "local-wordllama-256"
    dim = 256

    def __init__(self) -> None:
        import wordllama
        from wordllama import WordLlama

        # wordllama 0.4 bundles the tokenizer config in wordllama/tokenizers/ but looks for it
        # in a "tokenizer" folder, then tries Hugging Face. Copy it into the cache so loading
        # never needs the network.
        cache = WordLlama.get_file_path("tokenizer", None)
        fname = "l2_supercat_tokenizer_config.json"
        bundled = Path(wordllama.__file__).parent / "tokenizers" / fname
        if bundled.exists() and not (cache / fname).exists():
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copy(bundled, cache / fname)
        self._model = WordLlama.load(dim=256, disable_download=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(self._model.embed(texts, norm=True))


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-small (1536-d)."""

    dim = 1536

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.name = f"openai-{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):
            resp = self._client.embeddings.create(model=self._model, input=texts[i:i + 64])
            out.extend(d.embedding for d in resp.data)
        return _normalise(np.array(out))


def make_embedder(backend: str, openai_model: str = "text-embedding-3-small", api_key: str | None = None) -> Embedder:
    if backend == "openai":
        return OpenAIEmbedder(openai_model, api_key)
    if backend == "local":
        return LocalEmbedder()
    raise ValueError(f"unknown embedding backend: {backend}")
