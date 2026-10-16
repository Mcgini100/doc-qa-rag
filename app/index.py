"""Vector index (FAISS, exact inner product) plus a BM25 keyword index over the same chunks.

Retrieval fuses both rankings with Reciprocal Rank Fusion (RRF). Statutes are full of
exact terms ("data controller", "section 13", "level 10 fine") that keyword search
catches and small embedding models miss.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder
from app.ingest import Chunk

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an and are as at be by for from has have he her his in is it its of on or that the
their them they this to was were which who will with shall may any such what whom whose
how when where why do does did i my me we our you your can could would should if not no
under about into than then there these those also act""".split())


def bm25_tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float          # fused RRF score (higher is better)
    dense: float          # cosine similarity of the question to this chunk
    bm25: float           # raw BM25 score
    rank: int


class DocIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        self.chunks = chunks
        self.embedder_name = embedder_name
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
        self._faiss.add(self.vectors)
        self._bm25 = BM25Okapi([bm25_tokens(self._searchable(c)) for c in chunks])

    @staticmethod
    def _searchable(c: Chunk) -> str:
        return " ".join(c.sections) + "\n" + c.text

    # ------------------------------------------------------------------ build / persist
    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "DocIndex":
        vectors = embedder.embed([cls._searchable(c) for c in chunks])
        return cls(chunks, vectors, embedder.name)

    def add(self, chunks: list[Chunk], embedder: Embedder) -> None:
        new = DocIndex.build(chunks, embedder)
        self.__init__(self.chunks + chunks, np.vstack([self.vectors, new.vectors]), self.embedder_name)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder_name, "chunks": [c.to_dict() for c in self.chunks]}
        (directory / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, directory: Path) -> "DocIndex":
        meta = json.loads((directory / "chunks.json").read_text())
        chunks = [Chunk(**c) for c in meta["chunks"]]
        return cls(chunks, np.load(directory / "vectors.npy"), meta["embedder"])

    # ------------------------------------------------------------------ search
    def search(self, query: str, embedder: Embedder, k: int = 5, mode: str = "hybrid",
               rrf_k: int = 60) -> list[Hit]:
        n = len(self.chunks)
        qv = embedder.embed([query])
        dense_scores = (self.vectors @ qv[0]).astype(float)
        bm25_scores = np.asarray(self._bm25.get_scores(bm25_tokens(query)), dtype=float)
        dense_rank = np.argsort(-dense_scores)
        bm25_rank = np.argsort(-bm25_scores)
        if mode == "dense":
            order = list(dense_rank)
            fused = {i: dense_scores[i] for i in order}
        elif mode == "bm25":
            order = list(bm25_rank)
            fused = {i: bm25_scores[i] for i in order}
        elif mode == "hybrid":
            fused = np.zeros(n)
            for r, i in enumerate(dense_rank):
                fused[i] += 1.0 / (rrf_k + r + 1)
            for r, i in enumerate(bm25_rank):
                if bm25_scores[i] > 0:  # chunks sharing no query term get no keyword credit
                    fused[i] += 1.0 / (rrf_k + r + 1)
            order = list(np.argsort(-fused))
            fused = {i: fused[i] for i in order}
        else:
            raise ValueError(f"unknown mode {mode}")
        return [Hit(self.chunks[i], float(fused[i]), float(dense_scores[i]), float(bm25_scores[i]), r + 1)
                for r, i in enumerate(order[:k])]
