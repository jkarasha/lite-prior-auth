"""In-memory RAG retrieval over the guideline corpus.

Deliberately simple: Voyage embeddings + cosine similarity, no vector DB.
Swapping this for a real vector store is a one-interface change — the agent
and tools only depend on `Retriever.search`.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import voyageai

from .config import Config


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class Retriever:
    """Loads the corpus, embeds it once, answers similarity queries."""

    def __init__(self, config: Config, chunks: list[Chunk], embeddings: list[list[float]]):
        self._config = config
        self._chunks = chunks
        self._embeddings = embeddings
        self._client = voyageai.Client()

    @classmethod
    def from_corpus(cls, config: Config) -> "Retriever":
        chunks = _load_chunks(config.corpus_dir)
        if not chunks:
            raise RuntimeError(f"No documents found in {config.corpus_dir}")
        client = voyageai.Client()
        result = client.embed(
            [c.text for c in chunks],
            model=config.embed_model,
            input_type="document",
        )
        return cls(config, chunks, result.embeddings)

    def search(self, query: str) -> list[Hit]:
        """Return the top-k chunks most similar to the query."""
        if not query.strip():
            return []
        q_vec = self._client.embed(
            [query], model=self._config.embed_model, input_type="query"
        ).embeddings[0]
        scored = [Hit(c, _cosine(q_vec, e)) for c, e in zip(self._chunks, self._embeddings)]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[: self._config.top_k]


def _load_chunks(corpus_dir: str) -> list[Chunk]:
    """One chunk per file for the demo. TODO: split long docs into ~500-token chunks."""
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            chunks.append(Chunk(doc_id=os.path.basename(path), text=text))
    return chunks
