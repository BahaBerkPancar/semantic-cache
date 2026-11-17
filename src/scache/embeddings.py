"""Embedding providers and the on-disk embedding cache (data/embeddings.npz)."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .cache import normalize

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_PREFIX = "search_query: "
OLLAMA_URL = "http://127.0.0.1:11434"


def text_hash(text: str, model: str, prefix: str) -> str:
    return hashlib.sha256(f"{model}\x00{prefix}\x00{text}".encode()).hexdigest()[:16]


class OllamaEmbedder:
    """Local Ollama embeddings. Only loopback hosts are accepted."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        url: str = OLLAMA_URL,
        prefix: str = DEFAULT_PREFIX,
        batch: int = 32,
    ) -> None:
        if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Ollama URL must be a loopback address")
        self.model, self.url, self.prefix, self.batch = model, url.rstrip("/"), prefix, batch

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = [self.prefix + t for t in texts[i : i + self.batch]]
            body = json.dumps({"model": self.model, "input": chunk}).encode()
            req = urllib.request.Request(
                self.url + "/api/embed", body, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                out.extend(json.load(resp)["embeddings"])
        return normalize(np.array(out, dtype=np.float32))


class FakeEmbedder:
    """Deterministic character-trigram hashing embedder for tests (no model, no network)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            t = f"  {text.lower()}  "
            for i in range(len(t) - 2):
                h = int.from_bytes(hashlib.md5(t[i : i + 3].encode()).digest()[:4], "big")
                out[row, h % self.dim] += 1.0
        return normalize(out)


@dataclass
class EmbeddingStore:
    model: str
    prefix: str
    texts: list[str]
    hashes: list[str]
    vectors: np.ndarray

    @property
    def dims(self) -> int:
        return int(self.vectors.shape[1]) if len(self.vectors) else 0

    def index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.texts)}

    def stale(self, texts: Iterable[str]) -> list[str]:
        """Texts that are missing from the store or whose hash no longer matches."""
        idx = self.index()
        bad = []
        for t in texts:
            i = idx.get(t)
            if i is None or self.hashes[i] != text_hash(t, self.model, self.prefix):
                bad.append(t)
        return bad

    def save(self, path: Path) -> None:
        order = sorted(range(len(self.texts)), key=lambda i: self.texts[i])
        np.savez_compressed(
            path,
            texts=np.array([self.texts[i] for i in order]),
            hashes=np.array([self.hashes[i] for i in order]),
            vectors=self.vectors[order].astype(np.float32),
            model=np.array(self.model),
            prefix=np.array(self.prefix),
            dims=np.array(self.dims),
        )

    @classmethod
    def load(cls, path: Path) -> EmbeddingStore:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                model=str(z["model"]),
                prefix=str(z["prefix"]),
                texts=[str(t) for t in z["texts"]],
                hashes=[str(h) for h in z["hashes"]],
                vectors=z["vectors"],
            )


class StaleEmbeddingsError(RuntimeError):
    pass


class StoreEmbedder:
    """Serves vectors from an EmbeddingStore; refuses texts that are missing or stale."""

    def __init__(self, store: EmbeddingStore) -> None:
        self.store = store
        self._idx = store.index()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        bad = self.store.stale(texts)
        if bad:
            raise StaleEmbeddingsError(
                f"{len(bad)} text(s) missing or stale in the embedding cache "
                f"(first: {bad[0]!r}); run `scache embed`"
            )
        return self.store.vectors[[self._idx[t] for t in texts]]


def build_store(
    texts: Sequence[str],
    embedder: OllamaEmbedder,
    existing: EmbeddingStore | None = None,
) -> tuple[EmbeddingStore, int]:
    """Embed only what is missing or stale. Returns (store, number of texts embedded)."""
    model, prefix = embedder.model, embedder.prefix
    wanted = sorted(set(texts))
    keep: dict[str, tuple[str, np.ndarray]] = {}
    if existing is not None and existing.model == model and existing.prefix == prefix:
        idx = existing.index()
        for t in wanted:
            i = idx.get(t)
            if i is not None and existing.hashes[i] == text_hash(t, model, prefix):
                keep[t] = (existing.hashes[i], existing.vectors[i])
    todo = [t for t in wanted if t not in keep]
    if todo:
        vecs = embedder.embed(todo)
        for t, v in zip(todo, vecs, strict=True):
            keep[t] = (text_hash(t, model, prefix), v)
    store = EmbeddingStore(
        model=model,
        prefix=prefix,
        texts=wanted,
        hashes=[keep[t][0] for t in wanted],
        vectors=np.stack([keep[t][1] for t in wanted]).astype(np.float32),
    )
    return store, len(todo)
