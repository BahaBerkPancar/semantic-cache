"""Semantic cache: embeds prompts, finds the nearest cached prompt, applies guards."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .guards import Guard, Scope, Scoper, run_guards


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised vectors."""
        ...


@dataclass(frozen=True)
class Context:
    namespace: str = "default"
    user_id: str | None = None


@dataclass
class Entry:
    prompt: str
    answer: str
    vector: np.ndarray
    scope: Scope
    tags: frozenset[str] = frozenset()
    expires_at: float | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class Lookup:
    hit: bool
    answer: str | None = None
    similarity: float | None = None
    matched_prompt: str | None = None
    reason: str = ""


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one vector against each row of a matrix."""
    q = normalize(query)
    m = normalize(matrix)
    return m @ q


class Backend(Protocol):
    """Storage interface. M2 adds a Redis implementation behind the same methods."""

    def add(self, entry: Entry) -> None: ...

    def candidates(
        self, vector: np.ndarray, scope: Scope, now: float, k: int
    ) -> list[tuple[Entry, float]]: ...

    def invalidate_tag(self, tag: str) -> int: ...

    def __len__(self) -> int: ...


class InMemoryBackend:
    """Brute-force cosine search over a list; fine for thousands of entries. Not thread-safe."""

    def __init__(self) -> None:
        self._entries: list[Entry] = []

    def add(self, entry: Entry) -> None:
        self._entries.append(entry)

    def candidates(
        self, vector: np.ndarray, scope: Scope, now: float, k: int
    ) -> list[tuple[Entry, float]]:
        live = [
            e
            for e in self._entries
            if e.scope == scope and (e.expires_at is None or e.expires_at > now)
        ]
        if not live:
            return []
        sims = cosine_similarity(vector, np.stack([e.vector for e in live]))
        order = np.argsort(-sims, kind="stable")[:k]
        return [(live[i], float(sims[i])) for i in order]

    def invalidate_tag(self, tag: str) -> int:
        before = len(self._entries)
        self._entries = [e for e in self._entries if tag not in e.tags]
        return before - len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


class SemanticCache:
    def __init__(
        self,
        backend: Backend,
        embedder: Embedder,
        threshold: float,
        guards: Sequence[Guard] = (),
        scoper: Scoper | None = None,
        *,
        clock: Callable[[], float] = time.time,
        top_k: int = 5,
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.threshold = threshold
        self.guards = list(guards)
        self.scoper = scoper or Scoper()
        self.clock = clock
        self.top_k = top_k

    def _vector(self, prompt: str) -> np.ndarray:
        return normalize(self.embedder.embed([prompt]))[0]

    def get(self, prompt: str, ctx: Context = Context()) -> Lookup:
        scope = self.scoper.scope(prompt, ctx.namespace, ctx.user_id)
        if scope is None:
            return Lookup(False, reason="never_cache: personal prompt without user id")
        cands = self.backend.candidates(self._vector(prompt), scope, self.clock(), self.top_k)
        if not cands:
            return Lookup(False, reason="no_candidates")
        denial = ""
        for entry, sim in cands:
            if sim < self.threshold:
                break  # sorted descending: nothing further qualifies
            verdict = run_guards(prompt, entry.prompt, self.guards)
            if verdict.allow:
                return Lookup(True, entry.answer, sim, entry.prompt)
            denial = denial or verdict.reason
        return Lookup(False, similarity=cands[0][1], reason=denial or "below_threshold")

    def put(
        self,
        prompt: str,
        answer: str,
        ctx: Context = Context(),
        tags: Sequence[str] = (),
        ttl: float | None = None,
    ) -> bool:
        """Store an answer. Returns False when the prompt must never be cached."""
        scope = self.scoper.scope(prompt, ctx.namespace, ctx.user_id)
        if scope is None:
            return False
        expires = None if ttl is None else self.clock() + ttl
        self.backend.add(
            Entry(prompt, answer, self._vector(prompt), scope, frozenset(tags), expires)
        )
        return True

    def invalidate(self, tag: str) -> int:
        return self.backend.invalidate_tag(tag)
