"""Semantic cache: embeds prompts, finds the nearest cached prompt, applies guards."""

from __future__ import annotations

import re
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


ID_RE = re.compile(r"[A-Za-z0-9_.:@-]{1,128}")
ID_PATTERN = "^" + ID_RE.pattern + "$"  # anchored: pydantic searches, it does not fullmatch


def validate_id(value: str, what: str) -> str:
    """Namespaces, user ids and tags use a safe charset: no separators, spaces or wildcards."""
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ValueError(f"{what} must match [A-Za-z0-9_.:@-]{{1,128}}")
    return value


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
    ttl: float | None = None  # relative seconds as requested; backends may use it for GC
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class Lookup:
    hit: bool
    answer: str | None = None
    similarity: float | None = None
    matched_prompt: str | None = None
    reason: str = ""
    entry_id: str | None = None


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
    """Storage interface, implemented by InMemoryBackend and RedisBackend (scache.redis_backend)."""

    def add(self, entry: Entry) -> None: ...

    def candidates(
        self, vector: np.ndarray, scope: Scope, now: float, k: int
    ) -> list[tuple[Entry, float]]: ...

    def invalidate_tag(self, tag: str, namespace: str | None = None) -> int: ...

    def invalidate_namespace(self, namespace: str) -> int: ...

    def scope_of(self, entry_id: str) -> Scope | None: ...

    def delete(self, entry_id: str) -> bool: ...

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

    def invalidate_tag(self, tag: str, namespace: str | None = None) -> int:
        before = len(self._entries)
        self._entries = [
            e
            for e in self._entries
            if not (tag in e.tags and (namespace is None or e.scope[0] == namespace))
        ]
        return before - len(self._entries)

    def scope_of(self, entry_id: str) -> Scope | None:
        return next((e.scope for e in self._entries if e.id == entry_id), None)

    def invalidate_namespace(self, namespace: str) -> int:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.scope[0] != namespace]
        return before - len(self._entries)

    def delete(self, entry_id: str) -> bool:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.id != entry_id]
        return len(self._entries) < before

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
                return Lookup(True, entry.answer, sim, entry.prompt, entry_id=entry.id)
            denial = denial or verdict.reason
        return Lookup(False, similarity=cands[0][1], reason=denial or "below_threshold")

    def store(
        self,
        prompt: str,
        answer: str,
        ctx: Context = Context(),
        tags: Sequence[str] = (),
        ttl: float | None = None,
    ) -> Entry | None:
        """Store an answer. Returns the entry, or None when the prompt must never be cached."""
        scope = self.scoper.scope(prompt, ctx.namespace, ctx.user_id)
        if scope is None:
            return None
        for t in tags:
            validate_id(t, "tag")
        expires = None if ttl is None else self.clock() + ttl
        entry = Entry(prompt, answer, self._vector(prompt), scope, frozenset(tags), expires, ttl)
        self.backend.add(entry)
        return entry

    def put(
        self,
        prompt: str,
        answer: str,
        ctx: Context = Context(),
        tags: Sequence[str] = (),
        ttl: float | None = None,
    ) -> bool:
        """Like store(), returning whether the answer was stored."""
        return self.store(prompt, answer, ctx, tags, ttl) is not None

    def invalidate(self, tag: str, namespace: str | None = None) -> int:
        return self.backend.invalidate_tag(tag, namespace)

    def invalidate_namespace(self, namespace: str) -> int:
        return self.backend.invalidate_namespace(namespace)

    def evict(self, entry_id: str, ctx: Context) -> bool:
        """Delete an entry only if it lives in the caller's scope (namespace, shared or own)."""
        scope = self.backend.scope_of(entry_id)
        if scope is None or scope[0] != ctx.namespace:
            return False
        if scope[1] is not None and scope[1] != ctx.user_id:
            return False
        return self.backend.delete(entry_id)
