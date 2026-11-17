"""HTTP proxy: POST /v1/complete looks up the semantic cache, calls a generator on a miss."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .cache import ID_PATTERN, Context, Embedder, InMemoryBackend, SemanticCache
from .embeddings import FakeEmbedder, OllamaEmbedder
from .generators import FakeGenerator, Generator, GeneratorError, OllamaGenerator
from .guards import default_guards
from .policy import NeverCachePolicy


@dataclass
class Settings:
    threshold: float = 0.79  # the guarded threshold chosen by the M1 benchmark
    redis_url: str | None = None  # None: in-memory backend
    redis_prefix: str = "scache"
    embedder: Literal["fake", "ollama"] = "fake"
    generator: Literal["fake", "ollama"] = "fake"
    default_ttl: float | None = None

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ.get
        ttl = e("SCACHE_DEFAULT_TTL")
        return cls(
            threshold=float(e("SCACHE_THRESHOLD", cls.threshold)),
            redis_url=e("SCACHE_REDIS_URL") or None,
            redis_prefix=e("SCACHE_REDIS_PREFIX", cls.redis_prefix),
            embedder=e("SCACHE_EMBEDDER", "fake"),  # type: ignore[arg-type]
            generator=e("SCACHE_GENERATOR", "fake"),  # type: ignore[arg-type]
            default_ttl=float(ttl) if ttl else None,
        )


log = logging.getLogger("scache")
IdStr = Annotated[str, Field(pattern=ID_PATTERN)]


class CompleteRequest(BaseModel):
    prompt: str = Field(min_length=1)
    namespace: IdStr = "default"
    user_id: IdStr | None = None
    tags: list[IdStr] = []
    ttl: float | None = Field(default=None, gt=0)
    no_cache: bool = False


class CompleteResponse(BaseModel):
    id: str | None
    answer: str
    cache: Literal["hit", "miss", "bypass"]
    similarity: float | None
    matched_prompt: str | None
    reasons: list[str]
    latency_ms: float


class InvalidateRequest(BaseModel):
    namespace: IdStr  # required: invalidation is always scoped to a namespace
    tag: IdStr | None = None  # omitted: the whole namespace


class FeedbackRequest(BaseModel):
    id: str
    namespace: IdStr  # must match the entry's namespace
    user_id: IdStr | None = None  # required to evict a personal (per-user) entry
    wrong: bool = True


@dataclass
class Stats:
    hits: int = 0
    misses: int = 0
    bypasses: int = 0
    false_hit_reports: int = 0
    evictions: int = 0
    generator_errors: int = 0

    def as_dict(self) -> dict[str, float | int]:
        served = self.hits + self.misses + self.bypasses
        lookups = self.hits + self.misses
        return {
            **self.__dict__,
            "requests": served,
            "hit_rate": self.hits / lookups if lookups else 0.0,
            "reported_false_hit_rate": self.false_hit_reports / self.hits if self.hits else 0.0,
        }


@dataclass
class State:
    cache: SemanticCache
    generator: Generator
    policy: NeverCachePolicy
    settings: Settings
    stats: Stats = field(default_factory=Stats)
    lock: threading.Lock = field(default_factory=threading.Lock)


def build_state(
    settings: Settings,
    embedder: Embedder | None = None,
    generator: Generator | None = None,
    backend=None,
    policy: NeverCachePolicy | None = None,
) -> State:
    embedder = embedder or (OllamaEmbedder() if settings.embedder == "ollama" else FakeEmbedder())
    if backend is None:
        if settings.redis_url:
            from .redis_backend import RedisBackend

            dims = int(embedder.embed(["dimension probe"]).shape[1])
            backend = RedisBackend(settings.redis_url, dims, settings.redis_prefix)
        else:
            backend = InMemoryBackend()
    generator = generator or (
        OllamaGenerator() if settings.generator == "ollama" else FakeGenerator()
    )
    cache = SemanticCache(backend, embedder, settings.threshold, guards=default_guards())
    return State(cache, generator, policy or NeverCachePolicy(), settings)


def create_app(state: State | None = None) -> FastAPI:
    app = FastAPI(title="semantic-cache")
    holder: dict[str, State] = {}
    if state is not None:
        holder["s"] = state

    def st() -> State:  # lazy, so importing the module never touches Redis or Ollama
        if "s" not in holder:
            holder["s"] = build_state(Settings.from_env())
        return holder["s"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/complete", response_model=CompleteResponse)
    def complete(req: CompleteRequest) -> CompleteResponse:
        s = st()
        t0 = time.perf_counter()
        ctx = Context(req.namespace, req.user_id)
        ttl = req.ttl if req.ttl is not None else s.settings.default_ttl

        def done(cache: str, answer: str, entry_id=None, sim=None, matched=None, reasons=()):
            return CompleteResponse(
                id=entry_id,
                answer=answer,
                cache=cache,
                similarity=sim,
                matched_prompt=matched,
                reasons=list(reasons),
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )

        def generate() -> tuple[str, bool]:
            try:
                return s.generator.generate(req.prompt), False
            except GeneratorError as exc:
                s.stats.generator_errors += 1
                log.warning("generator failed: %s", exc)
                raise HTTPException(502, "generator failed") from exc

        bypass = s.policy.check_prompt(req.prompt, req.user_id, req.no_cache)
        if bypass:
            answer, _ = generate()
            s.stats.bypasses += 1
            return done("bypass", answer, reasons=[bypass])

        with s.lock:
            look = s.cache.get(req.prompt, ctx)
        if look.hit:
            s.stats.hits += 1
            return done(
                "hit", look.answer or "", look.entry_id, look.similarity, look.matched_prompt
            )

        answer, _ = generate()
        reasons = [look.reason] if look.reason else []
        entry_id = None
        skip = s.policy.check_answer(answer)
        if skip:
            reasons.append(skip)
        else:
            try:
                with s.lock:
                    entry = s.cache.store(req.prompt, answer, ctx, req.tags, ttl)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            entry_id = entry.id if entry else None
        s.stats.misses += 1
        return done("miss", answer, entry_id, look.similarity, reasons=reasons)

    @app.post("/v1/invalidate")
    def invalidate(req: InvalidateRequest) -> dict[str, int]:
        s = st()
        with s.lock:
            n = (
                s.cache.invalidate_namespace(req.namespace)
                if req.tag is None
                else s.cache.invalidate(req.tag, req.namespace)
            )
        return {"invalidated": n}

    @app.post("/v1/feedback")
    def feedback(req: FeedbackRequest) -> dict[str, bool]:
        s = st()
        if not req.wrong:
            return {"evicted": False}
        with s.lock:
            evicted = s.cache.evict(req.id, Context(req.namespace, req.user_id))
        s.stats.false_hit_reports += 1
        if evicted:
            s.stats.evictions += 1
        return {"evicted": evicted}

    @app.get("/v1/stats")
    def stats() -> dict[str, float | int]:
        s = st()
        with s.lock:
            size = len(s.cache.backend)
        return {**s.stats.as_dict(), "entries": size}

    return app


app = create_app()
