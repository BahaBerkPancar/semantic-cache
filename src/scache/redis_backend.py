"""Redis backend: one HASH per entry, a RediSearch HNSW vector index, scope filter in the query.

Raw redis-py with FT.SEARCH KNN (no RedisVL, see DECISIONS.md). The namespace/user filter is a
pre-filter inside the KNN query, so another scope's entries can neither leak nor crowd out the
top-k. Expiry is enforced twice: the key gets EXPIRE (garbage collection) and the query filters on
`expires_at` against the cache clock (exact, and testable with a fake clock).
"""

from __future__ import annotations

import re
import time

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from .cache import Entry
from .guards import Scope

SEP = "|"  # TAG separator; outside the allowed id charset, so values never split
NO_USER = "none"  # stored for "no user"; real ids are stored as "u:<id>", so no collision
NEVER = 9e15  # expires_at for entries without a TTL (a finite number RediSearch can index)
_ESCAPE = re.compile(r"([^0-9A-Za-z_])")


def _esc(value: str) -> str:
    """Escape a value for a TAG query (every non-word character, including space)."""
    return _ESCAPE.sub(r"\\\1", value)


def _tag_value(user: str | None) -> str:
    return NO_USER if user is None else f"u:{user}"


class RedisBackend:
    def __init__(
        self,
        url: str,
        dims: int,
        prefix: str = "scache",
        *,
        ef_runtime: int = 100,
        client: redis.Redis | None = None,
    ) -> None:
        self.r = client or redis.Redis.from_url(url, decode_responses=False)
        self.dims, self.prefix, self.ef_runtime = dims, prefix, ef_runtime
        self.index = f"{prefix}:idx"
        self._key_prefix = f"{prefix}:e:"
        self._ensure_index()

    # ---- index -------------------------------------------------------------------------------

    def _ensure_index(self) -> None:
        try:
            info = self.r.ft(self.index).info()
        except ResponseError:
            self._create_index()
            return
        for attr in info.get("attributes", []):
            flat = [x.decode() if isinstance(x, bytes) else x for x in attr]
            if "vec" in flat and "DIM" in flat:
                existing = int(flat[flat.index("DIM") + 1])
                if existing != self.dims:
                    raise ValueError(
                        f"index {self.index} has {existing} dims, embedder gives {self.dims}; "
                        "use another prefix or drop the index"
                    )
            if "ns" in flat and "CASESENSITIVE" not in flat:
                raise ValueError(
                    f"index {self.index} was created by an older version (case-insensitive tags); "
                    f"drop it (FT.DROPINDEX {self.index} DD) or use another prefix"
                )

    def _create_index(self) -> None:
        schema = (
            TextField("prompt"),
            TextField("answer"),
            TagField("ns", separator=SEP, case_sensitive=True),
            TagField("user", separator=SEP, case_sensitive=True),
            TagField("tags", separator=SEP, case_sensitive=True),
            NumericField("created_at"),
            NumericField("expires_at"),
            VectorField(
                "vec",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": self.dims, "DISTANCE_METRIC": "COSINE"},
            ),
        )
        self.r.ft(self.index).create_index(
            schema,
            definition=IndexDefinition(prefix=[self._key_prefix], index_type=IndexType.HASH),
        )

    def drop(self) -> None:
        """Drop the index and every entry under this prefix (used by tests)."""
        try:
            self.r.ft(self.index).dropindex(delete_documents=True)
        except ResponseError:
            pass

    # ---- Backend interface -------------------------------------------------------------------

    def add(self, entry: Entry) -> None:
        ns, user = entry.scope
        key = self._key_prefix + entry.id
        vec = np.asarray(entry.vector, dtype=np.float32)
        if vec.shape != (self.dims,):
            raise ValueError(f"vector has shape {vec.shape}, index expects ({self.dims},)")
        mapping = {
            "prompt": entry.prompt,
            "answer": entry.answer,
            "ns": ns,
            "user": _tag_value(user),
            "tags": SEP.join(sorted(entry.tags)),
            "created_at": time.time(),
            "expires_at": NEVER if entry.expires_at is None else entry.expires_at,
            "vec": vec.tobytes(),
        }
        pipe = self.r.pipeline()
        pipe.hset(key, mapping=mapping)
        if entry.ttl is not None:
            # Real-time GC only; correctness comes from the expires_at filter. +1 s of slack.
            pipe.pexpire(key, int((entry.ttl + 1) * 1000))
        pipe.execute()

    def candidates(
        self, vector: np.ndarray, scope: Scope, now: float, k: int
    ) -> list[tuple[Entry, float]]:
        ns, user = scope
        flt = f"@ns:{{{_esc(ns)}}} @user:{{{_esc(_tag_value(user))}}} @expires_at:[({now} +inf]"
        q = (
            Query(f"({flt})=>[KNN {int(k)} @vec $v EF_RUNTIME {self.ef_runtime} AS dist]")
            .sort_by("dist")
            .return_fields("prompt", "answer", "ns", "user", "tags", "expires_at", "dist")
            .dialect(2)
            .paging(0, int(k))
        )
        vec = np.asarray(vector, dtype=np.float32).tobytes()
        res = self.r.ft(self.index).search(q, query_params={"v": vec})
        # Binary fields do not survive the search reply decoding, so fetch vectors by key.
        pipe = self.r.pipeline()
        for doc in res.docs:
            pipe.hget(doc.id, "vec")
        vecs = pipe.execute()
        out: list[tuple[Entry, float]] = []
        for doc, raw in zip(res.docs, vecs, strict=True):
            if raw is None:  # expired or deleted between the search and the fetch
                continue
            f = {_s(k_): v for k_, v in doc.__dict__.items()}
            exp = float(_s(f["expires_at"]))
            tags = _s(f["tags"])
            user_v = _s(f["user"])
            entry = Entry(
                prompt=_s(f["prompt"]),
                answer=_s(f["answer"]),
                vector=np.frombuffer(raw, dtype=np.float32),
                scope=(_s(f["ns"]), None if user_v == NO_USER else user_v[2:]),
                tags=frozenset(t for t in tags.split(SEP) if t),
                expires_at=None if exp >= NEVER else exp,
                id=doc.id[len(self._key_prefix) :],
            )
            if entry.scope != scope:  # defence in depth: never return another scope's entry
                continue
            out.append((entry, 1.0 - float(_s(f["dist"]))))
        return out

    def _delete_matching(self, query: str) -> int:
        deleted = 0
        while True:
            res = self.r.ft(self.index).search(Query(query).no_content().paging(0, 500).dialect(2))
            if not res.docs:
                return deleted
            n = int(self.r.delete(*[d.id for d in res.docs]))
            if n == 0:  # nothing left to delete (index lag); avoid spinning
                return deleted
            deleted += n

    def invalidate_tag(self, tag: str, namespace: str | None = None) -> int:
        q = f"@tags:{{{_esc(tag)}}}"
        if namespace is not None:
            q = f"@ns:{{{_esc(namespace)}}} " + q
        return self._delete_matching(q)

    def scope_of(self, entry_id: str) -> Scope | None:
        ns, user = self.r.hmget(self._key_prefix + entry_id, "ns", "user")
        if ns is None or user is None:
            return None
        u = _s(user)
        return (_s(ns), None if u == NO_USER else u[2:])

    def invalidate_namespace(self, namespace: str) -> int:
        return self._delete_matching(f"@ns:{{{_esc(namespace)}}}")

    def delete(self, entry_id: str) -> bool:
        return bool(self.r.delete(self._key_prefix + entry_id))

    def __len__(self) -> int:
        return int(self.r.ft(self.index).search(Query("*").no_content().paging(0, 0)).total)


def _s(v: object) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)
