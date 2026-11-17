import os
import uuid

import pytest
from fastapi.testclient import TestClient

from scache.main import Settings, build_state, create_app

pytestmark = pytest.mark.redis


def test_api_end_to_end_on_redis():
    prefix = f"scache-test:{uuid.uuid4().hex[:10]}"
    state = build_state(Settings(redis_url=os.environ["SCACHE_REDIS_URL"], redis_prefix=prefix))
    try:
        c = TestClient(create_app(state))
        body = {
            "prompt": "What is the refund window for EU orders?",
            "namespace": "shop",
            "tags": ["policy"],
        }
        a = c.post("/v1/complete", json=body).json()
        b = c.post("/v1/complete", json=body).json()
        assert (a["cache"], b["cache"]) == ("miss", "hit") and a["id"] == b["id"]
        us = c.post("/v1/complete", json={**body, "prompt": body["prompt"].replace("EU", "US")})
        assert us.json()["cache"] == "miss"
        assert c.post("/v1/feedback", json={"id": b["id"], "namespace": "shop"}).json() == {"evicted": True}
        assert c.post("/v1/invalidate", json={"namespace": "shop", "tag": "policy"}).json() == {"invalidated": 1}
        assert c.get("/v1/stats").json()["entries"] == 0
    finally:
        state.cache.backend.drop()


def test_ttl_sets_real_expire_on_the_key():
    import redis

    from scache.cache import SemanticCache
    from scache.embeddings import FakeEmbedder
    from scache.redis_backend import RedisBackend

    url = os.environ["SCACHE_REDIS_URL"]
    b = RedisBackend(url, 256, prefix=f"scache-test:{uuid.uuid4().hex[:10]}")
    try:
        c = SemanticCache(b, FakeEmbedder(), 0.8)
        e = c.store("short lived", "A", ttl=5)
        keep = c.store("long lived entirely different", "B")
        r = redis.Redis.from_url(url)
        assert 0 < r.pttl(b._key_prefix + e.id) <= 6000
        assert r.pttl(b._key_prefix + keep.id) == -1
    finally:
        b.drop()


def test_cross_user_isolation_on_redis():
    prefix = f"scache-test:{uuid.uuid4().hex[:10]}"
    state = build_state(Settings(redis_url=os.environ["SCACHE_REDIS_URL"], redis_prefix=prefix))
    try:
        c = TestClient(create_app(state))

        def ask(user):
            body = {"prompt": "Where is my order?", "namespace": "shop", "user_id": user}
            return c.post("/v1/complete", json=body).json()

        assert ask("victim")["cache"] == "miss"
        assert ask("victim")["cache"] == "hit"
        for other in ("Victim", "x", "u"):
            assert ask(other)["cache"] == "miss"
        poison = {"prompt": "Where is my order?", "namespace": "shop", "user_id": "x,u:victim"}
        assert c.post("/v1/complete", json=poison).status_code == 422
    finally:
        state.cache.backend.drop()


def test_old_case_insensitive_index_is_refused():
    import redis

    from scache.redis_backend import RedisBackend

    url = os.environ["SCACHE_REDIS_URL"]
    prefix = f"scache-test:{uuid.uuid4().hex[:10]}"
    r = redis.Redis.from_url(url)
    r.execute_command(
        "FT.CREATE", f"{prefix}:idx", "ON", "HASH", "PREFIX", "1", f"{prefix}:e:", "SCHEMA",
        "ns", "TAG", "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", "256",
        "DISTANCE_METRIC", "COSINE",
    )
    try:
        with pytest.raises(ValueError, match="older version"):
            RedisBackend(url, 256, prefix=prefix)
    finally:
        r.execute_command("FT.DROPINDEX", f"{prefix}:idx")
