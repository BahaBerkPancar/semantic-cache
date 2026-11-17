import os
import uuid

import pytest

REDIS_URL = os.environ.get("SCACHE_REDIS_URL")


def pytest_collection_modifyitems(config, items):
    if REDIS_URL:
        return
    skip = pytest.mark.skip(reason="SCACHE_REDIS_URL not set")
    for item in items:
        if "redis" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(params=["memory", pytest.param("redis", marks=pytest.mark.redis)])
def backend_factory(request):
    """Returns make(dims) -> a fresh, isolated backend of the parametrised kind."""
    created = []

    def make(dims: int = 256):
        if request.param == "memory":
            from scache.cache import InMemoryBackend

            return InMemoryBackend()
        from scache.redis_backend import RedisBackend

        b = RedisBackend(REDIS_URL, dims, prefix=f"scache-test:{uuid.uuid4().hex[:10]}")
        created.append(b)
        return b

    yield make
    for b in created:
        b.drop()
