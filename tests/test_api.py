import pytest
from fastapi.testclient import TestClient

from scache.generators import FakeGenerator, GeneratorError
from scache.main import Settings, build_state, create_app


@pytest.fixture
def gen():
    return FakeGenerator()


@pytest.fixture
def client(gen):
    return TestClient(create_app(build_state(Settings(), generator=gen)))


def complete(client, prompt, **kw):
    r = client.post("/v1/complete", json={"prompt": prompt, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_miss_then_hit_skips_generator(client, gen):
    a = complete(client, "What is the refund window for EU orders?", namespace="shop")
    assert a["cache"] == "miss" and a["id"] and gen.calls == 1
    b = complete(client, "What is the refund window for EU orders?", namespace="shop")
    assert b["cache"] == "hit" and b["answer"] == a["answer"] and gen.calls == 1
    assert b["similarity"] == pytest.approx(1.0, abs=1e-4) and b["id"] == a["id"]
    assert b["latency_ms"] >= 0


def test_guard_reason_reported_on_near_miss(client):
    complete(client, "What is the refund window for EU orders?")
    r = complete(client, "What is the refund window for US orders?")
    assert r["cache"] == "miss" and any("entities differ" in x for x in r["reasons"])


def test_namespace_isolation(client, gen):
    complete(client, "What is the refund window?", namespace="a")
    assert complete(client, "What is the refund window?", namespace="b")["cache"] == "miss"


def test_cross_user_isolation(client):
    complete(client, "Where is my order?", user_id="u1")
    assert complete(client, "Where is my order?", user_id="u1")["cache"] == "hit"
    other = complete(client, "Where is my order?", user_id="u2")
    assert other["cache"] == "miss" and other["answer"]
    # u2's answer was stored under u2 and is not served to u1 as u2's
    assert complete(client, "Where is my order?", user_id="u2")["cache"] == "hit"


def test_never_cache_rules(client, gen):
    for kw in (
        {"prompt": "Where is my order?"},  # personal, no user
        {"prompt": "What is the weather today?", "user_id": "u"},
        {"prompt": "What is the refund window?", "no_cache": True},
    ):
        a = client.post("/v1/complete", json=kw).json()
        b = client.post("/v1/complete", json=kw).json()
        assert a["cache"] == b["cache"] == "bypass" and a["reasons"] and a["id"] is None
    assert gen.calls == 6
    assert client.get("/v1/stats").json()["entries"] == 0


def test_error_answer_not_stored(client):
    class Bad(FakeGenerator):
        def generate(self, prompt):
            return "Error: upstream timeout"

    c = TestClient(create_app(build_state(Settings(), generator=Bad())))
    r = complete(c, "What is the refund window?")
    assert r["cache"] == "miss" and r["id"] is None and "answer flagged as error" in r["reasons"]
    assert c.get("/v1/stats").json()["entries"] == 0


def test_generator_failure_is_502_and_counted():
    class Down(FakeGenerator):
        def generate(self, prompt):
            raise GeneratorError("ollama down")

    c = TestClient(create_app(build_state(Settings(), generator=Down())))
    assert c.post("/v1/complete", json={"prompt": "hello there"}).status_code == 502
    assert c.get("/v1/stats").json()["generator_errors"] == 1


def test_invalidate_by_tag_and_namespace(client, gen):
    complete(client, "q one about billing", tags=["billing"])
    complete(client, "q two about docs entirely", tags=["docs"], namespace="n2")
    assert client.post("/v1/invalidate", json={"namespace": "default", "tag": "billing"}).json() == {"invalidated": 1}
    assert complete(client, "q one about billing")["cache"] == "miss"
    assert client.post("/v1/invalidate", json={"namespace": "n2"}).json() == {"invalidated": 1}
    assert complete(client, "q two about docs entirely", namespace="n2")["cache"] == "miss"
    assert client.post("/v1/invalidate", json={}).status_code == 422
    assert client.post("/v1/invalidate", json={"tag": "docs"}).status_code == 422  # no namespace
    # a tag in another namespace is untouched
    complete(client, "q three about docs here", tags=["docs"], namespace="n3")
    assert client.post("/v1/invalidate", json={"namespace": "n4", "tag": "docs"}).json() == {
        "invalidated": 0
    }


def test_ttl(gen):
    state = build_state(Settings(), generator=gen)
    now = [1000.0]
    state.cache.clock = lambda: now[0]
    c = TestClient(create_app(state))
    complete(c, "short lived question", ttl=10)
    assert complete(c, "short lived question")["cache"] == "hit"
    now[0] += 11
    assert complete(c, "short lived question")["cache"] == "miss"


def test_feedback_evicts_and_counts(client):
    first = complete(client, "What is the refund window?")
    hit = complete(client, "What is the refund window?")
    assert hit["cache"] == "hit"
    r = client.post("/v1/feedback", json={"id": hit["id"], "namespace": "default", "wrong": True})
    assert r.json() == {"evicted": True}
    assert complete(client, "What is the refund window?")["cache"] == "miss"
    assert client.post("/v1/feedback", json={"id": first["id"], "namespace": "default"}).json() == {"evicted": False}
    s = client.get("/v1/stats").json()
    assert s["false_hit_reports"] == 2 and s["evictions"] == 1


def test_stats(client):
    complete(client, "What is the refund window?")
    complete(client, "What is the refund window?")
    complete(client, "anything", no_cache=True)
    s = client.get("/v1/stats").json()
    assert (s["hits"], s["misses"], s["bypasses"], s["requests"]) == (1, 1, 1, 3)
    assert s["hit_rate"] == 0.5 and s["entries"] == 1


def test_validation(client):
    assert client.post("/v1/complete", json={"prompt": ""}).status_code == 422
    assert client.post("/v1/complete", json={"prompt": "x", "ttl": -1}).status_code == 422
    assert client.post("/v1/complete", json={"prompt": "x", "tags": ["a,b"]}).status_code == 422


def test_feedback_is_scoped_to_namespace_and_user(client):
    shared = complete(client, "What is the refund window?", namespace="a")
    mine = complete(client, "Where is my order?", namespace="a", user_id="u1")
    fb = lambda **kw: client.post("/v1/feedback", json=kw).json()["evicted"]
    assert fb(id=shared["id"], namespace="b") is False  # wrong namespace
    assert fb(id=mine["id"], namespace="a") is False  # personal entry needs its user
    assert fb(id=mine["id"], namespace="a", user_id="u2") is False
    assert fb(id=mine["id"], namespace="a", user_id="u1") is True
    assert fb(id=shared["id"], namespace="a") is True


@pytest.mark.parametrize("bad", ["a,b", "x,u:victim", "x,none", "a b", "a|b", "", "x" * 129, "a*"])
def test_ids_with_separators_rejected(client, bad):
    for field in ("namespace", "user_id"):
        r = client.post("/v1/complete", json={"prompt": "hello there", field: bad})
        assert r.status_code == 422
    assert client.post("/v1/complete", json={"prompt": "x", "tags": [bad]}).status_code == 422


def test_case_sensitive_scopes_via_api(client):
    complete(client, "Where is my order?", user_id="Bob")
    assert complete(client, "Where is my order?", user_id="bob")["cache"] == "miss"
    complete(client, "What is the refund window?", namespace="Acme", tags=["Billing"])
    assert complete(client, "What is the refund window?", namespace="acme")["cache"] == "miss"
