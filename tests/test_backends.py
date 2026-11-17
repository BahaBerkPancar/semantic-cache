"""Contract tests: the same suite runs against InMemoryBackend and RedisBackend."""

import numpy as np
import pytest

from scache.cache import Context, SemanticCache
from scache.embeddings import FakeEmbedder
from scache.guards import default_guards


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def make(backend_factory):
    def build(threshold=0.8, guards=(), clock=None):
        return SemanticCache(
            backend_factory(), FakeEmbedder(), threshold, guards=guards, clock=clock or Clock()
        )

    return build


def test_put_get_and_miss(make):
    c = make()
    assert len(c.backend) == 0
    assert c.put("How do I reset the password?", "Use the reset link.")
    hit = c.get("How do I reset the password?")
    assert hit.hit and hit.answer == "Use the reset link."
    assert hit.similarity == pytest.approx(1.0, abs=1e-4) and hit.entry_id
    assert len(c.backend) == 1
    assert not c.get("Completely unrelated gardening question about tomatoes").hit


def test_nearest_of_several(make):
    c = make(0.3)
    c.put("How do I reset the password?", "A")
    c.put("What is the capital of France?", "B")
    c.put("Best way to cook rice on the stove", "C")
    assert c.get("capital of France?").answer == "B"


def test_namespace_and_user_isolation(make):
    c = make(guards=default_guards())
    a, b = Context("ns1", "u1"), Context("ns2", "u1")
    c.put("What is the refund window?", "30 days", a)
    assert c.get("What is the refund window?", a).hit
    assert not c.get("What is the refund window?", b).hit
    c.put("Where is my order?", "in transit", Context("ns1", "u1"))
    assert c.get("Where is my order?", Context("ns1", "u1")).hit
    assert not c.get("Where is my order?", Context("ns1", "u2")).hit
    assert c.get("Where is my order?", Context("ns1", None)).reason.startswith("never_cache")


def test_scope_filter_does_not_starve_topk(make):
    """Many closer entries in another namespace must not crowd out the in-scope match."""
    c = make(0.3)
    other = Context("other")
    for i in range(30):
        c.put("How do I reset the password?", f"other-{i}", other)
    c.put("How do I reset the password please", "mine", Context("mine"))
    assert c.get("How do I reset the password?", Context("mine")).answer == "mine"


def test_special_characters_in_ids(make):
    c = make()
    ctx = Context("team-a_b:c-1.2", "user@example.com")
    c.put("Where is my order?", "x", ctx, tags=["tag.with:chars", "a-b"])
    assert c.get("Where is my order?", ctx).hit
    assert not c.get("Where is my order?", Context("team-a_b:c-1.2", "user@example.org")).hit
    assert not c.get("Where is my order?", Context("team-a_b:c-1.3", "user@example.com")).hit
    assert c.invalidate("tag.with:chars") == 1


def test_ttl_expiry(make):
    clock = Clock()
    c = make(clock=clock)
    c.put("q one", "A", ttl=60)
    c.put("q forever different words", "B")
    assert c.get("q one").hit
    clock.t += 61
    assert not c.get("q one").hit
    assert c.get("q forever different words").hit


def test_tag_invalidation(make):
    c = make()
    c.put("q one", "A", tags=["billing", "x"])
    c.put("q two completely different", "B", tags=["docs"])
    c.put("q three entirely other topic", "C", tags=["billing"])
    assert c.invalidate("billing") == 2
    assert c.invalidate("billing") == 0
    assert not c.get("q one").hit
    assert c.get("q two completely different").hit
    assert len(c.backend) == 1


def test_namespace_invalidation(make):
    c = make()
    c.put("q one", "A", Context("ns1"))
    c.put("q one", "B", Context("ns2"))
    assert c.invalidate_namespace("ns1") == 1
    assert not c.get("q one", Context("ns1")).hit
    assert c.get("q one", Context("ns2")).answer == "B"


def test_evict_by_id(make):
    c = make()
    e = c.store("q one", "A")
    assert c.evict(e.id, Context()) is True
    assert c.evict(e.id, Context()) is False
    assert not c.get("q one").hit


def test_returned_entry_roundtrip(make):
    c = make()
    e = c.store("q one", "A", Context("ns", None), tags=["t1", "t2"], ttl=100)
    got = c.backend.candidates(c._vector("q one"), ("ns", None), c.clock(), 3)
    assert len(got) == 1
    entry, sim = got[0]
    assert entry.id == e.id and entry.scope == ("ns", None) and entry.tags == {"t1", "t2"}
    assert entry.expires_at == pytest.approx(e.expires_at)
    assert np.allclose(entry.vector, e.vector, atol=1e-6) and sim == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("bad", ["a,b", "x,u:victim", "x,none", "a b", "a|b", "", "a*"])
def test_unsafe_ids_rejected(make, bad):
    c = make()
    with pytest.raises(ValueError):
        c.put("q", "a", tags=[bad])
    with pytest.raises(ValueError):
        c.put("q", "a", Context(bad))
    with pytest.raises(ValueError):
        c.put("Where is my order?", "a", Context("ns", bad))
    with pytest.raises(ValueError):
        c.get("q", Context(bad))


def test_no_cross_user_poisoning(make):
    c = make(guards=default_guards())
    c.put("Where is my order?", "victim data", Context("ns", "victim"))
    c.put("Where is my order?", "none-scope data", Context("ns", "other"))
    for evil in ("x,u:victim", "x,none"):
        with pytest.raises(ValueError):
            c.put("Where is my order?", "evil", Context("ns", evil))
    assert c.get("Where is my order?", Context("ns", "x")).hit is False


def test_case_sensitive_ids(make):
    c = make()
    c.put("Where is my order?", "bob", Context("ns", "Bob"))
    assert c.get("Where is my order?", Context("ns", "Bob")).hit
    assert not c.get("Where is my order?", Context("ns", "bob")).hit
    c.put("What is the refund window?", "acme", Context("Acme"))
    assert not c.get("What is the refund window?", Context("acme")).hit
    assert c.get("What is the refund window?", Context("Acme")).hit
    c.put("q one about billing", "x", Context("Acme"), tags=["billing"])
    assert c.invalidate("Billing") == 0
    assert c.invalidate("billing", "acme") == 0
    assert c.invalidate("billing", "Acme") == 1
    assert c.invalidate_namespace("acme") == 0


def test_tag_invalidation_scoped_by_namespace(make):
    c = make()
    c.put("q one", "A", Context("n1"), tags=["t"])
    c.put("q one", "B", Context("n2"), tags=["t"])
    assert c.invalidate("t", "n1") == 1
    assert c.get("q one", Context("n2")).answer == "B"


def test_evict_checks_scope(make):
    c = make()
    e = c.store("Where is my order?", "A", Context("ns", "u1"))
    assert c.evict(e.id, Context("other", "u1")) is False
    assert c.evict(e.id, Context("ns", "u2")) is False
    assert c.evict(e.id, Context("ns", None)) is False
    assert c.evict(e.id, Context("ns", "u1")) is True
