import numpy as np
import pytest

from scache.cache import (
    Context,
    InMemoryBackend,
    SemanticCache,
    cosine_similarity,
    normalize,
)
from scache.embeddings import FakeEmbedder
from scache.guards import default_guards


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(threshold=0.8, guards=(), clock=None):
    return SemanticCache(
        InMemoryBackend(), FakeEmbedder(), threshold, guards=guards, clock=clock or Clock()
    )


def test_cosine_math():
    q = np.array([1.0, 0.0])
    m = np.array([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [1.0, 1.0]])
    assert cosine_similarity(q, m) == pytest.approx([1.0, 0.0, -1.0, 2**-0.5], abs=1e-6)
    assert np.linalg.norm(normalize(np.array([3.0, 4.0]))) == pytest.approx(1.0)
    assert cosine_similarity(q, np.zeros((1, 2)))[0] == 0.0


def test_put_get_exact_and_miss():
    c = make()
    assert c.put("How do I reset the password?", "Use the reset link.")
    hit = c.get("How do I reset the password?")
    assert hit.hit and hit.answer == "Use the reset link." and hit.similarity == pytest.approx(1.0)
    miss = c.get("Completely unrelated gardening question about tomatoes")
    assert not miss.hit and miss.reason == "below_threshold"
    assert make().get("anything").reason == "no_candidates"


def test_threshold_controls_hits():
    strict, loose = make(0.999), make(0.1)
    for c in (strict, loose):
        c.put("How do I reset the password?", "A")
    q = "How can I reset the password"
    assert not strict.get(q).hit
    assert loose.get(q).hit


def test_ttl_expiry():
    clock = Clock()
    c = make(clock=clock)
    c.put("q one", "A", ttl=60)
    assert c.get("q one").hit
    clock.t += 61
    assert not c.get("q one").hit


def test_tag_invalidation():
    c = make()
    c.put("q one", "A", tags=["billing"])
    c.put("q two completely different", "B", tags=["docs"])
    assert c.invalidate("billing") == 1
    assert not c.get("q one").hit
    assert c.get("q two completely different").hit


def test_namespace_and_user_scoping():
    c = make(guards=default_guards())
    a, b = Context("ns1", "u1"), Context("ns2", "u1")
    c.put("What is the refund window?", "30 days", a)
    assert c.get("What is the refund window?", a).hit
    assert not c.get("What is the refund window?", b).hit  # other namespace
    c.put("Where is my order?", "in transit", Context("ns1", "u1"))
    assert c.get("Where is my order?", Context("ns1", "u1")).hit
    assert not c.get("Where is my order?", Context("ns1", "u2")).hit  # other user
    anon = c.get("Where is my order?", Context("ns1", None))
    assert not anon.hit and anon.reason.startswith("never_cache")
    assert c.put("Where is my order?", "x", Context("ns1", None)) is False


def test_guards_block_a_near_identical_wrong_hit():
    q1, q2 = "What is the refund window for EU orders?", "What is the refund window for US orders?"
    plain, guarded = make(0.5), make(0.5, guards=default_guards())
    for c in (plain, guarded):
        c.put(q1, "14 days")
    assert plain.get(q2).hit
    r = guarded.get(q2)
    assert not r.hit and "entities differ" in r.reason


def test_next_candidate_is_tried_when_best_is_denied():
    c = make(0.3, guards=default_guards())
    c.put("What is the refund window for US orders?", "US answer")
    c.put("What is the refund window for EU orders?", "EU answer")
    assert c.get("What is the refund window for EU orders?").answer == "EU answer"
