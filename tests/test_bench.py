import itertools
import json
from pathlib import Path

import pytest

from scache import bench
from scache.embeddings import (
    EmbeddingStore,
    FakeEmbedder,
    StaleEmbeddingsError,
    StoreEmbedder,
    build_store,
    text_hash,
)

ROOT = Path(__file__).resolve().parent.parent


def toy_pairs():
    rows = []
    for i in range(12):
        rows.append(bench.Pair(f"p{i}", "How do I export data?", "How do I export data", "hit",
                               "case", "u", "u"))
        rows.append(bench.Pair(f"m{i}", "Refund for EU orders?", "Refund for US orders?", "miss",
                               "entity_swap", "u", "u"))
    return rows


def test_shipped_pair_set_meets_the_spec():
    pairs = bench.load_pairs(ROOT / "data" / "pairs.yaml")
    assert len(pairs) >= 300
    assert len({p.id for p in pairs}) == len(pairs)
    by = {}
    for p in pairs:
        by.setdefault(p.type, []).append(p)
    assert {"paraphrase", "reorder", "typo", "case", "entity_swap", "number_change", "negation",
            "time_scope", "personal", "topic_near"} <= set(by)
    assert all(len(v) >= 30 for v in by.values())
    n_hit = sum(p.label == "hit" for p in pairs)
    assert abs(n_hit - (len(pairs) - n_hit)) <= 0.05 * len(pairs)
    assert all(p.label in {"hit", "miss"} for p in pairs)


def test_split_is_stable_and_roughly_even():
    assert bench.split_of("paraphrase-001") == bench.split_of("paraphrase-001")
    pairs = bench.load_pairs(ROOT / "data" / "pairs.yaml")
    n_val = sum(bench.split_of(p.id) == "val" for p in pairs)
    assert 0.35 * len(pairs) < n_val < 0.65 * len(pairs)


def test_select_threshold_uses_only_the_numbers_it_is_given():
    th = [0.7, 0.8, 0.9]
    assert bench.select_threshold(th, [0.9, 0.8, 0.5], [0.2, 0.04, 0.0], 0.05) == (0.8, True)
    # ties on hit rate: higher threshold wins
    assert bench.select_threshold(th, [0.8, 0.8, 0.8], [0.0, 0.0, 0.0], 0.05) == (0.9, True)
    # infeasible: lowest false-hit rate, then higher hit rate
    assert bench.select_threshold(th, [0.9, 0.7, 0.6], [0.5, 0.3, 0.3], 0.05) == (0.8, False)


def test_selection_is_made_on_validation_only(monkeypatch):
    pairs = toy_pairs()
    seen = {}
    real = bench.select_threshold

    def spy(thresholds, hit_rate, false_rate, max_false=0.05):
        seen["called"] = True
        return real(thresholds, hit_rate, false_rate, max_false)

    monkeypatch.setattr(bench, "select_threshold", spy)
    res = bench.run_bench(pairs, FakeEmbedder())
    assert seen["called"]
    # Corrupt the test split's labels: the chosen threshold must not change.
    test_ids = {p.id for p in pairs if bench.split_of(p.id) == "test"}
    flipped = [
        bench.Pair(p.id, p.a, p.b, "miss" if p.label == "hit" else "hit", p.type, p.user_a, p.user_b)
        if p.id in test_ids else p
        for p in pairs
    ]
    res2 = bench.run_bench(flipped, FakeEmbedder())
    for mode in bench.MODES:
        assert res["selection"][mode]["threshold"] == res2["selection"][mode]["threshold"]


def test_wilson_matches_hand_computed_values():
    lo, hi = bench.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)  # not zero-width
    lo, hi = bench.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    lo, hi = bench.wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)
    assert bench.wilson(0, 0) == (0.0, 1.0)


def test_rates_are_hand_computable_on_a_tiny_set():
    pairs = toy_pairs()
    res = bench.run_bench(pairs, FakeEmbedder())
    t = res["selection"]["cosine"]["test"]
    n_hit_test = sum(p.label == "hit" and bench.split_of(p.id) == "test" for p in pairs)
    assert t["hit"]["n"] == n_hit_test


def test_bench_known_counts_with_fake_embedder():
    res = bench.run_bench(toy_pairs(), FakeEmbedder())
    sw = res["sweep"]
    # Same prompt up to a trailing '?': trigram vectors are very close at 0.70.
    assert sw["cosine"]["hit"]["rate"][0] == 1.0
    # EU/US swap is near-identical under cosine but the entity guard always denies it.
    assert sw["cosine"]["miss"]["rate"][0] == 1.0
    assert set(sw["cosine+guards"]["miss"]["rate"]) == {0.0}
    # Rates never increase with the threshold.
    r = sw["cosine"]["hit"]["rate"]
    assert all(x >= y for x, y in itertools.pairwise(r))


def test_bench_is_deterministic():
    a = bench.run_bench(toy_pairs(), FakeEmbedder())
    b = bench.run_bench(toy_pairs(), FakeEmbedder())
    assert json.dumps(a) == json.dumps(b)
    assert bench.diff_results(a, b) == []
    b["sweep"]["cosine"]["hit"]["rate"][0] += 0.1
    assert bench.diff_results(a, b, tol=0.005)
    assert not bench.diff_results(a, b, tol=0.2)


def test_reports_render(tmp_path):
    res = bench.run_bench(toy_pairs(), FakeEmbedder(), meta={"model": "fake", "dims": 256})
    bench.write_reports(res, tmp_path)
    assert "Benchmark results" in (tmp_path / "results.md").read_text()
    assert (tmp_path / "curve.svg").read_text().startswith("<svg")
    assert json.loads((tmp_path / "results.json").read_text())["thresholds"][0] == 0.7


def test_embedding_store_roundtrip_and_staleness(tmp_path):
    class Fake(FakeEmbedder):
        model, prefix = "fake-model", "search_query: "

    texts = ["alpha", "beta"]
    store, n = build_store(texts, Fake())
    assert n == 2
    path = tmp_path / "e.npz"
    store.save(path)
    loaded = EmbeddingStore.load(path)
    assert loaded.model == "fake-model" and loaded.dims == 256 and loaded.texts == texts
    assert loaded.stale(texts) == []
    assert loaded.stale(["alpha", "gamma"]) == ["gamma"]
    _again, n2 = build_store(texts + ["gamma"], Fake(), loaded)
    assert n2 == 1  # only the new text is embedded
    # a changed hash (e.g. different prefix/model) makes an entry stale
    loaded.hashes[0] = text_hash("alpha", "other-model", loaded.prefix)
    assert loaded.stale(["alpha"]) == ["alpha"]
    with pytest.raises(StaleEmbeddingsError):
        StoreEmbedder(loaded).embed(["alpha"])


def test_shipped_embeddings_are_not_stale():
    pairs = bench.load_pairs(ROOT / "data" / "pairs.yaml")
    store = EmbeddingStore.load(ROOT / "data" / "embeddings.npz")
    adv = bench.load_pairs(ROOT / "data" / "adversarial_hits.yaml")
    assert store.stale(bench.pair_texts(pairs + adv)) == []


def test_adversarial_set_is_held_out_hits():
    adv = bench.load_pairs(ROOT / "data" / "adversarial_hits.yaml")
    assert len(adv) >= 40 and all(p.label == "hit" for p in adv)
    main_ids = {p.id for p in bench.load_pairs(ROOT / "data" / "pairs.yaml")}
    assert not main_ids & {p.id for p in adv}


def test_adversarial_report_counts_guard_denials():
    toy = [bench.Pair("a1", "Refund for EU orders?", "Refund for US orders?", "hit", "x", "u", "u"),
           bench.Pair("a2", "How do I export?", "How do I export", "hit", "x", "u", "u")]
    res = bench.run_bench(toy_pairs(), FakeEmbedder(), adversarial=toy)
    a = res["adversarial"]
    assert a["n"] == 2 and a["guard_false_deny"]["rate"] == 0.5
    assert a["denied_by"] == {"entity_guard": 1}


def test_shipped_results_reproduce():
    pairs = bench.load_pairs(ROOT / "data" / "pairs.yaml")
    store = EmbeddingStore.load(ROOT / "data" / "embeddings.npz")
    adv = bench.load_pairs(ROOT / "data" / "adversarial_hits.yaml")
    res = bench.run_bench(
        pairs, StoreEmbedder(store), adversarial=adv,
        meta={"model": store.model, "dims": store.dims, "prefix": store.prefix},
    )
    old = json.loads((ROOT / "bench" / "results.json").read_text())
    assert bench.diff_results(old, res) == []
