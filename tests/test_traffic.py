from collections import Counter
from pathlib import Path

import pytest

from scache import bench, traffic
from scache.guards import default_guards, run_guards

ROOT = Path(__file__).resolve().parent.parent


def pair_texts() -> set[str]:
    pairs = bench.load_pairs(ROOT / "data/pairs.yaml") + bench.load_pairs(
        ROOT / "data/adversarial_hits.yaml"
    )
    return set(bench.pair_texts(pairs))


def test_generator_is_deterministic_and_seeded():
    a = traffic.generate(400, seed=1)
    assert a == traffic.generate(400, seed=1)
    assert a != traffic.generate(400, seed=2)
    assert [r.t for r in a] == sorted(r.t for r in a)
    assert [r.i for r in a] == list(range(400))


def test_shipped_log_matches_the_generator_and_avoids_pair_texts():
    shipped = traffic.load_traffic(ROOT / "data/traffic.jsonl")
    assert shipped == traffic.generate(exclude=pair_texts())
    assert len(shipped) == traffic.N_REQUESTS
    assert not {r.prompt for r in shipped} & pair_texts()


def test_generator_refuses_texts_from_the_pair_set():
    some = next(iter(traffic.catalogue_texts()))
    with pytest.raises(ValueError):
        traffic.generate(50, exclude={some})


def test_mix_proportions():
    reqs = traffic.generate()
    n = len(reqs)
    kinds = Counter(r.kind for r in reqs)
    near = sum(v for k, v in kinds.items() if k.startswith("near_miss"))
    assert 0.12 < near / n < 0.20
    assert 0.06 < kinds["personal"] / n < 0.11
    assert 0.05 < kinds["time_sensitive"] / n < 0.09
    assert 0.08 < kinds["unique"] / n < 0.12
    assert 0.10 < kinds["paraphrase"] / n < 0.18
    assert kinds["exact_repeat"] / n > 0.25
    assert len({r.user_id for r in reqs}) > 4
    assert len({r.namespace for r in reqs}) == 2
    # Zipf-like: a few intents take a large share of the traffic
    top = Counter(r.intent_id for r in reqs).most_common(10)
    assert sum(c for _, c in top) / n > 0.25


def test_intent_labels_are_consistent():
    reqs = traffic.generate()
    text_intents: dict[tuple[str, str], set[str]] = {}
    for r in reqs:
        text_intents.setdefault((r.namespace, r.prompt, r.user_id if r.kind == "personal" else ""), set()).add(r.intent_id)
    # one text (per user, for personal prompts) never maps to two intents, except time-sensitive
    # prompts, whose answer changes with the day
    for (_, prompt, _), ids in text_intents.items():
        if len(ids) > 1:
            assert all(i.startswith("status:") for i in ids), prompt
    # personal prompts: intent is per user
    assert all(r.intent_id.endswith(r.user_id) for r in reqs if r.kind == "personal")
    # time-sensitive intents are per day
    days = {(r.intent_id.split(":")[1], r.intent_id.split(":")[2])
            for r in reqs if r.kind == "time_sensitive"}
    assert len({d for _, d in days}) >= 5


def test_near_misses_are_new_intents_that_look_alike():
    reqs = traffic.generate()
    by_intent_text: dict[str, set[str]] = {}
    for r in reqs:
        by_intent_text.setdefault(r.intent_id, set()).add(r.prompt)
    near = [r for r in reqs if r.kind.startswith("near_miss")]
    assert near
    for r in near:
        family = r.intent_id.split(":")[0]
        siblings = [i for i in by_intent_text if i.startswith(family + ":") and i != r.intent_id]
        assert siblings, r.intent_id
    # intents never share a prompt text (entity/number/negation/time swaps change the text)
    owner: dict[str, str] = {}
    for r in reqs:
        if r.kind in {"personal", "time_sensitive"}:
            continue
        assert owner.setdefault(r.prompt, r.intent_id) == r.intent_id


def test_guards_catch_most_catalogued_swaps_and_miss_the_documented_gaps():
    groups = {g.name: g for g in traffic.build_catalogue()}
    guards = default_guards()

    def allows(a: str, b: str) -> bool:
        return run_guards(a, b, guards).allow

    # number, negation, weekday swaps are caught; the same wording with a different intent
    # is denied
    for name in ("upload", "seats", "refundable", "support-day"):
        g = groups[name]
        a, b = g.intents[0].paraphrases[0], g.intents[1].paraphrases[0]
        assert not allows(a, b), name
    # a region outside the gazetteer is invisible to the entity guard (documented blind spot)
    ret = groups["return"]
    brazil = next(i for i in ret.intents if i.id.endswith("brazil"))
    spain = next(i for i in ret.intents if i.id.endswith("spain"))
    assert allows(brazil.paraphrases[0], spain.paraphrases[0])


def test_roundtrip(tmp_path):
    reqs = traffic.generate(30, seed=3)
    traffic.dump_traffic(reqs, tmp_path / "t.jsonl")
    assert traffic.load_traffic(tmp_path / "t.jsonl") == reqs
