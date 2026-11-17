import json
from pathlib import Path

import pytest

from scache import replay
from scache.embeddings import FakeEmbedder
from scache.traffic import Request

ROOT = Path(__file__).resolve().parent.parent
PW = "How do I reset a forgotten password?"
EU = "What is the refund window for EU orders?"
US = "What is the refund window for US orders?"
WHERE = "Where is my order?"
TH = {"cosine": 0.5, "cosine+guards": 0.5}


def req(i, t, prompt, intent, kind, user="u1", ns="shop"):
    return Request(i, t, ns, user, prompt, intent, kind, 100, 100)


def tiny_log():
    return [
        req(0, 0, PW, "faq:pw", "first_seen"),
        req(1, 10, PW, "faq:pw", "exact_repeat"),  # hit, correct (all caching variants)
        req(2, 20, EU, "refund:eu", "first_seen"),
        req(3, 30, US, "refund:us", "near_miss_entity"),  # cosine: wrong hit; guards: miss
        req(4, 40, WHERE, "personal:order:u1", "personal", user="u1"),
        req(5, 50, WHERE, "personal:order:u2", "personal", user="u2"),  # cosine: leak
        req(6, 1000, PW, "faq:pw", "exact_repeat"),  # past the TTL of 100 s: miss
    ]


def run(spec_key, ttl=100.0):
    reqs = tiny_log()
    spec = next(m for m in replay.default_modes(TH) if m.key == spec_key)
    return replay.run_mode(reqs, FakeEmbedder(), spec, ttl)


def test_no_cache_never_hits():
    r = run("no_cache")
    assert not r.hit.any()


def test_exact_match_hits_and_ttl():
    r = run("exact")
    assert r.hit.tolist() == [False, True, False, False, False, False, False]
    assert r.correct.tolist() == r.hit.tolist()  # the TTL expired for the last request
    long = run("exact", ttl=10_000)
    assert long.hit[6] and long.correct[6]


def test_cosine_only_serves_the_wrong_and_leaking_hits():
    r = run("cosine")
    assert r.hit.tolist() == [False, True, False, True, False, True, False]
    assert r.correct.tolist() == [False, True, False, False, False, False, False]
    assert r.wrong_type[3] == "near_miss_entity"
    assert r.wrong_type[5] == "personal_leak"
    assert r.leak.tolist() == [False] * 5 + [True, False]


def test_guards_block_the_wrong_hits_and_leaks():
    r = run("guards")
    assert r.hit.tolist() == [False, True, False, False, False, False, False]
    assert not r.leak.any()
    assert not any(r.wrong_type)


def test_ttl_in_replay_uses_the_logs_clock():
    short, long = run("guards", ttl=100.0), run("guards", ttl=10_000.0)
    assert not short.hit[6]
    assert long.hit[6] and long.correct[6]


def test_policy_mode_bypasses_time_sensitive_prompts():
    reqs = [
        req(0, 0, "Is the api service down today?", "status:api:d0", "time_sensitive"),
        req(1, 86_400 + 5, "Is the api service down today?", "status:api:d1", "time_sensitive"),
    ]
    modes = {m.key: m for m in replay.default_modes(TH)}
    plain = replay.run_mode(reqs, FakeEmbedder(), modes["guards"], ttl=10 * 86400)
    pol = replay.run_mode(reqs, FakeEmbedder(), modes["guards_policy"], ttl=10 * 86400)
    assert plain.hit[1] and plain.wrong_type[1] == "stale"
    assert not pol.hit.any()


def test_summary_numbers_on_the_tiny_log():
    res = replay.run_replay(tiny_log(), FakeEmbedder(), TH, ttl=100.0)
    cos, grd = res["modes"]["cosine"], res["modes"]["guards"]
    assert (cos["hits"], cos["correct_hits"], cos["wrong_hits"], cos["leaks"]) == (3, 1, 2, 1)
    assert cos["wrong_by_type"] == {"near_miss_entity": 1, "personal_leak": 1}
    assert (grd["hits"], grd["wrong_hits"], grd["leaks"]) == (1, 0, 0)
    assert grd["model_calls"] == 6 and grd["calls_saved"] == 1
    # every request costs (100 * 1 + 100 * 4) / 1e6; one call saved
    assert res["totals"]["no_cache_cost_usd"] == pytest.approx(7 * 500 / 1e6, abs=1e-4)
    assert grd["cost_saved_pct"] == pytest.approx(100 / 7, abs=0.01)
    assert res["modes"]["no_cache"]["hit_rate"] == 0
    lo, hi = cos["ci95"]["hit_rate"]
    assert lo - 0.01 <= 100 * cos["hit_rate"] <= hi + 0.01


def test_replay_is_deterministic_apart_from_timing():
    a = replay.run_replay(tiny_log(), FakeEmbedder(), TH, ttl=100.0)
    b = replay.run_replay(tiny_log(), FakeEmbedder(), TH, ttl=100.0)
    a.pop("timing"), b.pop("timing")
    assert json.dumps(a) == json.dumps(b)


def test_thresholds_come_from_m1_results_not_traffic(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"selection": {
        "cosine": {"threshold": 0.91}, "cosine+guards": {"threshold": 0.66}}}))
    th = replay.load_thresholds(p)
    assert th == {"cosine": 0.91, "cosine+guards": 0.66}
    modes = {m.key: m for m in replay.default_modes(th)}
    assert modes["cosine_m1"].threshold == 0.91
    assert modes["cosine"].threshold == modes["guards"].threshold == 0.66
    res = replay.run_replay(tiny_log(), FakeEmbedder(), th, ttl=100.0)
    assert res["meta"]["thresholds_from_m1"] == th
    # the shipped file feeds the shipped replay
    shipped = replay.load_thresholds(ROOT / "bench/results.json")
    assert json.loads((ROOT / "bench/replay.json").read_text())["meta"][
        "thresholds_from_m1"] == shipped


def test_reports_render(tmp_path):
    res = replay.run_replay(tiny_log(), FakeEmbedder(), TH, ttl=100.0, meta={"model": "fake"})
    replay.write_reports(res, tmp_path)
    assert "Replay results" in (tmp_path / "replay.md").read_text()
    assert (tmp_path / "replay.svg").read_text().startswith("<svg")
    assert json.loads((tmp_path / "replay.json").read_text())["totals"]["requests"] == 7


def test_shipped_replay_guarded_has_no_leaks():
    res = json.loads((ROOT / "bench/replay.json").read_text())
    assert res["modes"]["guards"]["leaks"] == 0
    assert res["modes"]["guards_policy"]["leaks"] == 0
    assert res["modes"]["cosine"]["leaks"] > 0


def test_wrong_hit_is_not_stored_so_a_repeat_is_served_wrongly_again():
    reqs = tiny_log()[:4] + [req(4, 40, US, "refund:us", "exact_repeat")]
    spec = next(m for m in replay.default_modes(TH) if m.key == "cosine")
    r = replay.run_mode(reqs, FakeEmbedder(), spec, 100.0)
    assert r.hit[3] and not r.correct[3]
    assert r.hit[4] and not r.correct[4]  # still answered from the EU entry
    assert r.entries_stored == 2  # PW and EU only
