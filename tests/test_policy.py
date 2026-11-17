from scache.policy import NeverCachePolicy

P = NeverCachePolicy()


def test_default_rules():
    assert P.check_prompt("Where is my order?", None)
    assert not P.check_prompt("Where is my order?", "u1")
    assert P.check_prompt("What is the weather today?", "u1")
    assert P.check_prompt("Who is the president right now?", "u1")
    assert P.check_prompt("Report for 2026-11-17", None)
    assert P.check_prompt("anything", "u1", no_cache=True)
    assert not P.check_prompt("What is the refund window for EU orders?", None)
    assert P.check_answer("Error: model overloaded")
    assert P.check_answer("fine", is_error=True)
    assert not P.check_answer("The refund window is 30 days.")


def test_rules_can_be_switched_off():
    off = NeverCachePolicy(volatile_time=False, explicit_dates=False, error_answers=False)
    assert not off.check_prompt("weather today", "u")
    assert not off.check_prompt("on 2026-11-17", "u")
    assert not off.check_answer("Error: x")
    assert off.check_prompt("x", "u", no_cache=True)  # explicit no_cache is not optional
