import pytest

from scache.guards import (
    Scoper,
    extract_numbers,
    is_personal,
    make_entity_guard,
    negation_guard,
    numbers_guard,
    personal_guard,
    run_guards,
    time_scope_guard,
)


@pytest.mark.parametrize(
    "a,b",
    [
        ("migrate to v3", "migrate to version 3"),
        ("a 30-day trial", "a trial of 30 days"),
        ("2FA setup", "two-factor setup"),
        ("costs 1,000 dollars", "costs 1000 dollars"),
        ("python 3.11", "Python 3.11?"),
        ("thirty days", "30 days"),
        ("twenty-five users", "25 users"),
        ("seventeen items", "17 items"),
        ("seventy and eighty", "70 and 80"),
        ("v3", "version three"),
        ("Q3 results", "the third quarter results"),
        ("1.50", "1.5"),
        ("MP3 files", "mp3 files"),
    ],
)
def test_numbers_equal(a, b):
    assert numbers_guard(a, b).allow


@pytest.mark.parametrize(
    "a,b",
    [
        ("30 days", "60 days"),
        ("v2", "v3"),
        ("release 4.1", "release 4.2"),
        ("0.05 GB", "0.5 GB"),
        ("Q3 results", "Q4 results"),
        ("convert to MP3", "convert to MP4"),
        ("migrate from v2 to v3", "migrate from v3 to v2"),
        ("a refund", "a 30 day refund"),
        ("2 seats", "20 seats"),
    ],
)
def test_numbers_differ(a, b):
    assert not numbers_guard(a, b).allow


def test_extract_numbers_normalises():
    assert extract_numbers("v3 and version 03 in 2024-01-05") == ["3", "3", "2024", "1", "5"]
    assert extract_numbers("0.05 and 0.5") == ["0.05", "0.5"]
    assert extract_numbers("Q3 and second quarter") == ["q3", "q2"]


def test_entity_guard_default_gazetteer():
    g = make_entity_guard()
    assert not g("refund for EU orders", "refund for US orders").allow
    assert not g("the Pro plan", "the Enterprise plan").allow
    assert not g("Product A pricing", "Product B pricing").allow
    assert g("refund for EU orders", "return window for european customers? EU").allow
    assert g("WHAT ABOUT EU ORDERS", "what about eu orders").allow


def test_entity_guard_us_pronoun_is_not_the_country():
    g = make_entity_guard()
    assert g("can you tell us the price", "what is the price").allow


def test_entity_guard_is_configurable():
    g = make_entity_guard({"region:brazil": ["brazil"], "region:chile": ["chile"]})
    assert not g("ship to Brazil", "ship to Chile").allow
    assert g("ship to EU", "ship to US").allow  # not in this gazetteer


@pytest.mark.parametrize(
    "a,b,allowed",
    [
        ("Can I cancel?", "Can't I cancel?", False),
        ("Can I cancel?", "Can I not cancel?", False),
        ("Is it refundable?", "Is it non-refundable?", False),
        ("with cookies", "without cookies", False),
        ("Why can't I log in?", "Why am I unable to log in?", True),
        ("Is it not possible?", "Can it not be done?", True),
        ("Can I cancel?", "May I cancel?", True),
        ("Can I cancel?", "Can I cancel", True),
    ],
)
def test_negation_parity(a, b, allowed):
    assert negation_guard(a, b).allow is allowed


@pytest.mark.parametrize(
    "a,b,allowed",
    [
        ("orders today", "orders last year", False),
        ("sales this week", "sales last week", False),
        ("status now", "status yesterday", False),
        ("the latest release", "the current release", True),
        ("the latest release", "the first release", False),
        ("report for this month", "report for this month", True),
        ("how to export", "how to export today", False),
        ("every Monday", "every Tuesday", False),
    ],
)
def test_time_scope(a, b, allowed):
    assert time_scope_guard(a, b).allow is allowed


def test_personal_detection_and_guard():
    assert is_personal("Where is my order?")
    assert is_personal("I ordered a laptop")
    assert not is_personal("Where is the order form?")
    assert personal_guard("Where is my order?", "Where is my parcel?").allow
    assert not personal_guard("Where is my order?", "Where do orders ship?").allow


def test_scoper_scopes_personal_to_user_and_never_caches_anonymous():
    s = Scoper()
    assert s.scope("What is my balance?", "ns", "u1") == ("ns", "u1")
    assert s.scope("What is my balance?", "ns", "u2") != s.scope("What is my balance?", "ns", "u1")
    assert s.scope("What is my balance?", "ns", None) is None
    assert s.scope("What is a balance?", "ns", None) == ("ns", None)
    # ids containing separators cannot collide
    assert s.scope("my x", "a:user:b", None) is None
    assert s.scope("my x", "a", "user:b") != s.scope("my x", "a:user", "b")
    with pytest.raises(ValueError):
        s.scope("my x", "ns", "")
    with pytest.raises(ValueError):
        s.scope("my x", "", "u1")


def test_run_guards_reports_which_guard_denied():
    r = run_guards("30 days", "60 days", [numbers_guard, negation_guard])
    assert not r.allow and r.reason.startswith("numbers_guard")
    assert run_guards("a", "a", [numbers_guard, negation_guard]).allow


PERSONAL_PROMPTS = [
    "Where is my order?",
    "Which plan am I on?",
    "What plan do I have?",
    "Which username do I have?",
    "What did I order last time?",
    "Have I been charged twice?",
    "I'm locked out, what now?",
    "What is our storage quota?",
    "Send me the receipt",
    "I've paid already, why is it pending?",
    "What did we sign last year?",
]


@pytest.mark.parametrize("prompt", PERSONAL_PROMPTS)
def test_personal_prompts_never_cross_users(prompt):
    from scache.cache import Context, InMemoryBackend, SemanticCache
    from scache.embeddings import FakeEmbedder
    from scache.guards import default_guards

    assert is_personal(prompt)
    c = SemanticCache(InMemoryBackend(), FakeEmbedder(), 0.5, guards=default_guards())
    assert c.put(prompt, "answer for u1", Context("ns", "u1"))
    assert c.get(prompt, Context("ns", "u1")).hit
    other = c.get(prompt, Context("ns", "u2"))  # identical prompt, similarity 1.0
    assert not other.hit
    assert not c.get(prompt, Context("ns", None)).hit
    assert c.put(prompt, "x", Context("ns", None)) is False


@pytest.mark.parametrize(
    "prompt",
    ["How do I reset the password?", "What is the refund window?", "Can I cancel anytime?",
     "How do I order a licence?"],
)
def test_generic_how_to_prompts_are_shared(prompt):
    assert not is_personal(prompt)
