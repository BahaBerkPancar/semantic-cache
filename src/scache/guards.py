"""Guards: pure functions on (query, cached_prompt) that veto a cosine-similarity hit.

Each guard returns a GuardResult. Guards only ever deny; cosine similarity decides whether a
candidate is close enough, guards decide whether the two prompts still ask the same thing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class GuardResult:
    allow: bool
    reason: str = ""


Guard = Callable[[str, str], GuardResult]
ALLOW = GuardResult(True)


def _norm(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'").lower()


# ---- numbers / versions / dates ---------------------------------------------------------------

_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
# "one" only counts inside compounds ("twenty-one"): bare "one" is mostly a pronoun.
_BARE = {w: n for w, n in _UNITS.items() if w != "one"} | {"hundred": 100}
_COMPOUND_RE = re.compile(
    r"\b(" + "|".join(_TENS) + r")(?:[- ](" + "|".join(_UNITS) + r"))?\b"
)
_WORD_RE = re.compile(r"\b(" + "|".join(_BARE) + r")\b")
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
_QUARTER_RE = re.compile(r"\b(" + "|".join(_ORD) + r")\s+quarter\b|\bquarter\s+([1-4])\b")
# alphanumeric codes ("q3", "mp3", "ipv6") stay whole tokens; plain numbers and "v3" are numbers.
_NUM_RE = re.compile(
    r"\b([a-z]+\d+[a-z\d]*)\b|(?<![\w.])v?(\d+(?:,\d{3})*(?:\.\d+)*)"
)


def _canon_number(raw: str) -> str:
    raw = raw.replace(",", "")
    if raw.count(".") == 1:  # decimal: 0.05 != 0.5, 1.50 == 1.5
        return format(Decimal(raw).normalize(), "f")
    return ".".join(str(int(p)) for p in raw.split("."))  # integer or dotted version


def extract_numbers(text: str) -> list[str]:
    """Numbers and codes in order of appearance, normalised.

    "v3" == "version three" == "version 3"; "30-day" == "thirty days"; "Q3" == "third quarter";
    "0.05" != "0.5"; "MP3" != "MP4". Order is kept so "v2 to v3" != "v3 to v2".
    """
    t = _norm(text)
    t = _QUARTER_RE.sub(lambda m: f"q{_ORD[m.group(1)] if m.group(1) else m.group(2)}", t)
    t = _COMPOUND_RE.sub(
        lambda m: str(_TENS[m.group(1)] + (_UNITS[m.group(2)] if m.group(2) else 0)), t
    )
    t = _WORD_RE.sub(lambda m: str(_BARE[m.group(1)]), t)
    out = []
    for code, num in _NUM_RE.findall(t):
        if code:
            m = re.fullmatch(r"v(\d+)", code)
            out.append(_canon_number(m.group(1)) if m else code)
        else:
            out.append(_canon_number(num))
    return out


def numbers_guard(query: str, cached: str) -> GuardResult:
    a, b = extract_numbers(query), extract_numbers(cached)
    if a != b:
        return GuardResult(False, f"numbers differ: {a} vs {b}")
    return ALLOW


# ---- named entities (small configurable gazetteer) ---------------------------------------------

# canonical name -> aliases. An alias starting with "!" is matched case-sensitively (so the
# pronoun "us" does not count as the country).
DEFAULT_GAZETTEER: dict[str, list[str]] = {
    "region:eu": ["eu", "european union", "europe", "european"],
    "region:us": ["!US", "usa", "u.s.", "united states", "america", "american"],
    "region:uk": ["uk", "united kingdom", "britain", "british"],
    "region:canada": ["canada", "canadian"],
    "region:germany": ["germany", "german"],
    "region:france": ["france", "french"],
    "region:japan": ["japan", "japanese"],
    "region:india": ["india", "indian"],
    "region:australia": ["australia", "australian"],
    "plan:basic": ["basic"],
    "plan:starter": ["starter"],
    "plan:pro": ["pro"],
    "plan:enterprise": ["enterprise"],
    "product:a": ["product a"],
    "product:b": ["product b"],
    "product:model-x": ["model x"],
    "product:model-y": ["model y"],
    "product:widget-a": ["widget a"],
    "product:widget-b": ["widget b"],
}


def make_entity_guard(gazetteer: Mapping[str, Sequence[str]] | None = None) -> Guard:
    gaz = DEFAULT_GAZETTEER if gazetteer is None else gazetteer
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for canon, aliases in gaz.items():
        for alias in aliases:
            flags = 0
            if alias.startswith("!"):
                alias = alias[1:]
            else:
                flags = re.IGNORECASE
            compiled.append((canon, re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", flags)))

    def entities(text: str) -> set[str]:
        return {canon for canon, rx in compiled if rx.search(text)}

    def entity_guard(query: str, cached: str) -> GuardResult:
        a, b = entities(query), entities(cached)
        if a != b:
            return GuardResult(False, f"entities differ: {sorted(a)} vs {sorted(b)}")
        return ALLOW

    entity_guard.entities = entities  # type: ignore[attr-defined]
    return entity_guard


# ---- negation parity ---------------------------------------------------------------------------

_NEGATORS = {
    "not", "no", "never", "cannot", "cant", "without", "unable", "unavailable", "impossible",
    "non", "none", "nothing", "nobody", "neither", "nor", "disable", "disabled", "disallowed",
}
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def negation_count(text: str) -> int:
    n = 0
    for tok in _TOKEN_RE.findall(_norm(text)):
        if tok in _NEGATORS or tok.endswith("n't"):
            n += 1
    return n


def negation_guard(query: str, cached: str) -> GuardResult:
    a, b = negation_count(query) % 2, negation_count(cached) % 2
    if a != b:
        return GuardResult(False, "negation parity differs")
    return ALLOW


# ---- time scope --------------------------------------------------------------------------------

_TIME_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("today", re.compile(r"\b(today|tonight)\b")),
    ("yesterday", re.compile(r"\byesterday\b")),
    ("tomorrow", re.compile(r"\btomorrow\b")),
    ("now", re.compile(r"\b(now|currently|current|latest|recent|recently|right now)\b")),
    (
        "relative",
        re.compile(
            r"\b(last|this|next|previous)\s+(week|month|year|quarter|night|morning|evening)\b"
        ),
    ),
    (
        "weekday",
        re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"),
    ),
    (
        "month",
        re.compile(
            r"\b(january|february|march|april|june|july|august|september|october|november"
            r"|december)\b"
        ),
    ),
]


def time_scope(text: str) -> set[str]:
    t = _norm(text)
    out: set[str] = set()
    for name, rx in _TIME_PATTERNS:
        for m in rx.finditer(t):
            out.add(m.group(0) if name in {"relative", "weekday", "month"} else name)
    return out


def time_scope_guard(query: str, cached: str) -> GuardResult:
    a, b = time_scope(query), time_scope(cached)
    if a != b:
        return GuardResult(False, f"time scope differs: {sorted(a)} vs {sorted(b)}")
    return ALLOW


# ---- personal prompts --------------------------------------------------------------------------

_PERSONAL_RE = re.compile(
    r"\b(my|mine|myself|me|we|our|ours|ourselves|i'm|i've|i'd|i'll|am i|have i|did i"
    r"|do i (?:have|own|owe|use)"
    r"|i (?:have|own|owe|am|ordered|bought|placed|paid|purchased|subscribed|requested"
    r"|signed|got|received|sent|used|uploaded|created|deleted))\b"
)


def is_personal(text: str) -> bool:
    return bool(_PERSONAL_RE.search(_norm(text)))


def personal_guard(query: str, cached: str) -> GuardResult:
    """A personal prompt never matches a non-personal one. User scoping is the scoper's job."""
    if is_personal(query) != is_personal(cached):
        return GuardResult(False, "personal vs non-personal prompt")
    return ALLOW


def default_guards() -> list[Guard]:
    return [numbers_guard, make_entity_guard(), negation_guard, time_scope_guard, personal_guard]


def run_guards(query: str, cached: str, guards: Sequence[Guard]) -> GuardResult:
    for g in guards:
        r = g(query, cached)
        if not r.allow:
            name = getattr(g, "__name__", "guard")
            return GuardResult(False, f"{name}: {r.reason}")
    return ALLOW


# ---- scoping -----------------------------------------------------------------------------------


Scope = tuple[str, str | None]


class Scoper:
    """Maps (prompt, namespace, user) to the partition an entry lives in: (namespace, user).

    Personal prompts are scoped to the user; with no user id they are never cached (None). The
    scope is a tuple, so ids containing separators cannot collide. Empty ids are rejected.
    """

    def scope(self, prompt: str, namespace: str, user_id: str | None) -> Scope | None:
        from .cache import validate_id  # local import: cache imports this module

        validate_id(namespace, "namespace")
        if user_id is not None:
            validate_id(user_id, "user_id")
        if is_personal(prompt):
            if user_id is None:
                return None
            return (namespace, user_id)
        return (namespace, None)


class GlobalScoper(Scoper):
    """No scoping at all: everything in one namespace. Used for the cosine-only baseline."""

    def scope(self, prompt: str, namespace: str, user_id: str | None) -> Scope | None:
        return (namespace, None)
