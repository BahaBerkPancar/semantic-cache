"""Synthetic traffic log for the replay.

Everything here is invented: a fictional shop / SaaS support desk. Every request carries a hidden
`intent_id`: two requests with the same intent_id must get the same answer, different intent ids
must not share one. Labels hold by construction (a sibling that differs by region, plan, number,
negation or weekday is a different intent; a time-sensitive prompt is a new intent each day; a
personal prompt is a different intent for each user).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

SEED = 20261118
N_REQUESTS = 2000
BASE_TS = 1_780_000_000.0  # fixed start of the synthetic log (seconds); the log spans 7 days
SPAN_S = 7 * 86400.0
USERS = tuple(f"user-{i}" for i in range(1, 9))
ZIPF_S = 1.1

# Probability of each request kind (the rest, 5%, is "first_seen": a catalogue intent asked for
# the first time). Near-miss kinds are split by what differs: entity, number, negation, time.
MIX = {
    "exact_repeat": 0.30,
    "paraphrase": 0.25,
    "near_miss": 0.15,
    "personal": 0.08,
    "time_sensitive": 0.07,
    "unique": 0.10,
}


@dataclass(frozen=True)
class Request:
    i: int
    t: float
    namespace: str
    user_id: str
    prompt: str
    intent_id: str
    kind: str
    in_tokens: int
    out_tokens: int


def dump_traffic(requests: Sequence[Request], path: Path) -> None:
    Path(path).write_text("".join(json.dumps(asdict(r)) + "\n" for r in requests))


def load_traffic(path: Path) -> list[Request]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [Request(**r) for r in rows]


def traffic_texts(requests: Sequence[Request]) -> list[str]:
    return sorted({r.prompt for r in requests})


def mix_of(requests: Sequence[Request]) -> dict[str, int]:
    return dict(sorted(Counter(r.kind for r in requests).items()))


# ---- catalogue ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Intent:
    id: str
    namespace: str
    paraphrases: tuple[str, ...]


@dataclass(frozen=True)
class Group:
    name: str
    kind: str  # entity | number | negation | time | static
    intents: tuple[Intent, ...]


_REGIONS = [
    ("eu", "the EU"), ("us", "the US"), ("uk", "the UK"), ("canada", "Canada"),
    ("germany", "Germany"), ("france", "France"), ("japan", "Japan"), ("india", "India"),
    ("australia", "Australia"),
    # not in the guards' gazetteer: the entity guard cannot see these swaps
    ("brazil", "Brazil"), ("mexico", "Mexico"), ("spain", "Spain"), ("norway", "Norway"),
]
_PLANS = [
    ("basic", "Basic"), ("starter", "Starter"), ("pro", "Pro"), ("enterprise", "Enterprise"),
    ("team", "Team"), ("student", "Student"), ("nonprofit", "Nonprofit"),
]
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
]  # "May" is a modal verb to the time guard: a deliberate blind spot
_RELATIVE = ["this week", "next week", "this month", "next month"]

# (name, kind, namespace, 4 paraphrase templates with {x}, values as (key, surface))
_SLOT_FAMILIES = [
    ("return", "entity", "shop", [
        "What is the return period for parcels delivered to {x}?",
        "How long can a customer in {x} wait before asking for their money back?",
        "Up to which day is a return accepted for packages that arrived in {x}?",
        "What is the refund deadline for shipments to {x}?",
    ], _REGIONS),
    ("shipping", "entity", "shop", [
        "How many business days does delivery to {x} usually take?",
        "What is the typical shipping time for orders sent to {x}?",
        "How fast do packages reach {x}?",
        "What is the expected transit time to {x}?",
    ], _REGIONS),
    ("price", "entity", "shop", [
        "What is the monthly fee for the {x} plan?",
        "How much does the {x} tier cost each month?",
        "What is the monthly price of the {x} subscription?",
        "Monthly cost of the {x} plan, please?",
    ], _PLANS),
    ("upload", "number", "saas", [
        "What happens when a file larger than {x} MB is uploaded?",
        "Is a {x} MB attachment accepted by the uploader?",
        "Does the uploader reject anything over {x} MB?",
        "Can a single document of {x} MB be added to a project?",
    ], [(str(n), str(n)) for n in (5, 10, 25, 50, 100, 250)]),
    ("seats", "number", "saas", [
        "Is a team of {x} members covered by the shared workspace limit?",
        "Can {x} people collaborate in one workspace?",
        "Does one workspace allow {x} collaborators?",
        "Is it possible to add {x} users to a single workspace?",
    ], [(str(n), str(n)) for n in (3, 5, 10, 20, 50)]),
    ("retention", "number", "saas", [
        "How is data handled after {x} days of inactivity?",
        "What happens to an idle project once {x} days have passed?",
        "Is a project archived after {x} days without activity?",
        "What does the system do with data untouched for {x} days?",
    ], [(str(n), str(n)) for n in (7, 14, 30, 60, 90, 365)]),
    ("support-day", "time", "saas", [
        "What are the support hours on {x}?",
        "When is the help desk open on {x}?",
        "Is the support team reachable on {x}, and until when?",
        "Until what time is support staffed on {x}?",
    ], [(d.lower(), d) for d in _WEEKDAYS]),
    ("cutoff", "time", "shop", [
        "What is the order cutoff for shipping in {x}?",
        "Until when can orders be placed to ship in {x}?",
        "What is the last day to order for {x} dispatch?",
        "Order deadline for {x} shipments?",
    ], [(m.lower(), m) for m in _MONTHS]),
    ("roadmap", "time", "saas", [
        "Which features are planned for release {x}?",
        "What is on the release calendar {x}?",
        "Which updates does the roadmap list for {x}?",
        "What product changes land {x}?",
    ], [(r.replace(" ", "-"), r) for r in _RELATIVE]),
]

# negation groups: (name, namespace, positive paraphrases, negative paraphrases aligned by index)
_NEGATION = [
    ("refundable", "shop",
     ["Which orders qualify for a refund?", "Which purchases can be refunded?",
      "What kinds of orders are refundable?", "Which items are eligible for money back?"],
     ["Which orders do not qualify for a refund?", "Which purchases cannot be refunded?",
      "What kinds of orders are non-refundable?", "Which items are not eligible for money back?"]),
    ("export-formats", "saas",
     ["Which file formats are supported for export?", "What formats can a report be exported to?",
      "Which export formats does the tool handle?", "List the formats available when exporting."],
     ["Which file formats are not supported for export?", "What formats can a report not be exported to?",
      "Which export formats does the tool not handle?", "List the formats unavailable when exporting."]),
    ("sso", "saas",
     ["Which plans include single sign-on?", "On which tiers is SSO available?",
      "Which subscriptions come with SSO?", "Where is single sign-on offered?"],
     ["Which plans do not include single sign-on?", "On which tiers is SSO unavailable?",
      "Which subscriptions come without SSO?", "Where is single sign-on not offered?"]),
    ("webhooks", "saas",
     ["Which events can trigger a webhook?", "What actions are able to fire a webhook?",
      "Which activities send a webhook call?", "For which events does the platform notify via webhook?"],
     ["Which events cannot trigger a webhook?", "What actions are unable to fire a webhook?",
      "Which activities send no webhook call?", "For which events does the platform not notify via webhook?"]),
    ("free-shipping", "shop",
     ["Which orders ship for free?", "What purchases get free delivery?",
      "Which baskets are shipped at no charge?", "Which items qualify for free shipping?"],
     ["Which orders do not ship for free?", "What purchases get no free delivery?",
      "Which baskets are never shipped at no charge?", "Which items do not qualify for free shipping?"]),
]

# static FAQ intents: (name, namespace, 4 paraphrases)
_STATIC = [
    ("reset-password", "saas", [
        "How do I reset a forgotten password?", "What are the steps to recover a lost password?",
        "Password forgotten, how can it be reset?", "Where is the password recovery link?"]),
    ("invoice-download", "shop", [
        "Where can invoices be downloaded as PDF?", "How do I get a PDF copy of an invoice?",
        "Is there a place to fetch past invoices as PDF files?",
        "Which page lets me save an invoice as PDF?"]),
    ("two-factor", "saas", [
        "How do I switch on two-factor authentication?", "How can two-step login be activated?",
        "Where is the 2FA switch in the account settings?",
        "What is the procedure to enable two-factor login?"]),
    ("api-key", "saas", [
        "Where can a new API key be created?", "How do I generate an API token?",
        "Which screen creates a fresh API key?", "What is the way to get an API key for scripts?"]),
    ("delete-account", "saas", [
        "How can the account be deleted permanently?", "What is the procedure to close an account?",
        "Where do I find the option to remove the account for good?",
        "Steps to erase the whole account?"]),
    ("payment-methods", "shop", [
        "Which payment methods does the shop accept?", "What ways of paying does the shop offer?",
        "Can a purchase be paid by bank transfer or card?", "Which cards and wallets work at checkout?"]),
    ("gift-cards", "shop", [
        "Do gift cards expire?", "Is there an expiry date on gift cards?",
        "How long is a gift card valid?", "Can a gift card run out of validity?"]),
    ("change-address", "shop", [
        "How can the delivery address be changed after ordering?",
        "Is it possible to edit the shipping address once an order is placed?",
        "Where do I update the address for an order in progress?",
        "What are the options to correct a wrong delivery address?"]),
    ("dark-mode", "saas", [
        "Is there a dark theme for the dashboard?", "How do I switch the app to dark mode?",
        "Can the interface be shown with a dark colour scheme?",
        "Where is the dark theme setting?"]),
    ("mobile-app", "saas", [
        "Is there a mobile app for the platform?", "Which phones does the app run on?",
        "Can the service be used from a smartphone app?", "Do you offer iOS and Android clients?"]),
    ("data-location", "saas", [
        "Where is customer data stored?", "In which data centres do the servers live?",
        "Which hosting locations hold the stored data?", "What is the storage region of the data?"]),
    ("trial", "saas", [
        "Is there a free trial?", "Can the product be tested for free first?",
        "How long does the trial period last?", "Is a trial available before subscribing?"]),
    ("student-discount", "shop", [
        "Is there a discount for students?", "Do students get a reduced price?",
        "Can a student qualify for cheaper rates?", "Are there educational discounts?"]),
    ("warranty", "shop", [
        "How long is the warranty on devices?", "What is the guarantee period for hardware?",
        "For how many years are products covered by warranty?",
        "Does the hardware come with a manufacturer guarantee?"]),
    ("size-guide", "shop", [
        "Where is the size guide?", "How can the right clothing size be found?",
        "Is there a chart for picking a size?", "Which page lists measurements for each size?"]),
    ("newsletter", "shop", [
        "How do I stop the promotional newsletter?", "Where can newsletter emails be switched off?",
        "What is the way to unsubscribe from marketing mails?",
        "Option to stop receiving promotional messages?"]),
    ("audit-log", "saas", [
        "Is there an audit log of admin actions?", "Where can the history of admin changes be seen?",
        "Does the platform record who changed settings?", "How to view the activity trail of admins?"]),
    ("custom-domain", "saas", [
        "Can a custom domain be attached to a workspace?", "How do I use my own domain for the portal?",
        "Is a branded domain supported for the help centre?",
        "Steps to point a company domain at the portal?"]),
    ("uptime", "saas", [
        "What uptime does the service guarantee?", "Is there an availability commitment in the SLA?",
        "How reliable is the platform according to the SLA?", "Which uptime percentage is promised?"]),
    ("tax-invoice", "shop", [
        "Can an invoice carry a company tax number?", "How is a VAT number added to an invoice?",
        "Where does the tax id go so it appears on receipts?",
        "Is it possible to put a business tax number on the bill?"]),
    ("packaging", "shop", [
        "Is the packaging recyclable?", "Can the shipping boxes be recycled?",
        "What materials are used to pack orders?", "Is the parcel wrapping environmentally friendly?"]),
    ("import-csv", "saas", [
        "How can contacts be imported from a CSV file?", "What is the procedure to upload a spreadsheet of contacts?",
        "Where is the CSV import for contacts?", "Can a list of people be loaded from a CSV?"]),
    ("integrations", "saas", [
        "Which tools does the platform integrate with?", "What third-party apps can be connected?",
        "Is there a list of supported integrations?", "Which external services work with the product?"]),
    ("bulk-orders", "shop", [
        "Are there discounts for bulk orders?", "Is a volume price available for large purchases?",
        "Can big quantities be bought at a lower unit price?",
        "Do you offer wholesale rates for large orders?"]),
]

_PERSONAL = [
    ("order-status", [
        "Where is my parcel at the moment?", "Can you tell me where my package is?", "Has my order shipped yet?",
        "What is the tracking status of my parcel?"]),
    ("my-plan", [
        "Which plan is my account on?", "What subscription do I have?",
        "Tell me which plan my account uses.", "What is my subscription tier?"]),
    ("payment", [
        "Has the charge for my order gone through?", "Was my card charged for the last invoice?",
        "Has my payment been received?", "Is my invoice paid?"]),
    ("seats", [
        "How many seats does my workspace have?", "What is the seat count on my account?",
        "How many users can my team add?", "How many licenses do we own?"]),
    ("renewal", [
        "When does my subscription renew?", "What is the renewal day of my plan?",
        "On which day will my plan be renewed?", "When is my next billing date?"]),
]

_SERVICES = ["api", "dashboard", "billing", "exports", "webhooks", "login"]
_STATUS_TEMPLATES = [
    "Is the {x} service down today?", "What is the status of {x} today?",
    "Are there any {x} outages today?", "Is {x} working today?", "Is {x} working right now?",
]

_TAIL_APPS = [
    "Slack", "Jira", "Notion", "Zapier", "Trello", "Asana", "GitHub", "GitLab", "Salesforce",
    "HubSpot", "Shopify", "Stripe", "Zendesk", "Intercom", "Airtable", "Figma",
]
_TAIL_THINGS = [
    "invoices", "contacts", "tickets", "comments", "labels", "attachments", "tasks", "orders",
    "refunds", "customers",
]
_TAIL_TEMPLATES = [
    "How do I connect {a} to {b}?",
    "Is there a native {a} integration for {b} exports?",
    "Which scopes does the {a} connector need to read {t}?",
    "Can {t} be synced from {a} into {b} every hour?",
    "What permissions are required to push {t} to {b}?",
    "Why would the {a} sync skip {t}?",
]


def build_catalogue() -> list[Group]:
    groups: list[Group] = []
    for name, kind, ns, templates, values in _SLOT_FAMILIES:
        intents = tuple(
            Intent(f"{name}:{key}", ns, tuple(t.format(x=surface) for t in templates))
            for key, surface in values
        )
        groups.append(Group(name, kind, intents))
    for name, ns, pos, neg in _NEGATION:
        groups.append(Group(name, "negation", (
            Intent(f"{name}:pos", ns, tuple(pos)),
            Intent(f"{name}:neg", ns, tuple(neg)),
        )))
    for name, ns, paraphrases in _STATIC:
        groups.append(Group(name, "static", (Intent(f"faq:{name}", ns, tuple(paraphrases)),)))
    return groups


def catalogue_texts() -> set[str]:
    return {p for g in build_catalogue() for i in g.intents for p in i.paraphrases} | {
        p for _, paras in _PERSONAL for p in paras
    }


def _out_tokens(intent_id: str) -> int:
    return 60 + int(hashlib.sha256(intent_id.encode()).hexdigest(), 16) % 340


def _in_tokens(prompt: str) -> int:
    return 120 + round(len(prompt.split()) * 1.4)  # system prompt + question, illustrative


def generate(
    n: int = N_REQUESTS, seed: int = SEED, exclude: Collection[str] = ()
) -> list[Request]:
    """Deterministic synthetic log of n requests (same seed, same file)."""
    rng = random.Random(seed)
    groups = build_catalogue()
    group_of = {i.id: g for g in groups for i in g.intents}
    banned = set(exclude)
    clash = sorted(catalogue_texts() & banned)
    if clash:
        raise ValueError(f"traffic text also in the pair set: {clash[0]!r}")

    times = sorted(round(BASE_TS + rng.uniform(0, SPAN_S), 1) for _ in range(n))
    queue = [i for g in groups for i in g.intents]
    rng.shuffle(queue)

    seen_intents: list[Intent] = []  # first-appearance order = popularity rank
    seen_texts: list[tuple[Intent, int]] = []  # (intent, paraphrase index), first-use order
    seen_text_keys: set[tuple[str, int]] = set()
    used: dict[str, set[int]] = {}
    tail_used: set[str] = set()

    def zipf(k: int) -> int:
        return rng.choices(range(k), [1 / (r + 1) ** ZIPF_S for r in range(k)])[0]

    def note(intent: Intent, idx: int) -> None:
        if intent.id not in used:
            seen_intents.append(intent)
            used[intent.id] = set()
        used[intent.id].add(idx)
        if (intent.id, idx) not in seen_text_keys:
            seen_text_keys.add((intent.id, idx))
            seen_texts.append((intent, idx))

    def catalogue_request(intent: Intent, idx: int, kind: str) -> tuple[Intent, str, str]:
        note(intent, idx)
        return intent, intent.paraphrases[idx], kind

    def first_seen() -> tuple[Intent, str, str] | None:
        if not queue:
            return None
        intent = queue.pop()
        return catalogue_request(intent, rng.randrange(len(intent.paraphrases)), "first_seen")

    def exact_repeat() -> tuple[Intent, str, str] | None:
        if not seen_texts:
            return None
        intent, idx = seen_texts[zipf(len(seen_texts))]
        return catalogue_request(intent, idx, "exact_repeat")

    def paraphrase() -> tuple[Intent, str, str] | None:
        pool = [i for i in seen_intents if len(used[i.id]) < len(i.paraphrases)]
        if not pool:
            return exact_repeat()
        intent = pool[zipf(len(pool))]
        fresh = [k for k in range(len(intent.paraphrases)) if k not in used[intent.id]]
        return catalogue_request(intent, rng.choice(fresh), "paraphrase")

    def near_miss() -> tuple[Intent, str, str] | None:
        pool = [(i, k) for i, k in seen_texts if group_of[i.id].kind != "static"]
        if not pool:
            return None
        # pick the kind of swap first (entity/number/negation/time), then a popular text of it
        kinds = sorted({group_of[i.id].kind for i, _ in pool})
        want = rng.choice(kinds)
        pool = [(i, k) for i, k in pool if group_of[i.id].kind == want]
        intent, idx = pool[zipf(len(pool))]
        g = group_of[intent.id]
        sibling = rng.choice([s for s in g.intents if s.id != intent.id])
        return catalogue_request(sibling, idx, f"near_miss_{g.kind}")

    def tail() -> tuple[str, str, str, str]:
        while True:
            tpl = rng.choice(_TAIL_TEMPLATES)
            a, b = rng.sample(_TAIL_APPS, 2)
            text = tpl.format(a=a, b=b, t=rng.choice(_TAIL_THINGS))
            if text not in tail_used and text not in banned:
                tail_used.add(text)
                return text, f"tail:{len(tail_used)}", "saas", "unique"

    out: list[Request] = []
    for i, ts in enumerate(times):
        user = rng.choice(USERS)
        roll = rng.random()
        picked: tuple[str, str, str, str] | None = None  # prompt, intent_id, namespace, kind
        cat: tuple[Intent, str, str] | None = None
        edge = 0.0
        for kind, p in MIX.items():
            edge += p
            if roll < edge:
                break
        else:
            kind = "first_seen"
        if i == 0:
            kind = "first_seen"
        if kind == "exact_repeat":
            cat = exact_repeat()
        elif kind == "paraphrase":
            cat = paraphrase()
        elif kind == "near_miss":
            cat = near_miss()
        elif kind == "personal":
            topic, paras = _PERSONAL[zipf(len(_PERSONAL))]
            picked = (rng.choice(paras), f"personal:{topic}:{user}", "shop", "personal")
        elif kind == "time_sensitive":
            svc = _SERVICES[zipf(len(_SERVICES))]
            day = int((ts - BASE_TS) // 86400)
            text = rng.choice(_STATUS_TEMPLATES).format(x=svc)
            picked = (text, f"status:{svc}:d{day}", "saas", "time_sensitive")
        elif kind == "unique":
            picked = tail()
        if picked is None and cat is None and kind != "unique":
            cat = first_seen()
        if picked is None and cat is None:
            picked = tail()
        if cat is not None:
            intent, text, k = cat
            picked = (text, intent.id, intent.namespace, k)
        assert picked is not None
        prompt, intent_id, namespace, k = picked
        out.append(
            Request(i, ts, namespace, user, prompt, intent_id, k, _in_tokens(prompt),
                    _out_tokens(intent_id))
        )
    return out
