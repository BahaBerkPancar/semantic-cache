"""Replay a traffic log through the cache variants and report what they save and what they break.

A hit is correct iff the cached entry's intent_id equals the request's intent_id. The answer
stored for a miss is the request's intent_id itself, so no model is involved. The threshold comes
from bench/results.json (chosen in M1 on pair validation data), never from the traffic.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cache import Context, Embedder, InMemoryBackend, SemanticCache
from .guards import GlobalScoper, Scoper, default_guards, is_personal
from .policy import NeverCachePolicy
from .traffic import Request, mix_of

TTL_S = 24 * 3600.0
PRICES = {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 4.0}  # illustrative, not a real model
LATENCY = {"embed_ms": 25.0, "generation_ttft_ms": 300.0, "generation_ms_per_token": 12.0}
BLOCK = 50
BOOTSTRAP = 1000
BOOT_SEED = 20261118
TOLERANCE = 0.005


@dataclass(frozen=True)
class ModeSpec:
    key: str
    label: str
    kind: str  # none | exact | semantic
    threshold: float | None = None
    guarded: bool = False
    policy: bool = False


def load_thresholds(results_path: Path) -> dict[str, float]:
    """Thresholds chosen in M1 (bench/results.json), by mode name."""
    res = json.loads(Path(results_path).read_text())
    return {m: float(s["threshold"]) for m, s in res["selection"].items()}


def default_modes(thresholds: dict[str, float]) -> list[ModeSpec]:
    t_cos, t_grd = thresholds["cosine"], thresholds["cosine+guards"]
    return [
        ModeSpec("no_cache", "no cache", "none"),
        ModeSpec("exact", "exact match only", "exact"),
        ModeSpec("cosine_m1", f"cosine only, M1 threshold {t_cos:.2f}", "semantic", t_cos),
        ModeSpec("cosine", f"cosine only, threshold {t_grd:.2f}", "semantic", t_grd),
        ModeSpec("guards", f"cosine + guards, threshold {t_grd:.2f}", "semantic", t_grd, True),
        ModeSpec(
            "guards_policy",
            f"cosine + guards + never-cache policy, threshold {t_grd:.2f}",
            "semantic", t_grd, True, True,
        ),
    ]


@dataclass
class ModeRun:
    spec: ModeSpec
    hit: np.ndarray
    correct: np.ndarray
    wrong_type: list[str]  # "" when not a wrong hit
    leak: np.ndarray
    lookup_s: np.ndarray  # measured; not reproducible, never checked
    entries_stored: int = 0
    lookups: int = 0
    extra: dict = field(default_factory=dict)


def request_cost(r: Request) -> float:
    return (
        r.in_tokens * PRICES["input_usd_per_mtok"] + r.out_tokens * PRICES["output_usd_per_mtok"]
    ) / 1e6


def generation_ms(r: Request) -> float:
    return LATENCY["generation_ttft_ms"] + LATENCY["generation_ms_per_token"] * r.out_tokens


def _classify(req: Request, entry_intent: str, entry_personal: bool, entry_user: str) -> str:
    if entry_personal and entry_user != req.user_id:
        return "personal_leak"
    if req.intent_id.startswith("status:") and entry_intent.startswith("status:") and (
        req.intent_id.split(":")[1] == entry_intent.split(":")[1]
    ):
        return "stale"
    return req.kind


def run_mode(
    requests: Sequence[Request], embedder: Embedder, spec: ModeSpec, ttl: float = TTL_S
) -> ModeRun:
    n = len(requests)
    hit = np.zeros(n, dtype=bool)
    correct = np.zeros(n, dtype=bool)
    leak = np.zeros(n, dtype=bool)
    lookup_s = np.zeros(n)
    wrong_type = [""] * n
    clock = [0.0]
    cache: SemanticCache | None = None
    policy = NeverCachePolicy()
    meta: dict[str, tuple[str, str, bool]] = {}  # entry id -> (intent, owner user, personal)
    exact: dict[tuple, tuple[str, str, bool, float]] = {}
    stored = lookups = 0
    if spec.kind == "semantic":
        assert spec.threshold is not None
        cache = SemanticCache(
            InMemoryBackend(),
            embedder,
            spec.threshold,
            guards=default_guards() if spec.guarded else (),
            scoper=Scoper() if spec.guarded else GlobalScoper(),
            clock=lambda: clock[0],
        )
    for k, r in enumerate(requests):
        clock[0] = r.t
        personal = r.kind == "personal"
        if spec.kind == "none" or (spec.policy and policy.check_prompt(r.prompt, r.user_id)):
            continue
        ctx = Context(r.namespace, r.user_id)
        got: tuple[str, str, bool] | None = None
        t0 = time.perf_counter()
        if spec.kind == "exact":
            # key: namespace + prompt, plus the user for personal prompts (safe exact baseline)
            key = (r.namespace, r.user_id if is_personal(r.prompt) else None, r.prompt)
            e = exact.get(key)
            if e is not None and e[3] > r.t:
                got = (e[0], e[1], e[2])
        else:
            assert cache is not None
            res = cache.get(r.prompt, ctx)
            if res.hit and res.entry_id is not None:
                got = meta[res.entry_id]
        lookup_s[k] = time.perf_counter() - t0
        lookups += 1
        if got is not None:
            hit[k] = True
            intent, owner, entry_personal = got
            if intent == r.intent_id:
                correct[k] = True
            else:
                wrong_type[k] = _classify(r, intent, entry_personal, owner)
            leak[k] = entry_personal and owner != r.user_id
            continue
        # miss: "call the model", store the answer (its intent id)
        if spec.kind == "exact":
            key = (r.namespace, r.user_id if is_personal(r.prompt) else None, r.prompt)
            exact[key] = (r.intent_id, r.user_id, personal, r.t + ttl)
            stored += 1
        else:
            assert cache is not None
            entry = cache.store(r.prompt, r.intent_id, ctx, ttl=ttl)
            if entry is not None:
                meta[entry.id] = (r.intent_id, r.user_id, personal)
                stored += 1
    return ModeRun(spec, hit, correct, wrong_type, leak, lookup_s, stored, lookups)


# ---- aggregation -------------------------------------------------------------------------------


def _blocks(a: np.ndarray, block: int) -> np.ndarray:
    return np.add.reduceat(a.astype(float), np.arange(0, len(a), block))


def _ci(samples: np.ndarray) -> list[float]:
    return [round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)]


def summarise(requests: Sequence[Request], runs: Sequence[ModeRun], seed: int = BOOT_SEED) -> dict:
    n = len(requests)
    cost = np.array([request_cost(r) for r in requests])
    gen = np.array([generation_ms(r) for r in requests])
    nb = len(range(0, n, BLOCK))
    idx = np.random.default_rng(seed).integers(0, nb, size=(BOOTSTRAP, nb))
    b_n = _blocks(np.ones(n), BLOCK)
    b_cost = _blocks(cost, BLOCK)
    total_cost = float(cost.sum())
    modes: dict = {}
    timing: dict = {}
    for run in runs:
        spec = run.spec
        hit, correct = run.hit, run.correct
        wrong = hit & ~correct
        by_type = Counter(t for t in run.wrong_type if t)
        embeds = LATENCY["embed_ms"] if spec.kind == "semantic" else 0.0
        looked = np.array(
            [not (spec.policy and NeverCachePolicy().check_prompt(r.prompt, r.user_id))
             for r in requests]
        )  # policy-bypassed requests do no lookup, so no embedding either
        lat = np.where(hit, 0.0, gen) + embeds * looked
        saved = np.where(hit, cost, 0.0)
        saved_ok = np.where(correct, cost, 0.0)
        cols = {
            "hit_rate": (_blocks(hit, BLOCK), b_n, 100.0),
            "correct_hit_rate": (_blocks(correct, BLOCK), b_n, 100.0),
            "wrong_per_1000": (_blocks(wrong, BLOCK), b_n, 1000.0),
            "cost_saved_pct": (_blocks(saved, BLOCK), b_cost, 100.0),
            "cost_saved_correct_pct": (_blocks(saved_ok, BLOCK), b_cost, 100.0),
            "mean_latency_ms": (_blocks(lat, BLOCK), b_n, 1.0),
        }
        ci = {}
        for name, (num, den, scale) in cols.items():
            ci[name] = _ci(scale * num[idx].sum(axis=1) / den[idx].sum(axis=1))
        modes[spec.key] = {
            "label": spec.label,
            "threshold": spec.threshold,
            "hits": int(hit.sum()),
            "hit_rate": round(float(hit.mean()), 4),
            "correct_hits": int(correct.sum()),
            "wrong_hits": int(wrong.sum()),
            "wrong_per_1000": round(1000 * float(wrong.mean()), 2),
            "wrong_by_type": dict(sorted(by_type.items())),
            "leaks": int(run.leak.sum()),
            "model_calls": int(n - hit.sum()),
            "calls_saved": int(hit.sum()),
            "cost_usd": round(total_cost - float(saved.sum()), 4),
            "cost_saved_usd": round(float(saved.sum()), 4),
            "cost_saved_pct": round(100 * float(saved.sum()) / total_cost, 2),
            "cost_saved_correct_pct": round(100 * float(saved_ok.sum()) / total_cost, 2),
            "mean_latency_ms": round(float(lat.mean()), 1),
            "entries_stored": run.entries_stored,
            "ci95": ci,
            "series": {
                "step": BLOCK,
                "wrong_cum": np.cumsum(_blocks(wrong, BLOCK)).astype(int).tolist(),
                "cost_usd_cum": [round(x, 4) for x in np.cumsum(_blocks(cost - saved, BLOCK))],
            },
        }
        looked_up = run.lookup_s[run.lookup_s > 0]
        timing[spec.key] = {
            "lookup_median_us": round(float(np.median(looked_up)) * 1e6, 1) if len(looked_up) else 0,
            "lookup_p95_us": round(float(np.percentile(looked_up, 95)) * 1e6, 1)
            if len(looked_up) else 0,
        }
    return {
        "modes": modes,
        "timing": timing,
        "totals": {
            "requests": n,
            "no_cache_cost_usd": round(total_cost, 4),
            "no_cache_mean_latency_ms": round(float(gen.mean()), 1),
        },
    }


def run_replay(
    requests: Sequence[Request],
    embedder: Embedder,
    thresholds: dict[str, float],
    *,
    ttl: float = TTL_S,
    meta: dict | None = None,
) -> dict:
    requests = sorted(requests, key=lambda r: (r.t, r.i))
    runs = [run_mode(requests, embedder, spec, ttl) for spec in default_modes(thresholds)]
    res = {
        "meta": {
            **(meta or {}),
            "n_requests": len(requests),
            "days": round((requests[-1].t - requests[0].t) / 86400, 2),
            "mix": mix_of(requests),
            "n_unique_prompts": len({r.prompt for r in requests}),
            "n_intents": len({r.intent_id for r in requests}),
            "thresholds_from_m1": thresholds,
            "ttl_s": ttl,
            "prices_illustrative": PRICES,
            "latency_model_illustrative": LATENCY,
            "bootstrap": {"block": BLOCK, "resamples": BOOTSTRAP, "ci": "percentile 95%"},
        },
        **summarise(requests, runs),
    }
    return res


# ---- reports -----------------------------------------------------------------------------------


def _fmt(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}"


def _cell(m: dict, key: str, d: int = 1, suffix: str = "") -> str:
    lo, hi = m["ci95"][key]
    return f"{_fmt(m[key] if key != 'hit_rate' else 100 * m[key], d)}{suffix} [{_fmt(lo, d)}, {_fmt(hi, d)}]"


def render_markdown(res: dict) -> str:
    meta, modes, tot = res["meta"], res["modes"], res["totals"]
    order = list(modes)
    out = [
        "# Replay results",
        "",
        (f"Synthetic traffic: {meta['n_requests']} requests over {meta['days']} days, "
         f"{meta['n_unique_prompts']} distinct prompts, {meta['n_intents']} intents "
         f"(fictional shop / SaaS support desk, fixed seed). Embeddings: "
         f"`{meta.get('model', '?')}` ({meta.get('dims', '?')} dims). Entries live "
         f"{meta['ttl_s'] / 3600:.0f} h on the log's clock. Thresholds come from M1 "
         f"(`bench/results.json`), not from this traffic: "
         + ", ".join(f"{k} {v:.2f}" for k, v in meta["thresholds_from_m1"].items())
         + ". 95% CIs are block bootstrap percentiles "
         f"({meta['bootstrap']['resamples']} resamples of {meta['bootstrap']['block']}-request "
         "blocks)."),
        "",
        ("A hit is correct iff the cached entry's hidden intent id equals the request's. A wrong hit "
        "serves an answer to a different question. All dollar and latency figures use an "
        "**illustrative** price and latency model (below), not a real provider's."),
        "",
        "## Headline",
        "",
        ("| Variant | Hit rate % | Correct hits | Wrong hits | Wrong per 1,000 requests | "
        "Cross-user leaks |"),
        "|---|---|---|---|---|---|",
    ]
    for k in order:
        m = modes[k]
        out.append(
            f"| {m['label']} | {_cell(m, 'hit_rate')} | {m['correct_hits']} | {m['wrong_hits']} | "
            f"{_cell(m, 'wrong_per_1000')} | {m['leaks']} |"
        )
    out += [
        "",
        "## Cost and latency (illustrative model)",
        "",
        (f"Price table (illustrative): input ${meta['prices_illustrative']['input_usd_per_mtok']:.2f} "
         f"and output ${meta['prices_illustrative']['output_usd_per_mtok']:.2f} per million tokens. "
         f"No-cache spend: ${tot['no_cache_cost_usd']:.2f}. Latency model: embedding "
         f"{meta['latency_model_illustrative']['embed_ms']:.0f} ms per lookup, generation "
         f"{meta['latency_model_illustrative']['generation_ttft_ms']:.0f} ms + "
         f"{meta['latency_model_illustrative']['generation_ms_per_token']:.0f} ms per output "
         f"token (no-cache mean {tot['no_cache_mean_latency_ms']:.0f} ms). \"Saved\" counts every "
         "avoided model call, including calls whose wrong answer was served; \"saved by correct "
         "hits\" counts only correct ones."),
        "",
        ("| Variant | Model calls | Calls saved | Cost saved % | Saved by correct hits % | "
        "Mean latency ms |"),
        "|---|---|---|---|---|---|",
    ]
    for k in order:
        m = modes[k]
        out.append(
            f"| {m['label']} | {m['model_calls']} | {m['calls_saved']} | "
            f"{_cell(m, 'cost_saved_pct')} | {_cell(m, 'cost_saved_correct_pct')} | "
            f"{_cell(m, 'mean_latency_ms', 0)} |"
        )
    types = sorted({t for m in modes.values() for t in m["wrong_by_type"]})
    out += [
        "",
        "## Wrong hits by request type",
        "",
        ("Type is the request kind, except `personal_leak` (a personal answer cached for another "
        "user) and `stale` (a time-sensitive answer from an earlier day, served within the TTL)."),
        "",
        "| Type | " + " | ".join(modes[k]["label"] for k in order) + " |",
        "|---|" + "---|" * len(order),
    ]
    for t in types:
        out.append(f"| {t} | " + " | ".join(str(modes[k]["wrong_by_type"].get(t, 0)) for k in order) + " |")
    out += ["", "## Request mix", "", "| Kind | Requests |", "|---|---|"]
    for k, v in meta["mix"].items():
        out.append(f"| {k} | {v} |")
    if "timing" in res:
        out += [
            "",
            "## Measured lookup time (this machine)",
            "",
            ("Time of `SemanticCache.get` (vector search over the in-memory backend plus guards) "
            "with embeddings read from the cache, so it excludes the embedding call. It varies "
            "from run to run and is not part of `--check`."),
            "",
            "| Variant | Median us | p95 us |",
            "|---|---|---|",
        ]
        for k in order:
            tm = res["timing"][k]
            if tm["lookup_median_us"]:
                out.append(f"| {modes[k]['label']} | {tm['lookup_median_us']:.0f} | {tm['lookup_p95_us']:.0f} |")
    out += ["", "![cumulative wrong hits and spend](replay.svg)", ""]
    return "\n".join(out)


_COLORS = {
    "no_cache": "#6b7280", "exact": "#047857", "cosine_m1": "#7c3aed", "cosine": "#c2410c",
    "guards": "#1d4ed8", "guards_policy": "#0e7490",
}


def _panel(
    modes: dict, key: str, title: str, ylabel: str, x0: int, w: int, h: int, keys: Sequence[str]
) -> list[str]:
    top, bottom, left, right = 36, 46, 56, 12
    pw, ph = w - left - right, h - top - bottom
    series = {k: modes[k]["series"][key] for k in keys}
    npts = max(len(v) for v in series.values())
    ymax = max(max(v) for v in series.values()) or 1
    step = next(iter(modes.values()))["series"]["step"]
    parts = [f'<g transform="translate({x0} 0)">',
             f'<text x="{left + pw / 2}" y="20" text-anchor="middle" font-weight="bold">{title}</text>']
    for q in range(5):
        v = ymax * q / 4
        y = top + ph - ph * q / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        label = f"{v:.0f}" if ymax >= 10 else f"{v:.2f}"
        parts.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{label}</text>')
    for q in range(5):
        i = round((npts) * q / 4)
        x = left + pw * q / 4
        parts.append(f'<text x="{x:.1f}" y="{top + ph + 16}" text-anchor="middle">{i * step}</text>')
    parts.append(f'<text x="{left + pw / 2}" y="{h - 8}" text-anchor="middle">requests</text>')
    parts.append(f'<text transform="translate(12 {top + ph / 2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')
    for k, vals in series.items():
        pts = [(left, top + ph)] + [
            (left + pw * (i + 1) / npts, top + ph - ph * v / ymax) for i, v in enumerate(vals)
        ]
        parts.append(
            f'<polyline fill="none" stroke="{_COLORS[k]}" stroke-width="2" points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>'
        )
    parts.append("</g>")
    return parts


def render_svg(res: dict) -> str:
    modes = res["modes"]
    w, h = 1000, 400
    keys = [k for k in modes if k != "no_cache"]
    parts = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        'font-family="sans-serif" font-size="12">'),
        f'<rect width="{w}" height="{h}" fill="white"/>',
    ]
    parts += _panel(modes, "wrong_cum", "Cumulative wrong hits served", "wrong hits", 0, 500, 340, keys)
    parts += _panel(modes, "cost_usd_cum", "Cumulative spend (illustrative USD)", "USD", 500, 500, 340,
                    list(modes))
    for j, k in enumerate(modes):
        x, y = 20 + 160 * (j % 6), 372 + 0
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 16}" y2="{y}" stroke="{_COLORS[k]}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 20}" y="{y + 4}" font-size="10">{k}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_reports(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replay.json").write_text(json.dumps(res, indent=1) + "\n")
    (out_dir / "replay.md").write_text(render_markdown(res))
    (out_dir / "replay.svg").write_text(render_svg(res))
