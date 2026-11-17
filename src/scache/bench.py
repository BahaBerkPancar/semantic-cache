"""False-hit benchmark: how often would a semantic cache reuse an answer wrongly?

For every labelled pair, a fresh cache is primed with prompt `a` and queried with prompt `b`.
A hit on a `hit` pair is a correct reuse; a hit on a `miss` pair is a false hit. Thresholds are
swept 0.70-0.98 with cosine only and with cosine plus guards (guards + user scoping).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .cache import Context, Embedder, InMemoryBackend, SemanticCache
from .guards import GlobalScoper, Scoper, default_guards, run_guards

THRESHOLDS = [i / 100 for i in range(70, 99)]
MODES = ("cosine", "cosine+guards")
SEED = 20261117
MAX_FALSE_HIT = 0.05
TOLERANCE = 0.005


@dataclass(frozen=True)
class Pair:
    id: str
    a: str
    b: str
    label: str  # "hit" or "miss"
    type: str
    user_a: str | None = None
    user_b: str | None = None


def load_pairs(path: Path) -> list[Pair]:
    rows = yaml.safe_load(Path(path).read_text())
    return [Pair(**r) for r in rows]


def pair_texts(pairs: Sequence[Pair]) -> list[str]:
    return sorted({t for p in pairs for t in (p.a, p.b)})


def split_of(pair_id: str) -> str:
    """50/50 split by hash of the pair id: 'val' or 'test'. Stable across runs and machines."""
    return "val" if int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % 2 == 0 else "test"


def predict(
    pairs: Sequence[Pair],
    embedder: Embedder,
    guarded: bool,
    thresholds: Sequence[float] = THRESHOLDS,
) -> np.ndarray:
    """(n_pairs, n_thresholds) bool: would the cache serve prompt b from cached prompt a?"""
    out = np.zeros((len(pairs), len(thresholds)), dtype=bool)
    for i, p in enumerate(pairs):
        for j, t in enumerate(thresholds):
            cache = SemanticCache(
                InMemoryBackend(),
                embedder,
                t,
                guards=default_guards() if guarded else (),
                scoper=Scoper() if guarded else GlobalScoper(),
            )
            cache.put(p.a, "cached answer", Context(user_id=p.user_a))
            out[i, j] = cache.get(p.b, Context(user_id=p.user_b)).hit
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (never zero-width for small n)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def guards_allow(pair: Pair) -> tuple[bool, str]:
    """Threshold-independent guard verdict for a pair (scoping + guards). Returns (allow, guard)."""
    scoper = Scoper()
    sa = scoper.scope(pair.a, "bench", pair.user_a)
    sb = scoper.scope(pair.b, "bench", pair.user_b)
    if sa is None or sb is None or sa != sb:
        return False, "scoper"
    v = run_guards(pair.b, pair.a, default_guards())
    return v.allow, "" if v.allow else v.reason.split(":")[0]


def select_threshold(
    thresholds: Sequence[float],
    hit_rate: Sequence[float],
    false_hit_rate: Sequence[float],
    max_false: float = MAX_FALSE_HIT,
) -> tuple[float, bool]:
    """Pick a threshold from validation numbers only.

    Highest hit rate among thresholds with false-hit rate <= max_false (ties: higher threshold).
    If none qualifies, fall back to the lowest false-hit rate (ties: higher hit rate, then higher
    threshold) and report feasible=False.
    """
    ok = [i for i, f in enumerate(false_hit_rate) if f <= max_false + 1e-12]
    if ok:
        best = max(ok, key=lambda i: (hit_rate[i], thresholds[i]))
        return thresholds[best], True
    best = min(
        range(len(thresholds)),
        key=lambda i: (false_hit_rate[i], -hit_rate[i], -thresholds[i]),
    )
    return thresholds[best], False


def _r(x: float) -> float:
    return round(float(x), 3)


def _groups(pairs: Sequence[Pair]) -> dict[str, np.ndarray]:
    labels = np.array([p.label for p in pairs])
    types = np.array([p.type for p in pairs])
    groups = {"hit": labels == "hit", "miss": labels == "miss"}
    for t in sorted(set(types)):
        groups[t] = types == t
    return groups


def run_bench(
    pairs: Sequence[Pair],
    embedder: Embedder,
    *,
    seed: int = SEED,
    adversarial: Sequence[Pair] = (),
    max_false: float = MAX_FALSE_HIT,
    meta: dict | None = None,
) -> dict:
    pairs = list(pairs)
    groups = _groups(pairs)
    label_of = {p.type: p.label for p in pairs}
    split = np.array([split_of(p.id) for p in pairs])
    preds = {
        "cosine": predict(pairs, embedder, guarded=False),
        "cosine+guards": predict(pairs, embedder, guarded=True),
    }
    res: dict = {
        "meta": {
            **(meta or {}),
            "n_pairs": len(pairs),
            "n_hit_pairs": int(groups["hit"].sum()),
            "n_miss_pairs": int(groups["miss"].sum()),
            "n_validation": int((split == "val").sum()),
            "n_test": int((split == "test").sum()),
            "seed": seed,
            "ci": "Wilson 95%",
            "max_false_hit_rate": max_false,
            "selection": "highest hit rate on validation with false-hit rate <= max",
        },
        "thresholds": THRESHOLDS,
        "group_labels": {"hit": "hit", "miss": "miss", **label_of},
        "sweep": {},
        "selection": {},
    }
    for mode, pred in preds.items():
        sweep = {}
        for g, mask in groups.items():
            ks, n = pred[mask].sum(axis=0), int(mask.sum())
            cis = [wilson(int(k), n) for k in ks]
            sweep[g] = {
                "rate": [_r(k / n) for k in ks],
                "lo": [_r(c[0]) for c in cis],
                "hi": [_r(c[1]) for c in cis],
            }
        res["sweep"][mode] = sweep

        val = split == "val"
        v_hit = pred[groups["hit"] & val].mean(axis=0)
        v_false = pred[groups["miss"] & val].mean(axis=0)
        thr, feasible = select_threshold(THRESHOLDS, list(v_hit), list(v_false), max_false)
        j = THRESHOLDS.index(thr)
        test = split == "test"
        per_group = {}
        for g, mask in groups.items():
            m = mask & test
            n, k = int(m.sum()), int(pred[m][:, j].sum())
            lo, hi = wilson(k, n)
            per_group[g] = {"n": n, "rate": _r(k / n), "lo": _r(lo), "hi": _r(hi)}
        res["selection"][mode] = {
            "threshold": thr,
            "feasible": feasible,
            "validation": {"hit_rate": _r(v_hit[j]), "false_hit_rate": _r(v_false[j])},
            "test": per_group,
        }
    if adversarial:
        res["adversarial"] = _adversarial(list(adversarial), embedder, res)
    return res


def _adversarial(adv: list[Pair], embedder: Embedder, res: dict) -> dict:
    """Held-out hit pairs: where do guards misfire? (number-word/quarter/code categories were written alongside the guard fixes; see README.)"""
    verdicts = [guards_allow(p) for p in adv]
    types = sorted({p.type for p in adv})

    def rate(ks: int, n: int) -> dict:
        lo, hi = wilson(ks, n)
        return {"n": n, "rate": _r(ks / n), "lo": _r(lo), "hi": _r(hi)}

    deny = [not v[0] for v in verdicts]
    out: dict = {
        "n": len(adv),
        "guard_false_deny": rate(sum(deny), len(adv)),
        "guard_false_deny_by_type": {
            t: rate(sum(d for d, p in zip(deny, adv, strict=True) if p.type == t),
                    sum(p.type == t for p in adv))
            for t in types
        },
        "denied_by": dict(Counter(v[1] for v in verdicts if not v[0])),
        "hit_rate_at_chosen": {},
    }
    for mode in MODES:
        thr = res["selection"][mode]["threshold"]
        pred = predict(adv, embedder, guarded=mode != "cosine", thresholds=[thr])[:, 0]
        out["hit_rate_at_chosen"][mode] = {"threshold": thr, **rate(int(pred.sum()), len(adv))}
    return out


# ---- reports -----------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _cell(d: dict) -> str:
    return f"{_pct(d['rate'])} [{_pct(d['lo'])}, {_pct(d['hi'])}]"


def render_markdown(res: dict) -> str:
    m, th = res["meta"], res["thresholds"]
    sel, sw = res["selection"], res["sweep"]
    lab = res["group_labels"]
    c, g = "cosine", "cosine+guards"
    out = [
        "# Benchmark results",
        "",
        (f"Embedding model `{m.get('model', '?')}` ({m.get('dims', '?')} dims), "
        f"{m['n_pairs']} pairs ({m['n_hit_pairs']} hit / {m['n_miss_pairs']} miss). "
        f"Threshold chosen on the validation split ({m['n_validation']} pairs, split by pair-id "
        f"hash), reported on the test split ({m['n_test']} pairs). "
        f"Wilson 95% CIs."),
        "",
        "## Chosen thresholds",
        "",
        f"Rule: {m['selection']} (max {_pct(m['max_false_hit_rate'])}).",
        "",
        "| Mode | Threshold | Feasible | Val hit rate | Val false-hit rate |",
        "|---|---|---|---|---|",
    ]
    for mode in (c, g):
        s = sel[mode]
        out.append(
            f"| {mode} | {s['threshold']:.2f} | {'yes' if s['feasible'] else 'no'} | "
            f"{_pct(s['validation']['hit_rate'])} | {_pct(s['validation']['false_hit_rate'])} |"
        )
    out += [
        "",
        "## Test split at the chosen thresholds",
        "",
        ("Hit rate on `hit` pairs (higher is better), false-hit rate on `miss` pairs (lower is "
        "better). Each mode uses its own chosen threshold."),
        "",
        "| Group | Kind | n | cosine only | cosine + guards |",
        "|---|---|---|---|---|",
    ]
    for grp in sel[c]["test"]:
        name = {"hit": "ALL hit pairs", "miss": "ALL miss pairs"}.get(grp, grp)
        kind = "hit rate" if lab[grp] == "hit" else "false-hit rate"
        out.append(
            f"| {name} | {kind} | {sel[c]['test'][grp]['n']} | {_cell(sel[c]['test'][grp])} | "
            f"{_cell(sel[g]['test'][grp])} |"
        )
    tg = th.index(sel[g]["threshold"])
    out += [
        "",
        f"## Same threshold ({sel[g]['threshold']:.2f}), all pairs: what the guards buy and cost",
        "",
        "| Group | Kind | cosine only | cosine + guards |",
        "|---|---|---|---|",
    ]
    for grp in sw[c]:
        name = {"hit": "ALL hit pairs", "miss": "ALL miss pairs"}.get(grp, grp)
        kind = "hit rate" if lab[grp] == "hit" else "false-hit rate"
        out.append(
            f"| {name} | {kind} | {_pct(sw[c][grp]['rate'][tg])} | "
            f"{_pct(sw[g][grp]['rate'][tg])} |"
        )
    adv = res.get("adversarial")
    if adv:
        fd = adv["guard_false_deny"]
        out += [
            "",
            "## Adversarial hits: where the guards misfire",
            "",
            (
                f"{adv['n']} extra `hit` pairs written independently of the guard vocabulary "
                "(number words, version words, quarters, date formats, time and negation "
                "synonyms, entity plurals, alphanumeric codes). They are not part of the sweep "
                "or the threshold selection. Guard false-deny rate (guards refuse a reuse that "
                f"is fine): {_cell(fd)}. Denied by: {adv['denied_by']}."
            ),
            "",
            "| Category | n | Guard false-deny rate |",
            "|---|---|---|",
        ]
        for t, d in adv["guard_false_deny_by_type"].items():
            out.append(f"| {t} | {d['n']} | {_cell(d)} |")
        out += ["", "| Mode | Threshold | Hit rate on adversarial hits |", "|---|---|---|"]
        for mode, d in adv["hit_rate_at_chosen"].items():
            out.append(f"| {mode} | {d['threshold']:.2f} | {_cell(d)} |")
    out += [
        "",
        "## Threshold sweep, all pairs",
        "",
        ("| Threshold | Hit rate (cosine) | False-hit rate (cosine) | Hit rate (+guards) | "
        "False-hit rate (+guards) |"),
        "|---|---|---|---|---|",
    ]
    for i, t in enumerate(th):
        out.append(
            f"| {t:.2f} | {_pct(sw[c]['hit']['rate'][i])} | {_pct(sw[c]['miss']['rate'][i])} | "
            f"{_pct(sw[g]['hit']['rate'][i])} | {_pct(sw[g]['miss']['rate'][i])} |"
        )
    out += ["", "![hit rate vs false-hit rate](curve.svg)", ""]
    return "\n".join(out)


def render_svg(res: dict) -> str:
    """Hit rate (y) against false-hit rate (x) over the threshold sweep, all pairs."""
    w, h, left, bottom, top, right = 560, 420, 60, 50, 30, 20
    pw, ph = w - left - right, h - top - bottom
    sw, sel = res["sweep"], res["selection"]

    def xy(fx: float, fy: float) -> tuple[float, float]:
        return left + fx * pw, top + (1 - fy) * ph

    colors = {"cosine": "#c2410c", "cosine+guards": "#1d4ed8"}
    parts = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        'font-family="sans-serif" font-size="12">'),
        f'<rect width="{w}" height="{h}" fill="white"/>',
    ]
    for k in range(6):
        v = k / 5
        x, _ = xy(v, 0)
        _, y = xy(0, v)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + ph}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + ph + 16}" text-anchor="middle">{int(v * 100)}%</text>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>')
    parts.append(
        f'<text x="{left + pw / 2}" y="{h - 8}" text-anchor="middle">false-hit rate (miss pairs)</text>'
    )
    parts.append(
        f'<text transform="translate(14 {top + ph / 2}) rotate(-90)" text-anchor="middle">'
        "hit rate (hit pairs)</text>"
    )
    for mode, col in colors.items():
        pts = [
            xy(f, hh)
            for hh, f in zip(sw[mode]["hit"]["rate"], sw[mode]["miss"]["rate"], strict=True)
        ]
        parts.append(
            f'<polyline fill="none" stroke="{col}" stroke-width="2" points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            + '"/>'
        )
        j = res["thresholds"].index(sel[mode]["threshold"])
        x, y = pts[j]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}"/>')
        parts.append(
            f'<text x="{x + 8:.1f}" y="{y - 6:.1f}" fill="{col}">t={sel[mode]["threshold"]:.2f}</text>'
        )
    for k, (mode, col) in enumerate(colors.items()):
        yy = top + 12 + 16 * k
        parts.append(f'<line x1="{left + 10}" y1="{yy}" x2="{left + 34}" y2="{yy}" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{left + 40}" y="{yy + 4}">{mode}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_reports(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    (out_dir / "results.md").write_text(render_markdown(res))
    (out_dir / "curve.svg").write_text(render_svg(res))


def diff_results(old, new, tol: float = TOLERANCE, path: str = "") -> list[str]:
    """Differences between two results trees; numbers may differ by at most `tol`."""
    if isinstance(old, dict) and isinstance(new, dict):
        diffs = []
        for k in sorted(set(old) | set(new)):
            if k not in old or k not in new:
                diffs.append(f"{path}/{k}: missing on one side")
            else:
                diffs += diff_results(old[k], new[k], tol, f"{path}/{k}")
        return diffs
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: length {len(old)} vs {len(new)}"]
        return [d for i, (a, b) in enumerate(zip(old, new, strict=True))
                for d in diff_results(a, b, tol, f"{path}[{i}]")]
    if isinstance(old, bool) or isinstance(new, bool) or isinstance(old, str):
        return [] if old == new else [f"{path}: {old!r} vs {new!r}"]
    if isinstance(old, int | float) and isinstance(new, int | float):
        return [] if abs(old - new) <= tol else [f"{path}: {old} vs {new}"]
    return [] if old == new else [f"{path}: {old!r} vs {new!r}"]
