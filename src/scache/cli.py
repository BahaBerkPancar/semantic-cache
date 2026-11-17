"""`scache` command line: embed, bench, traffic (generate a log) and replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import bench, replay, traffic
from .embeddings import (
    DEFAULT_MODEL,
    DEFAULT_PREFIX,
    OLLAMA_URL,
    EmbeddingStore,
    OllamaEmbedder,
    StaleEmbeddingsError,
    StoreEmbedder,
    build_store,
)


def cmd_embed(args: argparse.Namespace) -> int:
    pairs = bench.load_pairs(Path(args.pairs)) + bench.load_pairs(Path(args.adversarial))
    texts = bench.pair_texts(pairs)
    if args.traffic and Path(args.traffic).exists():
        texts = sorted(set(texts) | set(traffic.traffic_texts(traffic.load_traffic(args.traffic))))
    out = Path(args.out)
    existing = EmbeddingStore.load(out) if out.exists() else None
    embedder = OllamaEmbedder(args.model, args.url, DEFAULT_PREFIX, batch=16)
    store, n_new = build_store(texts, embedder, existing)
    if n_new or existing is None or existing.texts != store.texts:
        store.save(out)
    print(f"{len(texts)} unique texts, {n_new} embedded, {len(texts) - n_new} reused -> {out}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    pairs = bench.load_pairs(Path(args.pairs))
    adv = bench.load_pairs(Path(args.adversarial))
    try:
        store = EmbeddingStore.load(Path(args.embeddings))
        embedder = StoreEmbedder(store)
        embedder.embed(bench.pair_texts(pairs + adv))  # fail early if stale
    except (FileNotFoundError, StaleEmbeddingsError) as e:
        print(f"error: {e}. Run `scache embed` first.", file=sys.stderr)
        return 2
    meta = {"model": store.model, "dims": store.dims, "prefix": store.prefix}
    res = bench.run_bench(pairs, embedder, max_false=args.max_false_hit, meta=meta, adversarial=adv)
    out_dir = Path(args.out)
    if args.check:
        old = json.loads((out_dir / "results.json").read_text())
        diffs = bench.diff_results(old, res, args.tolerance)
        if diffs:
            print(f"bench check FAILED ({len(diffs)} differences), first:", file=sys.stderr)
            for d in diffs[:10]:
                print("  " + d, file=sys.stderr)
            return 1
        print(f"bench check OK (tolerance {args.tolerance})")
        return 0
    bench.write_reports(res, out_dir)
    for mode, s in res["selection"].items():
        t = s["test"]
        print(
            f"{mode}: threshold {s['threshold']:.2f}"
            f"{'' if s['feasible'] else ' (no threshold met the false-hit limit)'}; "
            f"test hit rate {t['hit']['rate']:.3f}, false-hit rate {t['miss']['rate']:.3f}"
        )
    print(f"wrote {out_dir}/results.json, results.md, curve.svg")
    return 0


def cmd_traffic(args: argparse.Namespace) -> int:
    exclude = set(
        bench.pair_texts(bench.load_pairs(Path(args.pairs)) + bench.load_pairs(Path(args.adversarial)))
    )
    reqs = traffic.generate(args.n, args.seed, exclude)
    traffic.dump_traffic(reqs, Path(args.out))
    print(f"{len(reqs)} requests, {len(traffic.traffic_texts(reqs))} distinct prompts -> {args.out}")
    for kind, n in traffic.mix_of(reqs).items():
        print(f"  {kind}: {n}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    try:
        reqs = traffic.load_traffic(Path(args.traffic))
        store = EmbeddingStore.load(Path(args.embeddings))
        embedder = StoreEmbedder(store)
        embedder.embed(traffic.traffic_texts(reqs))  # fail early if stale
        thresholds = replay.load_thresholds(Path(args.results))
    except (FileNotFoundError, StaleEmbeddingsError) as e:
        print(f"error: {e}. Run `scache traffic` and `scache embed` first.", file=sys.stderr)
        return 2
    meta = {"model": store.model, "dims": store.dims, "prefix": store.prefix}
    res = replay.run_replay(reqs, embedder, thresholds, ttl=args.ttl, meta=meta)
    out_dir = Path(args.out)
    if args.check:
        old = json.loads((out_dir / "replay.json").read_text())
        diffs = bench.diff_results(
            {k: v for k, v in old.items() if k != "timing"},
            {k: v for k, v in res.items() if k != "timing"},
            args.tolerance,
        )
        if not diffs:
            res["timing"] = old["timing"]  # measured, so carried over; the rest must reproduce
            if replay.render_markdown(res) != (out_dir / "replay.md").read_text():
                diffs = ["replay.md differs from the report rendered from the recomputed numbers"]
        if diffs:
            print(f"replay check FAILED ({len(diffs)} differences), first:", file=sys.stderr)
            for d in diffs[:10]:
                print("  " + d, file=sys.stderr)
            return 1
        print(f"replay check OK (tolerance {args.tolerance})")
        return 0
    replay.write_reports(res, out_dir)
    for m in res["modes"].values():
        print(
            f"{m['label']}: hit {100 * m['hit_rate']:.1f}%, wrong {m['wrong_hits']} "
            f"({m['wrong_per_1000']:.1f}/1000), leaks {m['leaks']}, "
            f"cost saved {m['cost_saved_pct']:.1f}%"
        )
    print(f"wrote {out_dir}/replay.json, replay.md, replay.svg")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scache", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="embed pair texts with local Ollama into data/embeddings.npz")
    e.add_argument("--pairs", default="data/pairs.yaml")
    e.add_argument("--adversarial", default="data/adversarial_hits.yaml")
    e.add_argument("--out", default="data/embeddings.npz")
    e.add_argument("--traffic", default="data/traffic.jsonl", help="also embed these prompts")
    e.add_argument("--model", default=DEFAULT_MODEL)
    e.add_argument("--url", default=OLLAMA_URL)
    e.set_defaults(fn=cmd_embed)

    b = sub.add_parser("bench", help="run the threshold sweep and write bench/results.*")
    b.add_argument("--pairs", default="data/pairs.yaml")
    b.add_argument("--adversarial", default="data/adversarial_hits.yaml")
    b.add_argument("--embeddings", default="data/embeddings.npz")
    b.add_argument("--out", default="bench")
    b.add_argument("--max-false-hit", type=float, default=bench.MAX_FALSE_HIT)
    b.add_argument("--check", action="store_true", help="compare with bench/results.json")
    b.add_argument("--tolerance", type=float, default=bench.TOLERANCE)
    b.set_defaults(fn=cmd_bench)

    t = sub.add_parser("traffic", help="generate the synthetic traffic log data/traffic.jsonl")
    t.add_argument("--out", default="data/traffic.jsonl")
    t.add_argument("--n", type=int, default=traffic.N_REQUESTS)
    t.add_argument("--seed", type=int, default=traffic.SEED)
    t.add_argument("--pairs", default="data/pairs.yaml")
    t.add_argument("--adversarial", default="data/adversarial_hits.yaml")
    t.set_defaults(fn=cmd_traffic)

    r = sub.add_parser("replay", help="replay the traffic log and write bench/replay.*")
    r.add_argument("--traffic", default="data/traffic.jsonl")
    r.add_argument("--embeddings", default="data/embeddings.npz")
    r.add_argument("--results", default="bench/results.json", help="M1 results: thresholds")
    r.add_argument("--out", default="bench")
    r.add_argument("--ttl", type=float, default=replay.TTL_S, help="entry lifetime in seconds")
    r.add_argument("--check", action="store_true", help="compare with bench/replay.json and .md")
    r.add_argument("--tolerance", type=float, default=replay.TOLERANCE)
    r.set_defaults(fn=cmd_replay)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
