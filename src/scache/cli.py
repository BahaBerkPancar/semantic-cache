"""`scache` command line: embed (fill the embedding cache) and bench (run the benchmark)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import bench
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
    out = Path(args.out)
    existing = EmbeddingStore.load(out) if out.exists() else None
    embedder = OllamaEmbedder(args.model, args.url, DEFAULT_PREFIX)
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scache", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="embed pair texts with local Ollama into data/embeddings.npz")
    e.add_argument("--pairs", default="data/pairs.yaml")
    e.add_argument("--adversarial", default="data/adversarial_hits.yaml")
    e.add_argument("--out", default="data/embeddings.npz")
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

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
