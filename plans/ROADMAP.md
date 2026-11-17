# Build plan: semantic-cache

Written 2026-11-17 (away run). Source: vault `knowledge/project-ideas-backlog.md`, "Saved for later":
"Semantic Caching Layer: reuse answers for similar questions to cut repeated LLM calls, response time
and cost. Stack: Python, Redis, RedisVL, FastAPI. Thin; RedisVL does most of the work."

## How it stands out (not thin)
The hard part of semantic caching is **wrong hits**: "What is the refund window for EU orders?" and
"...for US orders?" are near-identical in embedding space but need different answers. This project
measures and controls that:
- a labelled pair set (paraphrases that SHOULD hit vs near-misses that must NOT: different entity,
  number, negation, time, user-specific context) and a benchmark of hit rate vs false-hit rate across
  thresholds, per pair type;
- guards beyond cosine similarity: entity/number/negation consistency checks, per-namespace and
  per-user scoping, TTL and explicit invalidation by tag, and a "never cache" policy for
  personal/time-sensitive prompts;
- an honest cost/latency report from a replayed traffic log.
Embeddings come from local Ollama `nomic-embed-text` (cached to disk for tests); answers from a fake
or local `qwen2.5:3b`. Storage: Redis (redis-stack in Docker, port 6390) with a pure-Python in-memory
backend for tests.

## Ground rules
- No paid APIs; Ollama on 127.0.0.1:11434 only; tests use cached embeddings / a fake embedder.
- Never read `.env`. Redis container bound to 127.0.0.1:6390 only, own compose project.
- Builders: Codex (Sonnet fallback); reviews by the `reviewer` agent; hub commits locally; created
  `--local`, no GitHub repo until MF says.

## Milestones
| # | Name | Contents |
|---|------|----------|
| M1 | Pair set + cache core + benchmark | Labelled pairs (≥ 300, typed), embedding cache, in-memory vector backend, cosine lookup, guard checks, threshold sweep benchmark (hit rate / false-hit rate per type, with CIs), README results. |
| M2 | Redis backend + API | RedisVL/redis-stack backend behind the same interface, FastAPI proxy (`/v1/complete` with cache), namespaces, TTL, tag invalidation, never-cache policy, metrics endpoint. |
| M3 | Traffic replay + showcase | Synthetic traffic log with repeats/paraphrases/near-misses, replay report (cost saved, latency, false hits served), README. |
