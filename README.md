# semantic-cache

Reuses answers for semantically similar LLM requests to cut cost and latency, with a measured
false-hit rate and safe invalidation.

## Run the cache as an HTTP proxy (M2)

Redis Stack (RediSearch HNSW, cosine) on `127.0.0.1:6390`, then the API on `127.0.0.1:8050`:

```sh
docker compose up -d --wait
SCACHE_REDIS_URL=redis://127.0.0.1:6390 uv run uvicorn scache.main:app --port 8050
# without SCACHE_REDIS_URL the in-memory backend is used
```

Settings (env): `SCACHE_REDIS_URL`, `SCACHE_REDIS_PREFIX`, `SCACHE_THRESHOLD` (0.79),
`SCACHE_EMBEDDER` and `SCACHE_GENERATOR` (`fake` default, or `ollama`: `nomic-embed-text` and
`qwen2.5:3b` on 127.0.0.1:11434), `SCACHE_DEFAULT_TTL` (seconds). The fake embedder is a trigram
hash, only good for trying the API; use `ollama` for real semantics.

```sh
H='Content-Type: application/json'
curl -s localhost:8050/v1/complete -H "$H" -d '{"prompt":"What is the refund window for EU orders?","namespace":"shop","tags":["policy"],"ttl":3600}'
# -> {"id":"...","cache":"miss",...}; repeat it -> "cache":"hit","similarity":1.0
curl -s localhost:8050/v1/complete -H "$H" -d '{"prompt":"What is the refund window for US orders?","namespace":"shop"}'
# -> miss, reasons: ["entity_guard: entities differ ..."]
curl -s localhost:8050/v1/complete -H "$H" -d '{"prompt":"Where is my order?","namespace":"shop","user_id":"u1"}'
curl -s localhost:8050/v1/feedback -H "$H" -d '{"id":"<id of a bad hit>","namespace":"shop","wrong":true}'  # evicts + counts
curl -s localhost:8050/v1/invalidate -H "$H" -d '{"namespace":"shop","tag":"policy"}'    # omit tag: whole namespace
curl -s localhost:8050/v1/stats
docker compose down
```

`POST /v1/complete` returns `cache: hit | miss | bypass`, `similarity`, `reasons` (guard denials,
never-cache reasons), `latency_ms` and the entry `id`. Never cached (answered with `bypass` or not
stored): `no_cache: true`, personal prompts without `user_id`, prompts with today/now/tomorrow or an
ISO date, and answers that look like errors. Each rule is a switch on `NeverCachePolicy`.
Personal prompts are stored per `user_id`; everything is partitioned by `namespace`.

**Security: the API trusts its caller.** There is no authentication. `user_id` and `namespace` come
from the request body, so anyone who can reach the port can claim any user, read that user's
cached personal answers, evict entries or wipe a namespace. It binds to 127.0.0.1 only; put it
behind your own auth/gateway that sets `namespace` and `user_id` from the authenticated identity,
and do not expose the port. `/v1/feedback` and `/v1/invalidate` require a `namespace` (feedback
also needs `user_id` for personal entries) and only touch entries in that scope. Namespaces, user
ids and tags are case-sensitive and limited to `[A-Za-z0-9_.:@-]`, 1-128 chars (422 otherwise).

Tests: `uv run pytest` runs without Redis (Redis tests are skipped). With the compose service up:
`SCACHE_REDIS_URL=redis://127.0.0.1:6390 uv run pytest` runs the same backend contract suite
against Redis as well.

## The problem

"What is the refund window for EU orders?" and "...for US orders?" are almost the same vector, but
they need different answers. A semantic cache that only looks at cosine similarity serves wrong
answers. M1 measures how often, offline and reproducibly, and what cheap guards fix.

## Results (M1)

360 synthetic pairs (180 `hit`: reuse is fine, 180 `miss`: must not reuse) in 10 types, embedded
with local Ollama `nomic-embed-text`, plus 51 held-out "adversarial hits" (below). Threshold chosen
on a validation split (50/50 by pair-id hash) and reported on the test split. Intervals are Wilson
95%. Full tables, the whole sweep and per-type numbers: `bench/results.md`.

| Test split | Threshold | Hit rate (hit pairs) | False-hit rate (miss pairs) |
|---|---|---|---|
| cosine only | 0.98 (no threshold met the 5% false-hit limit) | 43.8% [34.0, 54.2] | 9.7% [5.2, 17.4] |
| cosine + guards | 0.79 | 85.4% [76.6, 91.3] | 7.5% [3.7, 14.7] |

The guards' validation false-hit rate was 4.6% (limit 5%); on the test split it is 7.5%, so the
5% target is not met out of sample. The 93 test miss pairs give a wide interval.

At the same threshold (0.79, all 360 pairs) cosine alone serves a wrong answer for 85.0% of the
`miss` pairs (negation 100%, number change 100%, personal 100%, time scope 97%, entity swap 77%,
topic_near 37%) while the guards cut that to 6.1%. The guards do nothing for `topic_near`
(related but different question): 33% still slip through, and the 3% entity-swap leaks are
countries outside the gazetteer.

![hit rate vs false-hit rate](bench/curve.svg)

### What guards cost

- On the main pair set, hit rate at equal threshold: 90.0% to 87.2% (paraphrases 81.2% to 73.9%).
  This understates the cost, because those hit pairs contain almost no numbers, dates, time words
  or negations, so there was little for the guards to misfire on (and the miss pairs were built
  from the same vocabulary the guards check, so the false-hit side is optimistic too).
- Held-out adversarial hits: 51 pairs where the answer is the same but the surface differs
  ("thirty days" vs "30 days", "last week" vs "previous week", "Q3" vs "third quarter", "can't" vs
  "unable to", dates in different formats, "v3" vs "version three", entity plurals). Caveat: three
  categories (number words, quarters, alphanumeric codes) were written in the same fix round that
  taught the guards those forms, so their 0% is by construction; the other five (time synonyms,
  date formats, entity forms, negation synonyms, version words) were written without reference to
  the guard lists. The guards wrongly deny 27.5% [17.1, 40.9] of all 51 — treat that as a lower
  bound; on the independent categories alone the rate is much higher (time synonyms 100%, date
  formats 50%, entity forms 29%, version words 20%, negation synonyms 13%). Cosine only serves 39% of them at 0.98 and the
  guards 71% at 0.79, but those thresholds differ, so this is not an apples-to-apples cost.
  Treat roughly "a few points on typical traffic, a quarter on paraphrased numbers/dates/times".
- Complexity: the gazetteer is hand-maintained.

### Limits

- Pairs are synthetic, short and written by one author; numbers will differ on real traffic.
- Miss pairs reuse the vocabulary the guards check (digits differ, negators, time words), and the
  main hit pairs hardly contain such words. The guards' benefit on the miss side and their cost on
  the hit side are both therefore only bounded, not estimated; the adversarial set measures the
  cost side separately.
- Entities outside the gazetteer are not caught (6 of 30 entity-swap pairs use unknown countries).
- Personal detection is a regex (my/mine/me/we/our, "am I", "did I", "do I have", "I'm", "I've",
  "I ordered"...). It is a heuristic: paraphrases that avoid those words are treated as shared,
  and generic prompts with "me" or "we" are treated as personal (cached per user only).
  Concretely, prompts with no first-person word still leak across users: "Is the refund approved
  yet?", "Where is order 4412?" (order ids are shared), "Why was the card declined?", and bare
  "Do I qualify for the discount?". Use per-user namespaces for account-specific endpoints.
- Numbers are compared in order ("v2 to v3" differs from "v3 to v2"), so reordered questions with
  several numbers are denied. Decimals compare numerically ("1.50" = "1.5", "0.05" differs from
  "0.5"). Codes like Q3 or MP3 are whole tokens. Date formats and month names are not normalised.
- The benchmark primes a fresh cache with a single entry per pair, so it measures pairwise
  decisions, not competition between many cached entries or real traffic.
- `InMemoryBackend` is not thread-safe.
- Stale-cache detection hashes the model name, prefix and text, not the model digest: re-pulling
  `nomic-embed-text` with different weights under the same name is not detected.
- Hit pairs are single-user; cross-user sharing of non-personal answers is assumed fine.
- Small groups (about 15 per type on the test split) give wide CIs; the thresholds were selected
  from 178 validation pairs.

## Showcase: replaying a traffic log (M3)

`scache traffic` writes `data/traffic.jsonl`: 2,000 **synthetic, fictional** requests over 7 days
(a shop / SaaS support desk, seed 20261118, no text from the pair set). Each carries a hidden
`intent_id`: same intent means the same correct answer. The configured mix is 30% exact
repeats and 25% paraphrases, but the paraphrase pool (4 per intent) runs out and falls back to
exact repeats, so the log ends up as: 39% exact repeats, 14% paraphrases of
earlier intents, 17% near-misses that are new intents (region/plan swaps, number swaps, negation,
weekday/month/relative-time swaps), 9% personal prompts from 8 users, 7% time-sensitive prompts
(a new intent each day), 10% one-off long-tail prompts, 4% first asks; popularity is Zipf-like.
`scache replay` streams it in time order through the in-memory cache; thresholds come from
`bench/results.json` (M1, chosen on pair validation data, not tuned on this traffic) and entries
live 24 h on the log's clock. Full tables, wrong hits by type and a cumulative chart:
`bench/replay.md`, `bench/replay.svg`. Intervals are block-bootstrap 95%: they resample the
outcomes of this one fixed run, so they show sampling noise in it, ignore dependence between
blocks (cache warm-up) and do not show variation across different traffic.

| Variant | Hit rate | Wrong hits per 1,000 requests | Cross-user leaks | Cost saved (illustrative) |
|---|---|---|---|---|
| exact match only | 41.6% | 13.5 [9.0, 18.5] | 0 | 44.7% |
| cosine only, M1 threshold 0.98 | 46.5% | 69.5 [59.5, 80.5] | 99 | 48.9% |
| cosine only, threshold 0.79 | 76.8% | 516.0 [487.0, 544.5] | 111 | 78.2% |
| cosine + guards, threshold 0.79 | 62.9% | 78.0 [64.0, 92.0] | 0 | 65.0% |
| cosine + guards + never-cache policy | 58.1% | 53.0 [43.5, 64.5] | 0 | 60.2% |

What the guards buy and cost: at the same threshold they remove all 111 cross-user leaks and cut
wrong hits from 1,032 to 156 (a wrong answer on half of all requests becomes 8%), for 14 points of
hit rate (76.8% to 62.9%) and 13 points of saved spend. Correct hits save 25.1% of total
spend with cosine at 0.79 versus 57.6% with guards, i.e. 32% versus 89% of the savings each
variant reports (25.1/78.2, 57.6/65.0). What is still wrong with the
guards: 73 of the 156 are long-tail prompts that differ only by an app or object name the
gazetteer does not know ("Slack" vs "Jira"), 37 are stale time-sensitive answers (served within
the 24 h TTL), and some are guard-blind swaps (unknown countries, "May"). The never-cache policy
removes the stale ones at the price of 5 points of hit rate. Exact match alone is safe but only
reaches 42%, and still serves 27 stale answers.

Latency and cost use an **illustrative** model (input $1 / output $4 per million tokens;
generation 300 ms + 12 ms per output token; 25 ms per embedding), not a provider's real prices.
Measured cache lookup (search plus guards, embedding excluded) is a few hundred microseconds on
this machine, so the saving is the avoided generation; it is written to `bench/replay.md` but
not compared by `--check`.

Limits: the traffic is synthetic and written by the same author as the guards, with templates
close to the vocabulary the guards check; the intent labels hold by construction, not from human
judgement; the 8% of "wrong" served by the guards is likely optimistic for real traffic, and the
unknown-entity gap is only as wide as this catalogue makes it. A wrong hit is not stored, so a
repeated near-miss keeps being served wrongly (this inflates the cosine-only rows). The
in-memory backend ignores Redis effects (HNSW recall, network).

## Usage

```bash
uv sync
uv run scache traffic        # write data/traffic.jsonl (synthetic, seeded)
uv run scache embed          # fill data/embeddings.npz (pairs + traffic) from local Ollama (127.0.0.1:11434)
uv run scache bench          # write bench/results.{json,md} and bench/curve.svg
uv run scache bench --check  # recompute and compare with bench/results.json (tolerance 0.005)
uv run scache replay         # write bench/replay.{json,md,svg}
uv run scache replay --check # recompute; must reproduce replay.json (tolerance 0.005) and replay.md
uv run pytest && uv run ruff check .   # offline; tests never need Ollama
```

`scache embed` only embeds new or stale texts (hash of model, prefix and text); `scache bench`
refuses to run on a stale cache. Both sides use the `search_query: ` prefix.

## Layout

- `src/scache/cache.py` - `SemanticCache(backend, embedder, threshold, guards, scoper)`,
  `InMemoryBackend` (numpy brute force; interface ready for Redis in M2)
- `src/scache/guards.py` - numbers, entities, negation, time scope, personal; scoping
- `src/scache/bench.py`, `traffic.py`, `replay.py`, `cli.py`, `embeddings.py`
- `data/pairs.yaml` (generated by `scripts/make_pairs.py`), `data/embeddings.npz`, `data/traffic.jsonl`
- `plans/` - roadmap and milestone plans; `AGENTS.md` - rules for agents; `DECISIONS.md`
