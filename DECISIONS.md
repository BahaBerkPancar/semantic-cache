# Decisions

Newest first. One entry per decision: date, what was decided, why, and what was rejected.

## 2026-11-17 - M1 design choices
- Pairs were written by hand (lists in `scripts/make_pairs.py`) before any similarity score was
  computed; guards and gazetteer were written before the first benchmark run. After review the
  personal detector was widened (first person subject/object forms) and number handling fixed
  (decimals, codes, number words, order); the main pair set was not changed.
- A separate held-out set of 51 "adversarial hits" (`data/adversarial_hits.yaml`) was written to
  probe guard false-denies; it is reported separately and never used for threshold selection. The
  nit fixes to the number guard (words 13-19/70/80, quarters, codes) came before it was written; the
  guards were not tuned on it afterwards.
- Confidence intervals are Wilson 95% intervals, not bootstrap: the bootstrap gives zero-width
  intervals at 0% or 100% rates, which is misleading for groups of 15-30 pairs.
- Scope keys are `(namespace, user_id)` tuples; empty ids are rejected.
- Typo/case/reorder pairs are separate hit types (4 hit types, 6 miss types, 180 each overall).
  Six entity-swap pairs use countries outside the gazetteer on purpose.
- Non-personal pairs run as one user (`user-1`), personal `miss` pairs as two different users.
  Cosine-only mode uses a global scope (no user scoping); guard mode uses guards plus scoping.
- Threshold rule: highest hit rate on validation with false-hit rate <= 5%; if none qualifies, lowest
  false-hit rate. Cosine-only is infeasible (identical personal pairs always hit), so it lands on
  0.98. Guarded mode lands on 0.79 (val false-hit 4.6%, test 7.5%). The sweep and curve use all pairs (descriptive); headline numbers use the test split only.
- Guards only deny; all five run on every candidate, the next candidate is tried if the best one is
  denied. Personal prompts are user-scoped, and never cached without a user id.
- Embeddings are committed (`data/embeddings.npz`, ~1.9 MB) so tests and `bench --check` need no
  Ollama. FakeEmbedder (trigram hashing) is used in unit tests.
- Dropped Postgres, SQLAlchemy and Alembic from the template; the package is `scache`. Hand-written
  SVG instead of matplotlib (no extra dependency).

## 2026-11-17 – M2: Redis backend and API
- **Raw redis-py + FT.SEARCH, not RedisVL.** The schema is one HASH, one index, one KNN query;
  RedisVL would add a dependency (and its own schema layer) for roughly 60 lines. Easier to see
  exactly which filter runs.
- **HNSW, COSINE, FLOAT32.** Scales past brute force; at M2's sizes FLAT would be as good. The
  index is created with the embedder's dims and refuses to open with different dims.
- **Filter strategy: pre-filter inside KNN.** Query is `(@ns:{..} @user:{..} @expires_at:[(now +inf])
  =>[KNN k ...]`, so other scopes cannot leak in or crowd out the top-k (test: 30 closer entries in
  another namespace). Users are stored as `u:<id>`, "no user" as `none`; TAG values are escaped.
- **TTL twice.** `expires_at` is filtered against the cache clock (exact, testable with a fake
  clock); the key also gets PEXPIRE ttl+1s so Redis garbage-collects. Entries without TTL use a
  finite sentinel (9e15) because NUMERIC fields cannot hold inf.
- **Ids are validated, TAG fields are case-sensitive (review fix).** Namespace, user id and tags
  must match `[A-Za-z0-9_.:@-]{1,128}` (API and cache layer). TAG fields use `|` as separator (not
  in the charset) and CASESENSITIVE, so `x,u:victim` cannot index as two tags and Bob/bob,
  Acme/acme, Billing/billing stay apart, as in the in-memory backend. `candidates()` also drops any
  document whose decoded (ns, user) is not exactly the requested scope. An index created by the
  earlier version (case-insensitive) is refused with a clear error: drop it or change the prefix.
- **The API trusts its caller.** No auth; `user_id` is taken from the body. It is meant to sit
  behind a gateway that sets namespace and user from the authenticated identity, on 127.0.0.1.
  Feedback and invalidate require a namespace (feedback: also the user for personal entries) so a
  call can only touch its own scope; tag invalidation is per namespace. A caller that can pick any
  namespace can still wipe it, which is why auth belongs in front.
- **Vectors are fetched by key** after the search: binary fields in the FT.SEARCH reply get mangled
  by response decoding.
- **Never-cache** lives in `policy.py`, separate from the scoper: no_cache and personal-without-user
  (also enforced by the scoper), volatile time words (today/yesterday/tomorrow/now) and ISO dates,
  error-looking answers. Such requests are answered with `cache: bypass`: no lookup, no store.
  Month names and "last week" are left to the time-scope guard, not blocked outright.
- **API defaults are fake** (embedder and generator) so it runs with no model; threshold 0.79 is the
  M1 guarded choice. A lock serialises cache access (the in-memory backend is not thread-safe);
  generation happens outside it. Stats are in-process counters, not stored in Redis.
- **Feedback** counts every `wrong:true` report, and evicts the entry if it still exists and belongs to the given namespace (and, for a personal entry, the given user).

## 2026-11-17 – Project created from the `fastapi-app` template
Why: same layout, test setup and CI as the other projects, so nothing has to be set up by hand.

## 2026-11-18 – M3: traffic replay
- **Hidden intent ids, correct = same intent.** The generator builds every request from an intent
  catalogue (slot families: region/plan, number, negation pairs, weekday/month/relative time, FAQ),
  so labels are exact by construction. Cost: they encode the author's idea of "same question", and
  the templates sit close to the guards' vocabulary, so guard numbers are optimistic.
- **Personal and time-sensitive intents.** A personal prompt is a different intent per user; a
  time-sensitive prompt is a new intent each day. That makes cross-user leaks and stale hits
  countable without a model.
- **Threshold from M1, two cosine-only rows.** Guarded variants use M1's chosen 0.79. Cosine-only
  is shown at its own M1 choice (0.98, which met no false-hit target) and at 0.79, since 0.98 is
  almost exact match and the 0.79 row is the fair "what do the guards buy" comparison.
- **No re-tuning on traffic**, and nothing in the replay reads the traffic to pick a threshold
  (a test checks the value comes from the results file).
- **Answer = intent id.** The replay stores the request's intent id as the "answer", so no model is
  called; correctness is an equality check. Wrong hits are not stored (as in a real cache), so a
  repeated near-miss is served wrongly again; this inflates cosine-only wrong counts.
- **Never-cache policy as its own row.** `SemanticCache` does not know the policy (it lives in the
  API layer), so the replay applies `NeverCachePolicy` before lookup and store.
- **Cost and latency are modelled, not measured.** Illustrative price table and latency formula so
  `--check` is deterministic; the measured lookup time is reported separately and excluded from
  `--check`. `--check` compares replay.json (tolerance 0.005) and requires replay.md to equal the
  report rendered from the recomputed numbers (timing carried over).
- **Block bootstrap**, 50-request blocks, 1,000 resamples, fixed seed, same resamples for every
  variant. It resamples the outcomes of one fixed run: the CIs show sampling noise in that run,
  ignore dependence between blocks (cache warm-up) and say nothing about different traffic.
- **Embeddings in one file.** Traffic texts go into the same `data/embeddings.npz` (stale detection
  by hash as before); replay fails early with exit code 2 if any text is missing or stale. Embedding
  calls use batches of 16.
