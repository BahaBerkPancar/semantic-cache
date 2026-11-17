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

## 2026-11-17 – Project created from the `fastapi-app` template
Why: same layout, test setup and CI as the other projects, so nothing has to be set up by hand.
