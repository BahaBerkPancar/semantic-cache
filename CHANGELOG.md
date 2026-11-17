# Changelog

## Unreleased
- Redis backend (redis-stack, HNSW cosine, scope pre-filter, TTL, tag and namespace invalidation)
  behind the same interface as the in-memory one; `compose.yaml` on 127.0.0.1:6390.
- FastAPI proxy: `/v1/complete`, `/v1/invalidate`, `/v1/feedback`, `/v1/stats`, `/health`; fake or
  local Ollama generator; configurable never-cache policy. Ids are validated and case-sensitive;
  feedback and invalidate are namespace-scoped. The API has no auth: run it behind a gateway.
- Labelled pair set (360 synthetic pairs, 10 types), embedding cache, in-memory semantic cache
  with guards (numbers, entities, negation, time scope, personal/user scoping).
- `scache embed` and `scache bench` (threshold sweep, bootstrap CIs, validation-chosen threshold,
  `--check`), results in `bench/`.
- Held-out adversarial hit set (51 pairs) and guard false-deny report; Wilson CIs.
- Removed the template's items model, Postgres and Alembic.
- Project created.
