# Changelog

## Unreleased
- Labelled pair set (360 synthetic pairs, 10 types), embedding cache, in-memory semantic cache
  with guards (numbers, entities, negation, time scope, personal/user scoping).
- `scache embed` and `scache bench` (threshold sweep, bootstrap CIs, validation-chosen threshold,
  `--check`), results in `bench/`.
- Held-out adversarial hit set (51 pairs) and guard false-deny report; Wilson CIs.
- Removed the template's items model, Postgres and Alembic.
- Project created.
