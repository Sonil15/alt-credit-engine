# AGENTS.md — instructions for coding agents (Cursor, Antigravity, Claude, …)

This is a hackathon project; the goal is to **win**. Read this before changing anything.

## Hard constraints
- **Local only — never push to GitHub.**
- **White-label: never name the sponsoring bank or university** anywhere (code, UI,
  README, comments, commits). Use "the bank" / "a public-sector bank" / "Data Fiduciary".

## Pitch-framing protocol (REQUIRED after significant changes)
After any *significant* change — a new feature, a model/architecture/decision change, a
fixed credibility risk, or a defensible engineering decision — **add or update the
matching entry in [`PITCH.md`](PITCH.md)** so the user has judge-facing framing for it.

- Follow the **Framing Recipe** and **Maintenance Protocol** defined in `PITCH.md`.
- *Significant only* — skip typos, pure refactors, formatting, and dep bumps.
- If your change makes an existing `PITCH.md` entry inaccurate, fix that entry.

## Where to look
- `PITCH.md` — judge-facing framing for every significant feature (the win condition).
- `mlmodel.md` — the ML strategy (glass-box EBM champion; SHAP is removed from the
  decision/explanation path — don't reintroduce SHAP framing).
- `README.md` — project overview and run instructions.

## Running & verifying locally (avoid these known blockers)
Use the `alt-credit-demo` launch config (`.claude/launch.json`) to preview/verify —
it sets `USE_SQLITE=true AUTO_SEED_ON_STARTUP=true SIMULATE_ALL_FACETS=true` and runs
`.venv/bin/uvicorn`. `SIMULATE_ALL_FACETS` matters specifically for anything touching
consent/scope gating: it's what makes ingestion synthesize telecom/e-commerce/geo/
cashflow facets in the first place, so a revoked scope has something to actually gate.

If you write a one-off script to exercise the score engine directly (instead of hitting
the running server), you will hit every one of these — in order:
1. **Plain `python` fails on import.** The system/conda interpreter doesn't have this
   project's deps (`interpret`, `catboost`, etc.). Always use `.venv/bin/python`.
2. **`ModuleNotFoundError: No module named 'core'`** — the repo root isn't on
   `sys.path` outside of uvicorn's app-dir handling. Run with `PYTHONPATH=.`.
3. **`ModuleNotFoundError: No module named 'asyncpg'`** if `DATABASE_URL` isn't set to
   sqlite — `core/config.py` defaults to Postgres unless `USE_SQLITE=true` (or an env
   `DATABASE_URL=sqlite+...`) is present. Always set `USE_SQLITE=true` in the script's
   env before importing `core.database`.
4. **`FileNotFoundError: Champion model not trained`** — `core.model_cache` is a
   process-global cache that uvicorn populates at startup (`init_model_cache()`); a bare
   script must call `from core.model_cache import init_model_cache; init_model_cache()`
   itself before calling `score_user`/`score_all_users`.
5. **User-id format mismatch** — raw ids in `synthetic_data/mock_data_100_users.json` or
   a hand-typed sqlite query (no dashes) won't match rows from
   `core.feature_store.fetch_features_wide`, which returns dashed UUID strings. Pull the
   id from `fetch_features_wide(session).iloc[0]["user_id"]` rather than assuming a
   format.
6. **Consent state is in-process memory, not the DB** — `api/routes/consent.py` keeps
   `_active_consents` / `_revoked_users` / `_user_consent_map` as module-level dicts.
   A verification script in a separate process starts with empty consent state; to
   simulate a revoked scope, mutate those dicts directly (see
   `get_revoked_scopes` for the exact shape) rather than expecting `/consent/revoke`
   calls made through the browser/preview to be visible.
7. **Any scope you gate must be added to *every* feature it touches.** When adding/
   auditing a consent scope, cross-check `convergence.score_engine.SCOPE_TO_FEATURES`
   against `convergence.feature_meta.FEATURE_META` sources — it's easy to gate the
   obvious features (e.g. `cashflow` → `monthly_income_mean`) and miss derived ones from
   the same source (e.g. `monthly_expense_mean`, `adf_statistic`, `adf_pvalue` are all
   cash-flow-derived but lived outside the `cashflow` scope's list until this was fixed).
   A quick check: no `FEATURE_META` entry should be reachable from the model that isn't
   covered by some scope in `SCOPE_TO_FEATURES`, unless it's borrower-declared intake
   data (which consent doesn't gate).
