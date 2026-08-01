# Reset borrower account applications

Clears loan-application data for every registered borrower account (`borrower_accounts`). The synthetic demo cohort (seed users with no login) is **not** touched.

## Command

From the repo root:

```bash
USE_SQLITE=true PYTHONPATH=. .venv/bin/python -m scripts.reset_account_applications
```

Dry run (counts only, no deletes):

```bash
USE_SQLITE=true PYTHONPATH=. .venv/bin/python -m scripts.reset_account_applications --dry-run
```

## Postgres

Omit `USE_SQLITE=true` and use your normal `DATABASE_URL` (or `POSTGRES_*` env vars):

```bash
PYTHONPATH=. .venv/bin/python -m scripts.reset_account_applications
```

## What gets deleted

For each `user_id` in `borrower_accounts`:

| Table | Contents |
|-------|----------|
| `application_intake` | Onboarding / loan intent |
| `secure_vault` | Encrypted intake, survey, facet payloads |
| `ml_features` | Derived features from their applications |
| `feature_series` | Time-series features (e.g. cashflow) |
| `score_decisions` | Credit decisions |
| `decision_letters` | Decision letters |
| `audit_logs` | Officer overrides |

## What is kept

- Borrower login accounts (`borrower_accounts`)
- Auth tokens (`auth_tokens`)
- Synthetic seed data (demo borrowers without a login account)

## After running

1. **Restart the API server** if it is running - consent and assessment session state live in memory and are not cleared by this script.
2. **Browser localStorage** - the apply portal may still list old applications under `altcredit_applications` until you clear site data or use a fresh browser session.
