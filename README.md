# Alt-Credit Engine

Privacy-preserving alternate credit scoring system for thin-file borrowers in India. Ingests alternative data (telecom, e-commerce, geolocation, cashflow, psychometric survey), encrypts it at rest, extracts ML features, and produces a **300–900 credit score** with SHAP explainability, adverse-action reason codes, five-pillar sub-scores, risk-based lending terms, and portfolio fairness monitoring.

## Architecture

```
AA Consent Gateway → Ingest API → AES-256 Vault → Preprocessing → ml_features + feature_series
                                                                        ↓
                                                              ECM (real time series) + CatBoost
                                                                        ↓
              Convergence (PDO scorecard + pillars + SHAP + reason codes + lending) → Audit Trail
                                                                        ↓
                                              Bank Dashboard (portfolio, model card, fairness)
```

## Key Improvements (Hackathon-Ready)

| Feature | Description |
|---------|-------------|
| **Generative ground truth** | Latent creditworthiness → Bernoulli default; model learns from noisy features, not circular rules |
| **Real econometrics** | ADF/ECM on actual monthly cashflow / telecom payment series |
| **PDO scorecard** | Log-odds to points (base 600, PDO 50, range 300–900) |
| **Model card** | Holdout AUC, Gini, KS, calibration, CV metrics at `/score/model/card` |
| **Reason codes** | Plain-language adverse action reasons from SHAP drivers |
| **Fairness report** | Disparate impact ratio across protected groups |
| **RBI AA consent & DPDP rights** | Account Aggregator-style consent with purpose, expiry, revocation; borrower can revoke (by consent ID or user ID) or request full data erasure (DPDP Act 2023 §6/§12); raw PII and ML features deleted on erasure, anonymised `score_decisions` retained 5 years per RBI audit rules; bureau-caveat disclosed at consent time |
| **Borrower privacy dashboard** | Returning borrowers visit `/consent?user_id=<id>` to see live consent status (ACTIVE/REVOKED), data presence (PRESENT/DELETED), and take action — all without re-authenticating |
| **Borrower portal & session auth** | `/apply` page lists previous applications (stored in browser); borrower can only view their own score via session token (`X-Session-Id`); cannot see portfolio overview or other borrowers' data |
| **Role-based access control** | Borrowers see only their own assessment; bank officers (with API key) see full portfolio, all scores, and admin endpoints; portfolio endpoints forbidden without authentication |
| **Five-pillar sub-scores** | Per-data-source scores (telecom, spending, location, cashflow, psychometric), 0–100, population-normalised — drives the radar chart |
| **Coherent explainability** | SHAP log-odds contributions converted to score points so the waterfall reconciles to the headline score (`base_points + Σ points ≈ score`) |
| **Risk-based lending** | Recommends max loan, risk-priced rate, tenure, and EMI per applicant |
| **Confidence / thin-file** | Data-sufficiency score from how many pillars are backed by real data; low-confidence files routed to review, never silent auto-approve |
| **Audit trail** | Every decision logged to `score_decisions` table |
| **Self-seeding startup** | Demo cohort auto-loads into the DB on first boot (idempotent) — no manual seed/load/train; optional SQLite mode for offline laptop demos |
| **Deployment** | Docker + docker-compose; live demo on Render |
| **Multilingual psychometrics** | Agent-guided assessment in EN/HI/BN with deterministic trait scoring; language is locked after first selection (other buttons hidden); Likert questions use tap-to-select buttons (text input hidden); open-ended questions show the text input; animated processing screen shown while scoring runs |

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (for the primary Postgres stack)
- Groq API key (optional; survey NLP falls back to keyword heuristics)

## Quick Start (Docker + Postgres — primary / deployed configuration)

This is the configuration used for managed Postgres hosts (e.g. Render). On first startup the app **self-seeds** the 100-borrower demo cohort (encrypt → vault → preprocess → ECM) and loads the bundled pre-trained CatBoost artifact from `models_ai/artifacts/` — no manual seed/load/train steps needed:

```bash
cp .env.example .env
docker compose up --build
```

Open http://localhost:8000/dashboard — populated automatically.

Seeding is idempotent (it skips if the database already has data), so restarts are safe. Training stays an explicit action (`POST /score/train` or `python -m models_ai.train`) so you can demo the full ECM + CatBoost pipeline live.

`SEED_ON_STARTUP` defaults to `true` in `.env.example`. Set it to `false` if you want an empty database on boot.

## Quick Start (optional — zero-dependency local laptop demo)

For a quick offline run with **no Docker and no Postgres**, set `USE_SQLITE=true`. The engine uses a local SQLite file and self-seeds on first boot:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
USE_SQLITE=true uvicorn api.main:app --port 8000
```

This is only a convenience for local demos — Postgres remains the default and the deployed backend. (A provided `DATABASE_URL`, e.g. on Render, always takes precedence.)

## Demo UI

| URL | Description |
|-----|-------------|
| http://localhost:8000/ | Welcome page — choose bank dashboard or borrower portal |
| http://localhost:8000/apply | **Borrower portal** — start a new application or view previous results (stored in browser) |
| http://localhost:8000/consent | RBI AA consent gateway (scope selection + disclosure) |
| http://localhost:8000/consent?user_id=`<id>` | Borrower privacy dashboard — check consent & data status, revoke consent, or request data erasure |
| http://localhost:8000/assessment | Multilingual agentic psychometric chat (EN/HI/BN); redirects to result page when done |
| http://localhost:8000/borrower?session=`<session_id>` | Borrower-only result page (session-authenticated); shows score, PD, decision, SHAP drivers, reason codes |
| http://localhost:8000/dashboard | Bank LOS dashboard with portfolio, pillars radar, model card, fairness, lending terms (API key required) |
| http://localhost:8000/docs | FastAPI Swagger UI |
| http://localhost:8000/consent/compliance | Regulatory compliance summary |

**Psychometric constructs measured:** conscientiousness, locus of control, financial self-efficacy, present bias, debt attitude. The agent handles conversation; a fixed rubric produces the score.

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | — | Health check + model version |
| GET | `/consent/authorize?user_id=` | — | RBI AA consent authorization; optional `user_id` links the consent artifact to the borrower for status lookups |
| POST | `/consent/token` | — | AA token exchange |
| POST | `/consent/revoke` | — | Revoke consent; accepts `consent_id` or `user_id` (or both) |
| POST | `/consent/erasure` | — | DPDP right to erasure — deletes `secure_vault`, `ml_features`, `feature_series` for the user; retains anonymised `score_decisions` |
| GET | `/consent/status/{user_id}` | — | Returns consent status (active/revoked/unknown), data presence, and erasure state for the borrower privacy dashboard |
| GET | `/consent/compliance` | — | Regulatory compliance summary including borrower rights and bureau caveat |
| POST | `/ingest/{data_type}` | — | Ingest encrypted payload (`telecom`, `ecommerce`, `geo`, `cashflow`, `survey`) |
| POST | `/ingest/ground_truth` | API Key | Store generative labels (training only) |
| GET | `/score/me` | Session | Authenticated borrower's own score (X-Session-Id header) |
| GET | `/score/{user_id}` | Session or API Key | Credit score for user_id; borrower can only access their own via their session, bank officers use API key |
| GET | `/score/` | API Key | All user scores (bank only) |
| GET | `/score/portfolio/summary` | API Key | Portfolio + fairness metrics (bank only) |
| GET | `/score/model/card` | — | Model validation metrics |
| POST | `/score/train` | API Key | Train ECM + CatBoost |
| GET | `/assessment` | — | Multilingual psychometric UI |
| POST | `/assessment/start` | — | Start agentic assessment session |
| POST | `/assessment/answer` | — | Submit answer to current item |
| GET | `/assessment/session/{session_id}` | — | Session progress and trait snapshot |
| POST | `/api/verify-live-location` | — | Compare live GPS check-in to e-commerce delivery pin history |

## Authentication & Access Control

**Borrower (session-based):**
- After completing the psychometric assessment, borrower is redirected to `/borrower?session=<session_id>`
- This session ID acts as their credential
- Call `GET /score/me` with `X-Session-Id: <session_id>` header to retrieve **only their own score**
- Session is tied to a user_id; cannot be used to access another borrower's data
- Previous applications stored in browser localStorage; no account needed

**Bank Officer (API key):**
- Requires `API_KEY` env var (set in `.env`)
- Send as `X-API-Key` header to access:
  - `GET /score/` — all borrower scores
  - `GET /score/portfolio/summary` — portfolio overview, fairness metrics
  - `GET /score/{user_id}` — any borrower's full details
  - `POST /score/train` — trigger model training
  - `POST /ingest/ground_truth` — add training labels

**What borrowers cannot see:**
- Portfolio overview (approval rates, average scores, fairness metrics)
- Other borrowers' names, scores, or data
- Admin endpoints (training, ground truth)

**What bank officers can see:**
- Entire portfolio with aggregated stats
- Individual borrower scores with full audit trail
- Model card and validation metrics
- Fairness monitoring dashboard

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SEED_ON_STARTUP` | `true` | Auto-seed demo cohort into an empty DB on boot |
| `USE_SQLITE` | `false` | Use local SQLite instead of Postgres |
| `DATABASE_URL` | — | Managed Postgres URL (overrides `USE_SQLITE` and `POSTGRES_*`) |
| `API_KEY` | — | 32+ char alphanumeric string; protects bank endpoints (`/score/`, `/score/portfolio/summary`, `/score/train`, `/ingest/ground_truth`). If empty, these endpoints are publicly accessible (demo only). |
| `AES_SECRET_KEY` | dev default | 64-char hex key for AES-256-GCM vault encryption |
| `GROQ_API_KEY` | — | Optional LLM for psychometric agent (heuristic fallback) |

See `.env.example` for the full list.

## Fairness Monitoring — the 80% Rule (Disparate Impact)

The engine never feeds protected attributes (caste/community group) into the model; they are used **only after scoring** to audit fairness. Every portfolio is checked with the **80% rule** (a.k.a. *disparate impact* / *adverse impact ratio*), a standard fair-lending test:

1. Compute the **approval rate for each protected group** (general, OBC, SC, ST, minority).
2. Take the ratio **lowest group's rate ÷ highest group's rate** → the **disparate impact ratio**.
3. If that ratio is **below 0.80**, the model may be disadvantaging a group → it is **flagged for manual review** rather than acted on automatically.

Example: a ratio of `0.44` means the least-approved group is approved at only 44% of the most-approved group's rate — below 0.80, so it is flagged. On the synthetic demo cohort this is mostly small-sample noise (protected group is assigned independently of creditworthiness, ~20 borrowers per group), **not** real bias — but it demonstrates the safeguard actually detecting and surfacing disparity, exactly as a real deployment would. The dashboard shows the ratio, per-group approval rates, and a pass/flag verdict at `/score/portfolio/summary` and on the **Fairness Monitor** chart.

Implemented in [`convergence/fairness.py`](convergence/fairness.py); threshold `DISPARATE_IMPACT_THRESHOLD = 0.8`.

## Deployment

The live demo runs on **Render** (Web Service + managed Postgres). In the Render dashboard:

1. Connect this GitHub repo and set **Dockerfile** as the build method
2. Add a Postgres instance and set `DATABASE_URL` on the web service
3. Copy remaining env vars from `.env.example` (`API_KEY`, `AES_SECRET_KEY`, etc.)
4. Set start command to `bash scripts/entrypoint.sh` and health check path to `/health`

Set `SEED_ON_STARTUP=true` (the default) so the demo dashboard is populated on first boot. The legacy `AUTO_SEED_ON_STARTUP` flag in `scripts/entrypoint.sh` triggers an alternate full generate → API load → train path and is not needed when using the bundled model artifact.

Other options for local pitches or production narratives:

| Platform | Best For | Tradeoff |
|----------|----------|----------|
| **Render** (deployed) | Hosted demo URL | Cold starts on free tier |
| **docker-compose on Mumbai VM** | Data localization story | You manage ops |
| **ngrok** | Live pitch from laptop | Ephemeral URL |

## Testing Borrower vs Bank Flows

**Borrower flow:**
1. Open http://localhost:8000/apply
2. Click "Start New Application" → goes to `/consent`
3. Complete consent flow → redirects to `/assessment`
4. Answer assessment questions → auto-redirects to `/borrower?session=<sessionId>`
5. View your score (only your own data visible)
6. Click "Home" → returns to `/apply`, previous result now listed

**Bank flow:**
1. Set `API_KEY=test-key-123` in `.env`
2. Restart server
3. Open http://localhost:8000/dashboard
4. Add `X-API-Key: test-key-123` header (browser dev tools or use curl/Postman)
5. View portfolio overview, all borrowers, fairness metrics
6. Search for a borrower by ID to see their full details

**Access control test:**
```bash
# Borrower can only see their own score
curl -H "X-Session-Id: <their-session-id>" http://localhost:8000/score/me

# Borrower cannot access another borrower's score
curl -H "X-Session-Id: <their-session-id>" http://localhost:8000/score/<other-user-id>
# Response: 403 Forbidden

# Bank officer can see all scores
curl -H "X-API-Key: test-key-123" http://localhost:8000/score/

# Without API key, portfolio endpoints are blocked
curl http://localhost:8000/score/portfolio/summary
# Response: 401 Unauthorized
```

## Testing

```bash
pytest tests/ -v
```

## License

Hackathon prototype — not for production use without regulatory approval.
