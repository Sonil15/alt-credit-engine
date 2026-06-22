# Alt-Credit Engine

Privacy-preserving alternate credit scoring system for thin-file borrowers in India. Built for the **UCO Bank hackathon** — ingests alternative data (telecom, e-commerce, geolocation, cashflow, psychometric survey), encrypts it at rest, extracts ML features, and produces a **300–900 credit score** with SHAP explainability, adverse-action reason codes, and portfolio fairness monitoring.

## Architecture

```
AA Consent Gateway → Ingest API → AES-256 Vault → Preprocessing → ml_features + feature_series
                                                                        ↓
                                                              ECM (real time series) + CatBoost
                                                                        ↓
                                    Convergence (PDO scorecard + SHAP + reason codes) → Audit Trail
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
| **RBI AA consent** | Account Aggregator-style consent with purpose, expiry, revocation |
| **Five-pillar sub-scores** | Per-data-source scores (telecom, spending, location, cashflow, psychometric), 0–100, population-normalised — drives the radar chart |
| **Coherent explainability** | SHAP log-odds contributions converted to score points so the waterfall reconciles to the headline score (`base_points + Σ points ≈ score`) |
| **Risk-based lending** | Recommends max loan, risk-priced rate, tenure, and EMI per applicant |
| **Confidence / thin-file** | Data-sufficiency score from how many pillars are backed by real data; low-confidence files routed to review, never silent auto-approve |
| **Audit trail** | Every decision logged to `score_decisions` table |
| **Self-seeding startup** | Demo cohort auto-loads into the DB on first boot (idempotent) — no manual seed/load/train; optional SQLite mode for offline laptop demos |
| **Deployment** | Docker + docker-compose + Railway config |
| **Multilingual psychometrics** | Agent-guided assessment in EN/HI/BN with deterministic trait scoring + voice support |

## Psychometric Assessment

| URL | Description |
|-----|-------------|
| http://localhost:8000/assessment | Multilingual agentic psychometric chat (EN/HI/BN, voice + text) |
| http://localhost:8000/consent | AA consent gateway → redirects to assessment |

Constructs measured: conscientiousness, locus of control, financial self-efficacy, present bias, debt attitude. The agent handles conversation; a fixed rubric produces the score.

- Python 3.11+
- Docker & Docker Compose
- Groq API key (optional; survey NLP falls back to keyword heuristics)

## Quick Start (Docker + Postgres — primary / deployed configuration)

This is the configuration deployed on Render. The app **self-seeds** the 100-borrower
demo cohort into Postgres on first startup and loads the bundled pre-trained CatBoost
model — no manual seed/load/train steps needed:

```bash
cp .env.example .env
docker compose up --build
```

Open http://localhost:8000/dashboard — populated automatically.

Seeding is idempotent (it skips if the database already has data), so restarts are
safe. Training stays an explicit action (`POST /score/train` or
`python -m models_ai.train`) so you can demo the full ECM + CatBoost pipeline live.

## Quick Start (optional — zero-dependency local laptop demo)

For a quick offline run with **no Docker and no Postgres**, set `USE_SQLITE=true`.
The engine uses a local SQLite file and self-seeds on first boot:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
USE_SQLITE=true uvicorn api.main:app --port 8000
```

This is only a convenience for local demos — Postgres remains the default and the
deployed backend. (A provided `DATABASE_URL`, e.g. on Render, always takes precedence.)

## Demo UI

| URL | Description |
|-----|-------------|
| http://localhost:8000/ | Welcome page — dashboard or borrower consent flow |
| http://localhost:8000/consent | RBI AA consent gateway → psychometric assessment |
| http://localhost:8000/assessment | Multilingual agentic psychometric chat (EN/HI/BN, voice + text) |
| http://localhost:8000/dashboard | Bank LOS dashboard with portfolio, model card, fairness |
| http://localhost:8000/docs | FastAPI Swagger UI |
| http://localhost:8000/consent/compliance | Regulatory compliance summary |

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | — | Health check + model version |
| GET | `/consent/authorize` | — | RBI AA consent authorization |
| POST | `/consent/token` | — | AA token exchange |
| POST | `/consent/revoke` | — | Revoke consent (DPDP) |
| POST | `/ingest/{data_type}` | — | Ingest encrypted payload |
| POST | `/ingest/ground_truth` | API Key | Store generative labels (training only) |
| GET | `/score/{user_id}` | — | Credit score + reason codes |
| GET | `/score/` | — | All user scores |
| GET | `/score/portfolio/summary` | — | Portfolio + fairness metrics |
| GET | `/score/model/card` | — | Model validation metrics |
| POST | `/score/train` | API Key | Train ECM + CatBoost |
| GET | `/assessment` | — | Multilingual psychometric UI |
| POST | `/assessment/start` | — | Start agentic assessment session |
| POST | `/assessment/answer` | — | Submit answer to current item |

## Fairness Monitoring — the 80% Rule (Disparate Impact)

The engine never feeds protected attributes (caste/community group) into the model; they
are used **only after scoring** to audit fairness. Every portfolio is checked with the
**80% rule** (a.k.a. *disparate impact* / *adverse impact ratio*), a standard fair-lending
test:

1. Compute the **approval rate for each protected group** (general, OBC, SC, ST, minority).
2. Take the ratio **lowest group's rate ÷ highest group's rate** → the **disparate impact ratio**.
3. If that ratio is **below 0.80**, the model may be disadvantaging a group → it is **flagged
   for manual review** rather than acted on automatically.

Example: a ratio of `0.44` means the least-approved group is approved at only 44% of the
most-approved group's rate — below 0.80, so it is flagged. On the synthetic demo cohort this
is mostly small-sample noise (protected group is assigned independently of creditworthiness,
~20 borrowers per group), **not** real bias — but it demonstrates the safeguard actually
detecting and surfacing disparity, exactly as a real deployment would. The dashboard shows
the ratio, per-group approval rates, and a pass/flag verdict at `/score/portfolio/summary`
and on the **Fairness Monitor** chart.

Implemented in [`convergence/fairness.py`](convergence/fairness.py); threshold
`DISPARATE_IMPACT_THRESHOLD = 0.8`.

## Deployment Options

| Platform | Best For | Tradeoff |
|----------|----------|----------|
| **Railway** (recommended) | Always-on demo URL | Cost after free tier |
| **Render** | Free hosting | Cold starts on free tier |
| **docker-compose on Mumbai VM** | Data localization story | You manage ops |
| **ngrok** | Live pitch from laptop | Ephemeral URL |

See `railway.toml` and `Dockerfile` for container deployment. Set `AUTO_SEED_ON_STARTUP=true` so judges see populated data.

## Compliance

See [docs/COMPLIANCE.md](docs/COMPLIANCE.md) for RBI AA, DPDP Act 2023, and Digital Lending Guidelines alignment.

## Testing

```bash
pytest tests/ -v
```

## License

Hackathon prototype — not for production use without regulatory approval.
