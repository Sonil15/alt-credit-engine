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
| **Audit trail** | Every decision logged to `score_decisions` table |
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

## Quick Start (Local)

```bash
cp .env.example .env
docker compose up -d postgres
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m synthetic_data.generate_raw_mock
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# In another terminal:
python -m synthetic_data.load_mock_data
python -m models_ai.train
```

## Quick Start (Docker — Full Stack)

```bash
cp .env.example .env
# Set AUTO_SEED_ON_STARTUP=true in .env for auto demo data
docker compose up --build
```

Open http://localhost:8000/dashboard

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
