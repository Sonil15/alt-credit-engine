# Alt-Credit Engine

Privacy-preserving alternate credit scoring system for thin-file borrowers in India. Ingests alternative data (telecom, e-commerce, geolocation, cashflow, psychometric survey), encrypts it at rest, extracts ML features, and produces a 300–900 credit score with SHAP explainability.

## Architecture

```
Ingest API → AES-256 Vault → Preprocessing → ml_features
                                              ↓
                                    ECM (statsmodels) + CatBoost
                                              ↓
                              Convergence (rules + SHAP) → Credit Score
```

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Groq API key (for survey NLP; falls back to keyword heuristics if unset)

## Quick Start

### 1. Environment

```bash
cp .env.example .env
# Edit .env: set AES_SECRET_KEY (64 hex chars) and GROQ_API_KEY
```

### 2. Start PostgreSQL

```bash
docker compose up -d
```

### 3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Generate mock data (if not already present)

```bash
python -m synthetic_data.generate_raw_mock
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Load mock users into the database

```bash
python -m synthetic_data.load_mock_data
# Optional: load fewer users for quick testing
python -m synthetic_data.load_mock_data --limit 10
```

Wait ~30–60 seconds for background preprocessing to complete (Groq calls for survey data).

### 7. Train models

```bash
python -m models_ai.train
```

Or via API:

```bash
curl -X POST http://localhost:8000/score/train
```

### 8. Get a credit score

```bash
# List all scores
curl http://localhost:8000/score/

# Score a specific user
curl http://localhost:8000/score/{user_id}
```

## Demo UI

| URL | Description |
|-----|-------------|
| http://localhost:8000/consent | Mobile consent gateway + psychometric survey |
| http://localhost:8000/dashboard | Bank LOS dashboard with score, PD gauge, SHAP chart |
| http://localhost:8000/docs | FastAPI Swagger UI |

## Live Demo with ngrok

Expose the local API to a physical phone during a pitch:

```bash
# Install ngrok: https://ngrok.com/download
ngrok http 8000
```

Use the ngrok HTTPS URL on your phone:
- Consent Gateway: `https://<ngrok-id>.ngrok.io/consent`
- Bank Dashboard: `https://<ngrok-id>.ngrok.io/dashboard`

Run the backend and Docker on your presentation laptop; judges can interact from their phones.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/consent/authorize` | Mock OAuth authorization URL |
| POST | `/consent/token` | Mock OAuth token exchange |
| POST | `/ingest/{data_type}` | Ingest raw payload (telecom, ecommerce, geo, cashflow, survey) |
| GET | `/score/{user_id}` | Credit score + PD + SHAP drivers |
| GET | `/score/` | All user scores |
| POST | `/score/train` | Run ECM + CatBoost training |
| GET | `/consent` | Consent gateway UI |
| GET | `/dashboard` | Bank LOS dashboard UI |

## Project Structure

```
alt-credit-engine/
├── api/                    # FastAPI routes
├── core/                   # Config, DB, encryption, feature store
├── models/                 # SQLAlchemy ORM + Pydantic schemas
├── preprocessing/          # Data cleaners (Phase 1–2)
├── synthetic_data/         # Mock data generation + bulk loader
├── models_econometric/     # ADF + ECM resilience coefficient
├── models_ai/              # CatBoost PD model + training script
├── convergence/            # Score fusion, red-flag rules, SHAP
└── frontend/               # Jinja2 HTML templates
```

## Scoring Logic

1. **Red-flag overrides** (deterministic, before AI):
   - High geographic instability + zero income → auto-reject
   - 5+ missed telecom payments → auto-reject

2. **CatBoost PD** (0.0–1.0): trained on synthetic labels derived from feature heuristics

3. **Credit score**: `900 - (PD × 600)`, clamped to 300–900

4. **Decision**: APPROVE (≥750), REVIEW (550–749), REJECT (<550 or red-flag)

5. **SHAP**: top 3 feature drivers per user for interpretability

## Development Notes

- Raw PII never enters models — only tokenized `ml_features` table
- Survey NLP uses Groq (`llama3-8b-8192`) with keyword fallback
- CatBoost model artifact saved to `models_ai/artifacts/catboost_model.cbm`
- Re-run `python -m models_ai.train` after loading new users

## License

Hackathon prototype — not for production use.
