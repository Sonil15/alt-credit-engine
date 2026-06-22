# Regulatory & Compliance Framework

This document summarizes how the Alt-Credit Engine prototype aligns with Indian regulatory expectations for digital lending and data privacy.

## RBI Account Aggregator Framework

- Consent is captured via an AA-style artifact with explicit **purpose**, **data fiduciary**, **scopes**, and **expiry**.
- Endpoints: `GET /consent/authorize`, `POST /consent/token`, `POST /consent/revoke`.
- Data is fetched only after consent; revocation stops further processing.

## DPDP Act 2023

- **Data minimization**: Models consume tokenized `ml_features`, never raw PII from `secure_vault`.
- **Encryption at rest**: AES-256-GCM on all ingested payloads before persistence.
- **Consent & withdrawal**: AA consent flow with revocation endpoint.
- **Purpose limitation**: Consent purpose is bound to alternate credit assessment only.

## RBI Digital Lending Guidelines (2022)

- **Key Fact Statement (KFS)**: Score API returns PD, decision, and plain-language reason codes suitable for KFS disclosure.
- **Cooling-off period**: Decision tiers (APPROVE / REVIEW / REJECT) support manual review before disbursement.
- **Fair practices**: SHAP explainability + adverse-action reason codes on every decision.

## Data Localization

- Production deployment should use **India-region** infrastructure (e.g. AWS Mumbai, Azure Central India).
- `docker-compose.yml` supports self-hosted deployment with encrypted PostgreSQL volumes.
- See deployment guide in README for Render (demo) vs Mumbai VM (production narrative).

## Fairness & Non-Discrimination

- Protected attributes are **excluded from model features**.
- Portfolio fairness report (`GET /score/portfolio/summary`) monitors approval-rate parity across groups.
- 80% disparate-impact rule is computed and surfaced on the bank dashboard.

## Audit Trail

- Every scoring decision is persisted to `score_decisions` with model version, score, PD, decision, and reason codes.
- Supports regulatory audit and model governance reviews.

## Multilingual Psychometric Assessment

- **Agent does not decide**: An agentic conversational UI (`/assessment`) guides onboarding in English, Hindi, or Bengali, but scoring uses a **fixed, pre-translated item bank** with deterministic keyed scoring.
- **Construct-based traits**: conscientiousness, locus of control, financial self-efficacy, present bias, debt attitude, plus a response-validity signal from consistency-check pairs.
- **Transcript encryption**: Full assessment transcript (PII) is encrypted in `secure_vault`; only numeric trait features enter `ml_features`.
- **Voice inclusion**: Browser Web Speech API (free) supports voice input/output for low-literacy users; provider is abstracted for future Bhashini/AI4Bharat integration.
- **Explainability**: Psychometric traits appear in SHAP drivers and adverse-action reason codes alongside behavioral/alternative data features.
