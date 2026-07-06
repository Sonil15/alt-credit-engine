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
- **Fair practices**: Glass-box EBM (Explainable Boosting Machine) native additive-term explainability + adverse-action reason codes on every decision. Each feature's contribution is the model's own exact points term (`base_points + Σ feature_points == credit_score`), not a post-hoc approximation, so the same global curve is stable and publishable for every borrower.

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
- **Voice & keyboard inclusion**: Voice input (speech-to-text) and output (text-to-speech) run through a vendor-agnostic provider layer, Sarvam (Indian-language-tuned) primary, Gemini fallback, with the browser's own Web Speech API as a zero-config, zero-cost default when neither is configured. An in-page on-screen Hindi/Bengali keyboard covers borrowers whose device has no Indic system keyboard installed, on the assessment's open-ended questions and the onboarding free-text fields (business description, "Other" purpose).
- **Explainability**: Psychometric traits appear in the EBM's native additive drivers and adverse-action reason codes alongside behavioral/alternative data features, no SHAP or other post-hoc approximation in the decision or explanation path.
