# Speech Script: Alt-Credit Engine Presentation

A ~15-minute slide presentation + ~5-minute product demo. **Gauri** (time-series
econometrics, risk, and credit policy) anchors most of the narration; **Sonil** takes the
two heaviest engineering slides and drives the **live product demo**, so his slide time is
lighter. Most slides are single-speaker to keep the flow clean — we only hand off mid-slide
when there are genuinely two distinct ideas that belong to two different people. Tone:
professional, conversational, jargon-light. Lines are written to be *spoken*, not read —
short sentences, one idea at a time.

> **Judging framework (map the talk to it explicitly):** Slides should land the *problem
> statement*, the *uniqueness* of the solution, a *comparison against benchmarks*, the
> *technical moat*, and *practical deployability inside banking infrastructure*. The demo
> should justify our *TRL (Technology Readiness Level)*, walk the *roadmap / timeline*, and
> show *how cleanly this integrates with existing systems*. Cue words below are wired to
> these: uniqueness (S1), benchmark (S6), moat (S6), deployability (S2, S7), TRL + roadmap
> + integration (demo + S8).

> **Demo cue:** Sonil runs the live walkthrough of the borrower flow and dashboard at a
> natural break (recommended: right after Slide 2, once the ingestion story is set, or
> just before the roadmap). Keep it to ~5 minutes and open it by anchoring the TRL:
> *"What you're about to see is a working end-to-end system, not a mockup — TRL 5 to 6.
> Every screen is talking to the live scoring engine."* Then walk borrower onboarding →
> voice/psychometric intake → EBM scorecard + reason codes → the fairness/affordability
> review queue. Close the demo on integration: the scorecard is a standard 300–900 output
> and the reason codes are RBI Key Fact Statement fields, so it drops into a bank's existing
> loan-origination stack rather than replacing it.

---

## Slide 1: Hardening the Architecture for Production
**Visual Cues:** Core Underwriting Thesis — 5 Raw Data Pillars + Cohort Arrays → Sequential Dual-Engine (Engine A: Econometric Pre-Processing, Engine B: EBM Champion + Challenger Panel) → Calibrated 300–900 PDO Scorecard. Three progress milestones: Econometric Detrending, Matrix Expansion, Glass-Box & Safety Gates.

* **Gauri:**
  "Good morning, everyone. Here's the problem we set out to solve: how do you safely lend to someone who has no credit history at all?

  About half the country can't get a fair loan for exactly that reason — there's no file to score. The easy answer is to point modern AI at the problem and hope. As a bank, we can't do that. We answer to regulators, we carry the risk, and we have to justify every single decision.

  So here's what makes our approach different. Most alt-credit players chase accuracy with a black box and worry about explaining it later. We flipped that. We started with a model that's explainable from day one — and still matches a black box on accuracy. That's the uniqueness: a model a regulator can audit line by line, that still holds its own against the best opaque models out there.

  Concretely, we built a two-stage scoring core. It starts with five raw data pillars and cohort arrays. Engine A uses time-series econometrics to clean the data and measure how stable someone's cash flow really is. Engine B is our main model — an Explainable Boosting Machine, backed by a panel of challenger models. And it all converges into one familiar output: a calibrated 300-to-900 credit score that drops straight into the tools banks already use.

  Everything you'll see today comes down to three things we hardened to get from prototype to production. One — we fixed a math bug where rising income looked like instability. Two — we expanded the feature matrix: smarter imputation for missing data, plus real alt-data signals for UPI, agriculture, and vendors. And three — the glass box and the safety gates: we moved off a black-box CatBoost model onto an intrinsically explainable EBM, with a conformal abstention layer on top. Let me walk you through each one."

---

## Slide 2: Ingestion Infrastructure — Cryptographic Vaults & Vernacular Voice AI
**Visual Cues:** The Cryptographic Flow (Raw JSON → AES-256-GCM → air-gapped `secure_vault` → background worker extracts anonymized floats → `ml_features`) and the Voice-Native STT Router (Sarvam AI primary STT/TTS → Groq `llama-3.1-8b-instant`, Gemini fallback). "Capped at 1 application / 30 days to prevent rehearsal."

* **Sonil:**
  "We're working with sensitive alternative data — telecom, cash flow, behavioral surveys — so security isn't a feature here, it's the foundation.

  On the left is the cryptographic flow. The raw JSON payload is encrypted at the door with AES-256-GCM and lands in an air-gapped vault we call `secure_vault`. The model never touches that. A background worker reaches in, extracts *only* anonymized float values, and hands those to the feature store. So the ML engine never interacts with raw personal data. That's the whole point of the design.

  On the right is access. If an applicant can't comfortably read or write English, they tap a microphone and just speak. Their voice runs through a voice-native router — Sarvam AI first, because it's tuned for Indian languages and handles both speech-to-text and text-to-speech, with Gemini as a fallback. And for the open-ended answers, we don't keyword-match — we send them to Groq running `llama-3.1-8b-instant`, which scores *financial responsibility* on a zero-to-one scale.

  One detail I want to flag, bottom right: we cap applications at one per thirty days per borrower. A psychometric test is a finite instrument — if you let someone re-take it on demand, they'll rehearse it. The cap protects the signal, and the refusal is explicit: we tell them the exact date they can re-apply."

---

## Slide 3: DPDP Compliance — Resolving Data Leakage via Strict Scope Gating
**Visual Cues:** Left — Review 1 Vulnerability: "Revoke Consent," raw JSON X'd out, but derived features (`adf_pvalue`, `resilience_coefficient`) still scoring the borrower. Right — The Production Patch: Cascading Purge, `S_scopes ↦ F_features`, consent OFF purging the full derived chain. Notes on Fair Revocation.

* **Gauri:**
  "Now, compliance. Under India's Digital Personal Data Protection Act, a borrower can revoke consent and demand their data be deleted.

  In our first review we caught exactly how most systems get this wrong — it's on the left. They delete the raw data, the bank statements, the raw JSON. But the numbers *derived* from it survive: the volatility figures, the stationarity statistics, `adf_pvalue`, `resilience_coefficient`. Those orphaned features are still quietly scoring the borrower. If you erase the source, you are obligated to erase everything computed from it."

* **Sonil:**
  "So we made deletion a property of the math, on the right. We built an explicit map from every consent scope to every feature it touches — `S_scopes` to `F_features`. When a borrower flips consent off, the system doesn't just drop the statements; it follows the map and runs a *cascading purge* — `monthly_income_mean`, `cashflow_volatility`, `trend_slope`, `is_stationary`, `adf_statistic`, `adf_pvalue`, `resilience_coefficient`, the entire downstream chain.

  And crucially, revocation is *fair*. We mask those revoked features to null and impute cohort-typical medians in their place. So the borrower loses confidence and routing signal — not points. Revoking your consent can never be used to punish you. We're left with zero orphaned data anywhere in the system."

---

## Slide 4: Feature Matrix Expansion — Engineering Signals for Diverse Cohorts
**Visual Cues:** Academic vs. Agricultural cohort table (cashflow profile, extracted signals, processing logic) and "The Ipsative Psychometric Fix" (forced-choice testing + deterministic `response_validity` score).

* **Gauri:**
  "For thin-file borrowers, context is everything. You can't grade a university student and a seasonal farmer on the same ruler — their cash flow doesn't even look the same.

  A student's profile is high-frequency, low-value digital spend, so we read signals like UPI spend consistency, promptness on small dues, and e-wallet top-up frequency — early evidence of discipline. A farmer's profile is lumpy and seasonal, so we built a harvest-income-spike feature that stops the model from flagging an *expected* surge as instability, plus input-purchase consistency as a read on necessity spending. Both cohorts also produce relative facet sub-scores, zero to a hundred, for the dashboard — sitting *alongside* the strict twenty-three-feature model schema, never feeding into it.

  On the bottom: the psychometric fix. A normal survey is easy to game — everyone picks the responsible-sounding answer. So we replaced Likert scales with *forced-choice* testing. Instead of 'Are you responsible?', we make you choose between two options that are both good, so you can only reveal what you actually prioritize. And we pair it with a deterministic `response_validity` score that checks whether your answers stay consistent across statement pairs. Straight-line faking gets neutralized."

---

## Slide 5: Engine A — Structural Stability & Income Growth Normalization
**Visual Cues:** Before/after graphs — "Increasing Salary Trend (The Problem)" vs. "Error Correction Model Shock (The Fix)" — and "The New Detrending Math" (deterministic trend, detrending residuals, new ECM) with final Engine A vectors.

* **Gauri:**
  "This slide is the econometrics bug we caught, and the fix — and it's my favorite one.

  Look at the graph on the left. Standard time-series models assume a flat, stable baseline. So when a borrower's income climbs — a promotion, a growing business — the old model read that upward slope as *volatility* and marked them down. The stationarity test failed, p greater than 0.05, and we penalized growth as if it were instability. That's the Increasing Salary Paradox. It's exactly backwards.

  The fix is on the right, and it's three clean steps. First, we fit a simple trend line and pull out the slope as its own feature, `trend_slope`, so a rising income becomes a *positive* signal. Second, we subtract the trend to get the residuals. Third, we run the stability model on those *detrended* residuals — so now we're measuring the wobble *around* the trend line, not the trend itself.

  The output is three honest vectors: `trend_slope`, `is_stationary`, and a `resilience_coefficient` that combines how fast income corrects after a shock with how stable it is overall. So a rising income is rewarded as growth, while genuine month-to-month instability still gets caught. The paradox is gone, and we didn't lose any real risk signal to get there."

---

## Slide 6: Engine B — AI for Regulatory Compliance (The EBM Pivot)
**Visual Cues:** The Compliance Matrix (Legacy CatBoost + SHAP vs. Production EBM), the Shape Function Additive Equation `logit(PD) = β₀ + Σ fᵢ(xᵢ)`, and the Conformal Abstention Gate.

* **Sonil:**
  "Most ML credit models are black boxes, and the industry has a standard workaround, up top on the left. Train something powerful like CatBoost, discover you can't explain it to a regulator, then bolt on SHAP to *guess* what the model probably did. But SHAP is a post-hoc estimate — it's not a legal proof. The real decider stays opaque.

  So we pivoted to the glass box on the right: an Explainable Boosting Machine. It's natively additive — the log-odds of default is just a base score plus the sum of one curve per feature. Income gets one curve, age gets another, and so on. No hidden interactions. The explanations map one-to-one to the feature columns — the model *is* the explanation. And for a credit officer, that means if the model ever learns something absurd from noise, you can look at that one curve and correct it by hand.

  Bottom of the slide is the safety net: a conformal abstention gate. When both a default and a no-default outcome are statistically plausible for an applicant, the model refuses to gamble — it abstains and defers to a human review. That's direct NPA protection.

  Now the benchmark, because this is the number that matters. The industry assumption is that a glass-box model costs you accuracy. It doesn't here. We ran our EBM head-to-head against the CatBoost black box on the same data — our EBM scores an AUC of 0.753 against CatBoost's 0.733. So the transparent model actually *edges out* the black box. That's our technical moat in one line: full native auditability *and* benchmark-beating performance, together — not a trade-off. Anyone can train a black box; very few can hand a regulator a model that's this explainable without giving up the accuracy to get there."

---

## Slide 7: Actuarial Convergence — The PDO Scorecard Transformation
**Visual Cues:** EBM Raw PD (centered on the cohort typical applicant) → PDO Scaling Constants (`BASE_SCORE=600`, `BASE_ODDS=10.0`, `PDO=50`) → Final 300–900 Credit Score, with the PDO conversion equations. Bottom: Regulatory Compliance — KFS reason-code mapping and the Affordability Gate.

* **Gauri:**
  "Credit officers don't want a machine-learning probability. They want a number between 300 and 900 that behaves the way a credit score is supposed to.

  So we run a standard actuarial conversion — the same Points-to-Double-the-Odds method behind traditional scorecards. We take the EBM's raw probability of default, anchor a base score of 600 at base odds of ten-to-one, and set fifty points to double the odds. That gives us a clean, calibrated score. And notice — the raw probability is centered on the *cohort typical applicant*, so a factor only counts as a strength when you're genuinely better than your peers, not against some flattering artificial baseline.

  The bottom half is where this pays off for compliance. Because the EBM is additive, the adverse-action drivers are sorted by their *exact* mathematical feature weight — so the Key Fact Statement reason codes are accurate by construction, not guessed. And the Affordability Gate sits on top as a post-decision overlay: if an approved borrower asks for more than they can safely service, the engine routes to human review with a counter-offer. That deliberately keeps loan-amount policy *out* of the risk model — the model assesses risk, and affordability is a separate, human-owned decision."

---

## Slide 8: Strategic Roadmap — Scaling for Institutional Deployment
**Visual Cues:** Three-phase timeline — Phase 1 (Current, complete), Phase 2 (Immediate: RBI Account Aggregator), Phase 3 (Scaling: Institutional Pilot) — and the closing tagline: "Mathematically Transparent. Compliance Ready. Built to Scale."

* **Gauri:**
  "To close, here's where we're headed — and where we are today on the readiness scale.

  Phase one is done — everything you've seen today. That puts us at TRL 5 to 6: a fully integrated system, validated end-to-end in a realistic environment on representative data. The EBM pivot, the challenger panel, the conformal abstention gate, cohort-aware imputation, and a locked test suite that keeps it all from regressing.

  Phase two is immediate, and it's about deployability: integrating with the RBI Account Aggregator framework. That's what moves `secure_vault` from synthetic simulation to live, API-driven bank-statement feeds, straight from the source — and it's where we finalize the automated RBI Key Fact Statement outputs. Because we already emit a standard 300–900 score and native KFS reason codes, this slots into a bank's existing loan-origination stack — it augments the pipeline, it doesn't replace it.

  Phase three is scaling to TRL 7 and beyond: a live institutional pilot with a Tier-1 NBFC or public-sector bank, targeting exactly the student and agricultural cohorts we validated — so we can refine the EBM shape functions on real default data."

* **Sonil:**
  "And that's the whole thesis in one line. Most teams build a black box and try to explain it afterward. We did the opposite — a transparent decider whose every curve can be audited, backed by a panel of challenger models, with a conformal safety net that hands genuine uncertainty to a human.

  Mathematically transparent. Compliance ready. Built to scale.

  Thank you — we'd love to take your questions."
