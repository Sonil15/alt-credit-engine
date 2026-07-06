# PITCH.md: Judge-Facing Framing (living doc)

**Why this file exists.** The goal of this project is to *win the hackathon*. Good
engineering doesn't win on its own, it wins when each feature is framed as the answer
to a question a skeptical judge (a credit-risk officer, a regulator, an ML lead) is
already asking. This file is the single source for **how to pitch every significant
feature**. It is maintained by humans *and* by coding agents (see the protocol below).

> **White-label rule:** never name the sponsoring bank or university in this file or
> anywhere in the repo. Use neutral terms, "the bank", "a public-sector bank",
> "the lender", "Data Fiduciary".

---

## How to use this file

- **Presenting?** For any feature, grab its **Pitch line** + **Demo moment** below.
- **A coding agent?** After a significant change, update this file per the
  **Maintenance Protocol**. This is a hard expectation, not a nicety.

---

## Maintenance Protocol (for coding agents: Claude, Cursor, Antigravity, …)

**When to update.** After any *significant* change, i.e. anything a judge could be
told about that gives an edge. Examples that qualify:
- a new feature or user-visible capability,
- a model/architecture/decision change (e.g. swapping a model, adding a gate),
- a credibility risk you fixed (a calibration bug, a fairness gap, a silent default),
- a defensible engineering decision with a "why" worth saying out loud.

**When to skip.** Typos, pure refactors, formatting, dependency bumps, or anything with
no judge-visible or credibility effect. Don't bloat this file, *significant only*.

**What to do.**
1. Add a new entry (or update the existing one) in **Feature Framing** using the
   **Framing Recipe** below.
2. Keep it in **judge language**: business value + credibility, not implementation
   detail. No code, no file paths in the pitch fields.
3. Obey the white-label rule. Never push to GitHub (local-only project).
4. If a change makes an existing entry inaccurate, *fix that entry*; stale pitch
   framing is worse than none (a judge will catch it).

---

## Framing Recipe (the template for each feature)

Copy this block for each feature:

```
### <Feature name>
- **Judge problem it answers:** <the objection / question this kills, e.g. "is the
  black box explainable to RBI?", "how do you avoid bad loans on thin files?">
- **Pitch line:** <one punchy sentence a judge remembers>
- **Differentiator:** <what most other teams will NOT have>
- **Demo moment:** <the concrete thing to show live that proves it>
- **Honest caveat (optional):** <state weaknesses before a judge finds them, this
  itself scores credibility points>
```

---

## Feature Framing

### Glass-box EBM champion (replaced CatBoost + SHAP)
- **Judge problem it answers:** "Post-hoc explanations (SHAP) narrate a black box: the
  *decider* is still opaque. Can a risk officer or regulator actually audit the model?"
- **Pitch line:** "Our decision model isn't explained after the fact: its decision
  function *is* the explanation. There's nothing left to narrate."
- **Differentiator:** Most teams ship a gradient-boosting black box + a SHAP chart. We
  moved the *decider* to an intrinsically interpretable model at **zero measured accuracy
  cost** (cross-validated AUC 0.753 vs 0.733, statistically indistinguishable on
  synthetic data).
- **Demo moment:** Open a shape-function curve on the dashboard, "this line *is* the
  model; read the risk straight off the axis; a risk officer can hand-edit a wrong curve."
- **Honest caveat:** On ~100 synthetic rows we show *equivalence*, not superiority: the
  point is transparency is free, not that glass-box beats the black box.

### Champion–Challenger panel + agreement gate
- **Judge problem it answers:** "How confident are you in a single model's call,
  especially on a thin-file borrower?"
- **Pitch line:** "We run structurally different model families and *use their
  disagreement as a signal*, when they argue, a human decides."
- **Differentiator:** This is real bank model-risk-management practice (champion/
  challenger), not a single-model demo. Diversity is genuine (PD rank-correlation
  ~0.6, different function families, nowhere near lockstep).
- **Demo moment:** Find a borrower the black-box challenger would APPROVE but the panel
  routes to REVIEW, "the black box alone would have lent; the committee caught it."

### Conformal abstention (statistically-grounded "I don't know")
- **Judge problem it answers:** "What stops the model from confidently auto-approving a
  case it genuinely can't call?"
- **Pitch line:** "When even the champion can't statistically commit at our confidence
  level, we *abstain and route to review* instead of silently lending."
- **Differentiator:** A calibrated, distribution-free guarantee (split conformal) on top
  of the panel. Most hackathon projects have no abstention story at all.
- **Honest caveat:** With ~18 calibration rows the guarantee is structurally correct but
  empirically noisy on synthetic data, say so before a judge asks.

### Honest scorecard re-anchoring
- **Judge problem it answers:** "Did you tune the cutoffs to make the demo look good?"
- **Pitch line:** "Our honest model needed an honest scorecard. The old cutoffs were
  propped up by an over-confident black box, so we re-anchored to the real ~9% base rate."
- **Differentiator:** Shows calibration literacy and intellectual honesty, turns a
  potential "gotcha" into a credibility win.
- **Demo moment:** The decision split (APPROVE / REVIEW / REJECT) across the portfolio
  sitting at a believable distribution, not 100% approvals.

### Typical-applicant-centered drivers (honest "What Affected Your Score")
- **Judge problem it answers:** "Your borrower explanation shows only positives, is
  this real explainability or a feel-good marketing panel?"
- **Pitch line:** "We explain each borrower *against the typical applicant*, not against
  the model's intercept, so 'What Affected Your Score' shows genuine strengths *and*
  what needs work, instead of an all-green wall."
- **Differentiator:** We diagnosed a real bias: the EBM is trained with balanced class
  weights, so its intercept sits at a ~48% coin-flip, not the real ~9% base rate.
  Measured against that intercept, almost every applicant beats the baseline on almost
  every feature, so ~50% of borrowers saw *zero* negative drivers. We re-center each
  contribution on the population-average (the typical applicant); a driver is positive
  only when the borrower genuinely beats a peer on that signal. All-positive cases fell
  from ~50% to ~14% (the remainder are genuinely strong borrowers, honest, not faked).
- **Negatives lead:** "What Affected Your Score" now orders the strongest *adverse*
  drivers first, then fills the rest with positives, so a rejected or marginal borrower
  can't be shown an all-green list produced by magnitude-only ranking. If nothing is
  negative, it stays positives-only (honest, not manufactured). The adverse-action reason
  codes and improvement tips are selected from those negative drivers directly rather than
  sliced from the top-3-by-magnitude, so a reject always yields real reasons instead of
  falling through to "No adverse factors identified."
- **Demo moment:** Open a mid-band borrower (~550) and read the mixed drivers: the
  unstable-income flag surfaces *first*, then the strong cash-flow: then note the
  score/PD/decision are untouched: we fixed the *explanation baseline*, not the model.
- **Honest caveat:** This re-centers the explanation only; the model, PD and decision are
  unchanged, and `base_points + Σ driver_points` still reconciles to the score exactly
  (base_points now reads as "the typical applicant's score").

### Zero-friction demo run
- **Judge problem it answers:** "Will this actually run, or is it a fragile demo?"
- **Pitch line:** "One command, populated dashboard (SQLite default, self-seeds demo
  borrowers on startup. No Docker, no setup."
- **Differentiator:** Demo reliability. Judges have seen too many projects die at setup.
- **Demo moment:** Cold start to a full dashboard in front of them.

### Multilingual psychometric assessment (English / Hindi / Bengali)
- **Judge problem it answers:** "How do you assess thin-file borrowers who don't speak
  English, may not read comfortably, and have no credit history?"
- **Pitch line:** "We score financial character through a conversational assessment in
  the borrower's own language, spoken *to* them and spoken *by* them."
- **Differentiator:** True vernacular inclusion, both directions. **Input:** a mic on
  every open-ended question; speech-to-text runs through a vendor-agnostic provider
  layer (Sarvam (tuned for Indian-language accents) as primary, Gemini as fallback)
  behind a single env var, with the browser's own Web Speech API as a zero-config,
  zero-cost path. A dictated answer lands in the text box for the borrower to review
  and edit before submitting, so a misheard word never silently becomes a wrong answer.
  **Output:** the agent's prompts are read aloud with real Sarvam Hindi/Bengali voices
  (`bulbul`), toggled live on the assessment page. This fixes a genuine inclusion gap:
  most laptops/phones ship no `hi-IN`/`bn-IN` system voice, so the browser's built-in
  synthesis is *silent* for exactly the low-literacy vernacular borrowers who most need
  audio; the server voice makes the assessment truly usable by someone who can't read
  the screen. Toggling the voice off cleanly reverts to the device's own synthesis.
- **Demo moment:** Take the assessment live in Hindi. The question is *read aloud in a
  natural Hindi voice*, then tap the mic and speak the answer, and show it land as
  editable text. Then toggle "AI voice" off and note the device falls silent on Hindi:
  "that silence is the inclusion gap; our voice layer closes it."
- **Honest caveat:** No audio (spoken answers or synthesised prompts) is stored, only
  transcribed text, so there's no audio trail to show, by design. Server voices need a
  network round-trip (~0.6s per prompt); if a call fails it falls back to the device
  voice rather than blocking the flow.

### Open-ended answer scoring: LLM with confidence routing + deterministic fallback
- **Judge problem it answers:** "Free-text is messy: is your scoring of it robust and
  trustworthy, or a brittle keyword hack?"
- **Pitch line:** "We score open answers for *financial responsibility, not mood*, an
  LLM does the nuanced read, and when it's unsure or unavailable we fall back to a
  deterministic scorer instead of guessing."
- **Differentiator:** Confidence-aware routing + a reproducible fallback (cached,
  seeded), graceful degradation, not a single fragile call. Scores stance, not
  sentiment (a stressed-but-responsible answer still scores high). The fallback is
  *genuinely multilingual*: a curated Hindi/Bengali lexicon with negation handling, so
  an offline Hindi or Bengali borrower gets a real score, not a neutral default that
  an English-only fallback (or an English-only tool like VADER) would hand them. It
  also handles voice input: a Sarvam-transcribed Hindi answer often transliterates
  spoken English terms *into Devanagari* (e.g. "रेंट", "इंटरेस्ट रीपेमेंट"), which the
  keyword lexicon can't recognise (it expects either script's native vocabulary): the
  LLM reads code-mixed transliteration natively, so the graceful-degradation story
  extends cleanly from typed to spoken answers.
- **Demo moment:** Show the same answer scoring identically twice (deterministic), the
  system deferring to the safe fallback on a low-confidence read, and a Hindi/Bengali
  answer scoring as responsible-vs-avoidant even with the LLM switched off.
- **Honest caveat:** The offline fallback is a curated keyword+negation heuristic, not a
  language model, it reads clear stances well; the LLM remains the primary, nuanced
  path when available. We also caught and fixed a real instance of exactly this risk:
  the configured Groq model (`llama3-8b-8192`) had been silently decommissioned
  upstream, so the LLM path was quietly falling back on *every* open-ended answer with
  zero trace. Fixed by pinning a current model (`llama-3.1-8b-instant`) and adding a
  warning log on any future Groq failure. The fallback should be a rare safety net,
  not an invisible default.

### Application throttle (anti-gaming the psychometric assessment)
- **Judge problem it answers:** "A psychometric questionnaire is only predictive the
  first time: what stops a rejected borrower from re-applying repeatedly, memorising the
  items, and rehearsing the 'right' answers until they pass?"
- **Pitch line:** "We cap applications per borrower per rolling window, a repeat
  applicant already knows the questionnaire, so unlimited retries would let them game the
  behavioural signal. One honest sitting, not a coached retake."
- **Differentiator:** Most teams treat the assessment as replayable; we recognise the
  psychometric items as a *finite, learnable* instrument and protect their validity with
  a server-side limit (default 1 per 30 days, keyed on borrower identity, configurable).
  The refusal is explicit (a 429 with the date they can re-apply) not a silent failure,
  and it's fully localised (EN/HI/BN) with a human-readable date ("5 August 2026", not a
  raw "2026-08-05"), since this message is read aloud by the assessment's TTS, a spoken
  ISO date reads as disconnected digits, not a date.
- **Demo moment:** Complete an assessment, immediately try to start another for the same
  borrower, and show the block with the concrete "apply again on or after 5 August 2026"
  message, spoken correctly by the AI voice in Hindi or Bengali.
- **Honest caveat:** The limit is keyed on the borrower's identifier; a determined actor
  forging fresh identities is an identity/KYC problem upstream, not one this control
  claims to solve.

### Five-facet sub-scores + thin-file confidence indicator
- **Judge problem it answers:** "A single number is opaque: what is the borrower
  actually strong or weak on?"
- **Pitch line:** "We break the score into five readable facets, normalised against the
  population, with an explicit confidence flag when the file is thin."
- **Demo moment:** The facet radar on the dashboard + the thin-file confidence badge.

### Cohort-aware imputation (missing data ≠ bad data)
- **Judge problem it answers:** "Your borrowers are thin-file by definition: what does
  the model do when a whole data source is missing? Doesn't a blank field silently
  penalise exactly the excluded people you claim to serve?"
- **Pitch line:** "A missing source resolves to *typical-for-someone-like-you*, not to
  zero, because zero isn't neutral, it's a verdict."
- **Differentiator:** Most teams zero-fill missing features and never notice that zero is
  directional; zero income reads as 'destitute', zero missed-payments reads as 'flawless
  history', so the same blank both punishes *and* rewards depending on the field. We learn
  a per-cohort typical-applicant profile at training time and impute an absent source with
  the median of the borrower's *own* cohort. The system also tells apart "applicable but
  not collected" (a genuine thin file → fill the cohort-typical value) from "structurally
  not applicable" (business vintage for a salaried worker → correctly stays zero). It's
  the same mechanism that makes consent-revocation fair: withdrawing a source makes you
  look *average* on it, never worst-case.
- **Demo moment:** Take an approved salaried applicant, blank their entire cashflow
  source live, and re-score. The score barely moves and the confidence badge drops,
  instead of the applicant cratering to a reject. "Missing data makes us less *confident*,
  not more *punitive*."
- **Honest caveat:** Imputing to the cohort typical is a deliberately conservative
  central estimate, it can flatter a genuinely weak thin file, which is exactly why a
  low-confidence thin file is routed to human review rather than auto-approved.

### Auto-drafted adverse-action letter + officer sign-off (closing the accountability loop)
- **Judge problem it answers:** "A model that rejects people has to answer *to* someone:
  is this deployable under fair-lending rules, and who is accountable for a rejection?"
- **Pitch line:** "We close the loop: a glass-box reason becomes a regulator-format
  notice, in the borrower's own language, that a named officer signs, every rejection is
  defensible and has a human on the hook."
- **Differentiator:** This is not templating garnish. It's a capstone that assembles five
  things we already built (glass-box reason codes, the audit trail, tri-lingual i18n, the
  grievance/ombudsman path, and the REVIEW state) into one artifact. The letter is drafted
  *deterministically* from the same reason codes the model produced (never by an LLM) so
  it literally cannot hallucinate or disagree with the decision it explains. Approvals
  issue automatically; only rejections and review cases queue for a loan officer, who can
  edit the wording and sign, stamping their identity and a timestamp onto the record. The
  borrower retrieves the signed notice from their own account, asynchronously, so no
  officer needs to be online when they apply. In-app delivery keeps the whole flow local
  and private (no SMS gateway, no data leaving the Data Fiduciary). Every date on the
  letter (decision date, signature date) renders in the borrower's own language ("5
  अगस्त 2026" / "5 আগস্ট 2026"), not a raw ISO string dropped into an otherwise fully
  translated notice.
- **Demo moment:** On the officer dashboard, open the review queue, pick a rejection,
  switch the letter to Hindi or Bengali live, sign it: then show it appear in the
  borrower's account as a downloadable notice. "The AI drafts; the human signs; the
  borrower gets it in their language."
- **Honest caveat:** Sign-off is a demo-grade control keyed on a stated officer ID; in
  production it would sit behind authenticated officer accounts and role-based access.

### Multi-dimension Fairness Monitor
- **Judge problem it answers:** "Alternate data can encode bias, how do you know you're
  not discriminating, and against which groups?"
- **Pitch line:** "We monitor approval-rate parity across five slices simultaneously:
  borrower category, gender, geography, income bracket, and social category, with a
  live 80% rule check on each."
- **Differentiator:** Most teams check one protected attribute. We built a configurable
  dimension framework: adding a new slice is a one-line entry, and the dashboard selector
  lets the loan officer or regulator switch views in one click. Demographic fields are
  monitoring-only. They are never model inputs. Default view is borrower category
  (Individual vs MSME). The slice a loan officer reasons about, rather than leading
  with a sensitive attribute.
- **Demo moment:** Open the Fairness Monitor, switch between dimensions live, point out
  that geography passes the 80% rule (rural borrowers approved at comparable rates) while
  income bracket flags a disparity worth investigating, which is exactly the kind of
  signal a responsible lending programme should surface and act on.
- **Honest caveat:** Demographic fields are synthetic, distributions approximate
  realistic proportions but are not derived from real borrower data. The monitoring
  framework is what to demonstrate, not the specific ratios.

### Adverse-action reason codes
- **Judge problem it answers:** "If you decline someone, can you tell them why, as
  regulation requires?"
- **Pitch line:** "Every decision produces plain-language reasons a borrower can act on,
  derived from the model's own additive terms."
- **Demo moment:** Open a declined/review borrower and read the human-readable reasons.

### Loan-officer-readable dashboard labels (no psychometric jargon)
- **Judge problem it answers:** "A loan officer isn't a psychologist, will they actually
  understand what the score is telling them?"
- **Pitch line:** "The dashboard speaks the loan officer's language: 'Sense of financial
  control' and 'Tendency to spend impulsively', not 'locus of control' and 'present bias'."
- **Differentiator:** Construct names stay intact under the hood (auditable, in the item
  bank and model features), but every term the reviewer sees in the Signal Trace, the
  five-facet profile ("Character & Money Mindset"), and the reason codes is rephrased in
  plain English, so the explainability is usable, not just present.
- **Demo moment:** Point at the psychometric signals in the trace and read them aloud;
  they need no translation for the panel.

### Consent & data-protection posture (Data Fiduciary / DPDP-aligned)
- **Judge problem it answers:** "You're using alternate personal data: is this lawful
  and consented?"
- **Pitch line:** "Consent-first by design: the borrower is a data principal, we act as a
  Data Fiduciary, with scoped, revocable consent tracked end to end."
- **Demo moment:** The consent flow and revocation / scope-tracking screen.
- **Scope choice is enforced, not cosmetic:** Unchecking a data source on the consent
  screen actually gates it. The unchecked scope is revoked at grant time, so that source
  is never collected, never scored, and never appears in "What Affected Your Score." A
  survey-only applicant is explained purely by their survey signals; the revoked sources
  are masked out of the model input *and* the driver/lineage view, with a matching
  "Consent withdrawn for data source(s)" reason code. (Demo: uncheck telecom + cash-flow,
  finish the assessment, and show the drivers contain no telecom/cash-flow factors.)

### Borrower accounts: own and protect your assessment
- **Judge problem it answers:** "The borrower's credit assessment is sensitive personal
  data, who can see it, and how does the borrower control access to their own file?"
- **Pitch line:** "The borrower flow is login-gated end to end: no assessment, no
  profile, no result exists without an authenticated account. The data principal
  literally holds the key to their own file."
- **Differentiator:** Most teams demo an anonymous, link-based result page anyone can
  open; ours *requires* an authenticated borrower identity for the entire journey
  (consent → assessment → result), so every profile is bound to an owner and persists
  across devices, reinforcing the consent-first / data-principal story end to end.
- **Demo moment:** Try to open the borrower portal signed out, it redirects to sign-in.
  Log in, complete an application, then open the result on a "fresh" browser via **View
  my latest assessment**, proving the file follows the borrower's login, not a shareable
  URL. Sign out and show the file is no longer reachable.
- **Honest caveat:** Local password auth built for the demo, PBKDF2-hashed passwords and
  server-side revocable bearer tokens, but production would add reset/lockout flows and a
  hardened secret store. No third-party auth service; runs fully offline.

### Borrower onboarding: intent captured before consent
- **Judge problem it answers:** "You score people and even compute a loan offer,
  but you never asked what they actually want. How is an offer meaningful without
  the ask? And what happens to the borrower whose need doesn't fit your dropdown?"
- **Pitch line:** "Before any data is shared, the borrower tells us what they need
  and why (purpose, amount, category) so every decision downstream answers *their*
  request, not a hypothetical one, and there's always an honest way to say 'none of
  these' instead of picking a wrong box."
- **Differentiator:** Purpose options are recommended per category (a farmer sees
  crop inputs or irrigation first, a street vendor sees inventory or working
  capital), every category's list ends in an "Other, please specify" free-text
  option, and a cross-category pick is never blocked: it becomes a soft
  `purpose_consistent=False` signal the loan officer sees next to the offer instead
  of a hard rejection at the form. The free-text business-profile capture (see
  below) now also reaches Gig Workers by default and Homemakers whenever they
  declare a home-business purpose, not just Vendor/Farmer, so more thin-file
  borrowers get the chance to self-report vintage and turnover.
- **Demo moment:** Walk the onboarding page in Hindi or Bengali, switch category
  and watch the recommended purposes change; pick "Other" and type a reason; then
  show the same purpose (and the officer-facing consistency chip) on the loan
  officer's dashboard next to the offer.
- **Honest caveat:** Broadening categories to a fully open text purpose would lose
  the consistency signal entirely; the recommended-list-plus-Other design is a
  deliberate middle ground between rigid enums and unstructured free text.

### LLM business profiler: borrower-confirmed, with a deterministic fallback
- **Judge problem it answers:** "MSME borrowers have business facts no data source
  carries, but letting an LLM invent structured features for a credit model is
  reckless. How do you get the data without the hallucinations?"
- **Pitch line:** "The borrower describes their business in their own words, any of
  our three languages. The LLM reads it into structured fields, and *the borrower
  confirms every field before anything is used*. No confirmation, no data."
- **Differentiator:** Same architecture judges already saw in our answer scorer:
  confidence-aware routing to a deterministic multilingual extractor (lakh/hazaar
  numerals, Devanagari and Bengali digits, sector keywords) when the LLM is unsure
  or offline. Runs at ₹0 marginal cost, degrades gracefully with the API switched
  off, and the raw description is AES-encrypted in the vault like every other raw
  payload.
- **Demo moment:** Type "मैं 8 साल से सब्ज़ी की दुकान चलाता हूँ, महीने में ₹40,000
  कमाता हूँ", watch sector/vintage/turnover fill in, edit one field to prove the
  borrower owns the record, then kill the API key and show the offline extractor
  reading the same sentence.
- **Honest caveat:** The fallback is a curated regex/keyword extractor, not a
  language model, it reads clearly stated facts well; the LLM remains the nuanced
  primary path.

### Affordability gate: no "approvals" the borrower can't actually get
- **Judge problem it answers:** "Your model can approve a borrower who asked for
  ₹10 lakh when their income services ₹1.2 lakh. Telling them 'approved' would be
  mis-selling: what happens on that file?"
- **Pitch line:** "An approval the bank can't fund at the requested amount is not an
  approval. When the ask exceeds the serviceable maximum, the borrower gets an
  explicit counter-offer message and the file routes to a loan officer. Never a
  silent yes."
- **Differentiator:** The gate is a *lending-policy overlay*, deliberately separated
  from the model: PD, score, the model's decision, and the fairness parity metrics
  are untouched (a big ask is borrower intent, not model bias). The audit trail
  stores both the model's call and the final outcome, so the two are never
  conflated.
- **Demo moment:** Apply as a vendor asking ₹10,00,000 on a modest cash-flow
  profile: the model approves, the amber affordability gate fires, and the borrower
  page reads "cannot be approved as requested, counter-offer up to ₹X". Re-apply
  asking ₹50,000 and watch it clear.

### Self-report honesty check as a model feature
- **Judge problem it answers:** "Self-declared turnover is unverifiable and
  gameable. You trained on something the borrower can just inflate?"
- **Pitch line:** "We never score the claim. We score its *consistency* with the
  observed cash-flow. Declaring ₹1 lakh while the bank sees ₹40,000 lowers the
  signal; declaring what the data confirms raises it. Honesty is the feature."
- **Differentiator:** Inflating the declared turnover strictly hurts the applicant,
  so the feature is anti-gameable by construction. Business vintage and the
  consistency ratio are the only two onboarding features in the model (23 total),
  both bounded, both readable on the glass-box shape curves, and accuracy-neutral
  on the benchmark (OOF AUC 0.6875 with vs 0.6861 without).
- **Honest caveat:** On synthetic data the feature is demonstrative; its real value
  is the *design pattern*, cross-checking self-reports against observed data
  instead of trusting or discarding them.

---

## Cross-cutting narrative (the one-paragraph story)

> *"Most teams build a black box and bolt an explanation onto it. We did the opposite:
> a transparent decider whose curves a regulator can audit, wrapped in a champion–
> challenger panel that turns model disagreement into an early-warning system, with a
> conformal layer that abstains rather than guessing, all serving thin-file borrowers
> in their own language, consent-first. We replaced an unexplainable decider with a
> transparent one at zero accuracy cost, and turned the cases the models argue about
> into a human-review safety net instead of silent auto-approvals."*
