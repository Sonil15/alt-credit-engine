# PITCH.md — Judge-Facing Framing (living doc)

**Why this file exists.** The goal of this project is to *win the hackathon*. Good
engineering doesn't win on its own — it wins when each feature is framed as the answer
to a question a skeptical judge (a credit-risk officer, a regulator, an ML lead) is
already asking. This file is the single source for **how to pitch every significant
feature**. It is maintained by humans *and* by coding agents (see the protocol below).

> **White-label rule:** never name the sponsoring bank or university in this file or
> anywhere in the repo. Use neutral terms — "the bank", "a public-sector bank",
> "the lender", "Data Fiduciary".

---

## How to use this file

- **Presenting?** For any feature, grab its **Pitch line** + **Demo moment** below.
- **A coding agent?** After a significant change, update this file per the
  **Maintenance Protocol**. This is a hard expectation, not a nicety.

---

## Maintenance Protocol (for coding agents: Claude, Cursor, Antigravity, …)

**When to update.** After any *significant* change — i.e. anything a judge could be
told about that gives an edge. Examples that qualify:
- a new feature or user-visible capability,
- a model/architecture/decision change (e.g. swapping a model, adding a gate),
- a credibility risk you fixed (a calibration bug, a fairness gap, a silent default),
- a defensible engineering decision with a "why" worth saying out loud.

**When to skip.** Typos, pure refactors, formatting, dependency bumps, or anything with
no judge-visible or credibility effect. Don't bloat this file — *significant only*.

**What to do.**
1. Add a new entry (or update the existing one) in **Feature Framing** using the
   **Framing Recipe** below.
2. Keep it in **judge language** — business value + credibility, not implementation
   detail. No code, no file paths in the pitch fields.
3. Obey the white-label rule. Never push to GitHub (local-only project).
4. If a change makes an existing entry inaccurate, *fix that entry* — stale pitch
   framing is worse than none (a judge will catch it).

---

## Framing Recipe (the template for each feature)

Copy this block for each feature:

```
### <Feature name>
- **Judge problem it answers:** <the objection / question this kills — e.g. "is the
  black box explainable to RBI?", "how do you avoid bad loans on thin files?">
- **Pitch line:** <one punchy sentence a judge remembers>
- **Differentiator:** <what most other teams will NOT have>
- **Demo moment:** <the concrete thing to show live that proves it>
- **Honest caveat (optional):** <state weaknesses before a judge finds them — this
  itself scores credibility points>
```

---

## Feature Framing

### Glass-box EBM champion (replaced CatBoost + SHAP)
- **Judge problem it answers:** "Post-hoc explanations (SHAP) narrate a black box — the
  *decider* is still opaque. Can a risk officer or regulator actually audit the model?"
- **Pitch line:** "Our decision model isn't explained after the fact — its decision
  function *is* the explanation. There's nothing left to narrate."
- **Differentiator:** Most teams ship a gradient-boosting black box + a SHAP chart. We
  moved the *decider* to an intrinsically interpretable model at **zero measured accuracy
  cost** (AUC 0.830 vs 0.828).
- **Demo moment:** Open a shape-function curve on the dashboard — "this line *is* the
  model; read the risk straight off the axis; a risk officer can hand-edit a wrong curve."
- **Honest caveat:** On ~100 synthetic rows we show *equivalence*, not superiority — the
  point is transparency is free, not that glass-box beats the black box.

### Champion–Challenger panel + agreement gate
- **Judge problem it answers:** "How confident are you in a single model's call,
  especially on a thin-file borrower?"
- **Pitch line:** "We run structurally different model families and *use their
  disagreement as a signal* — when they argue, a human decides."
- **Differentiator:** This is real bank model-risk-management practice (champion/
  challenger), not a single-model demo. Diversity is genuine (Spearman 0.87, not 0.99).
- **Demo moment:** Find a borrower the black-box challenger would APPROVE but the panel
  routes to REVIEW — "the black box alone would have lent; the committee caught it."

### Conformal abstention (statistically-grounded "I don't know")
- **Judge problem it answers:** "What stops the model from confidently auto-approving a
  case it genuinely can't call?"
- **Pitch line:** "When even the champion can't statistically commit at our confidence
  level, we *abstain and route to review* instead of silently lending."
- **Differentiator:** A calibrated, distribution-free guarantee (split conformal) on top
  of the panel — most hackathon projects have no abstention story at all.
- **Honest caveat:** With ~18 calibration rows the guarantee is structurally correct but
  empirically noisy on synthetic data — say so before a judge asks.

### Honest scorecard re-anchoring
- **Judge problem it answers:** "Did you tune the cutoffs to make the demo look good?"
- **Pitch line:** "Our honest model needed an honest scorecard — the old cutoffs were
  propped up by an over-confident black box, so we re-anchored to the real ~9% base rate."
- **Differentiator:** Shows calibration literacy and intellectual honesty — turns a
  potential "gotcha" into a credibility win.
- **Demo moment:** The decision split (APPROVE / REVIEW / REJECT) across the portfolio
  sitting at a believable distribution, not 100% approvals.

### Typical-applicant-centered drivers (honest "What Affected Your Score")
- **Judge problem it answers:** "Your borrower explanation shows only positives — is
  this real explainability or a feel-good marketing panel?"
- **Pitch line:** "We explain each borrower *against the typical applicant*, not against
  the model's intercept — so 'What Affected Your Score' shows genuine strengths *and*
  what needs work, instead of an all-green wall."
- **Differentiator:** We diagnosed a real bias: the EBM is trained with balanced class
  weights, so its intercept sits at a ~48% coin-flip, not the real ~9% base rate.
  Measured against that intercept, almost every applicant beats the baseline on almost
  every feature, so ~50% of borrowers saw *zero* negative drivers. We re-center each
  contribution on the population-average (the typical applicant) — a driver is positive
  only when the borrower genuinely beats a peer on that signal. All-positive cases fell
  from ~50% to ~14% (the remainder are genuinely strong borrowers — honest, not faked).
- **Negatives lead:** "What Affected Your Score" now orders the strongest *adverse*
  drivers first, then fills the rest with positives — so a rejected or marginal borrower
  can't be shown an all-green list produced by magnitude-only ranking. If nothing is
  negative, it stays positives-only (honest, not manufactured). The adverse-action reason
  codes and improvement tips are selected from those negative drivers directly rather than
  sliced from the top-3-by-magnitude, so a reject always yields real reasons instead of
  falling through to "No adverse factors identified."
- **Demo moment:** Open a mid-band borrower (~550) and read the mixed drivers — the
  unstable-income flag surfaces *first*, then the strong cash-flow — then note the
  score/PD/decision are untouched: we fixed the *explanation baseline*, not the model.
- **Honest caveat:** This re-centers the explanation only; the model, PD and decision are
  unchanged, and `base_points + Σ driver_points` still reconciles to the score exactly
  (base_points now reads as "the typical applicant's score").

### Zero-friction demo run
- **Judge problem it answers:** "Will this actually run, or is it a fragile demo?"
- **Pitch line:** "One command, populated dashboard — SQLite default, self-seeds demo
  borrowers on startup. No Docker, no setup."
- **Differentiator:** Demo reliability. Judges have seen too many projects die at setup.
- **Demo moment:** Cold start to a full dashboard in front of them.

### Multilingual psychometric assessment (English / Hindi / Bengali)
- **Judge problem it answers:** "How do you assess thin-file borrowers who don't speak
  English and have no credit history?"
- **Pitch line:** "We score financial character through a conversational assessment in
  the borrower's own language — text and voice."
- **Differentiator:** True vernacular inclusion (Devanagari + Bengali rendering, voice),
  aimed squarely at the financially-excluded thin-file segment.
- **Demo moment:** Take the assessment live in Hindi or Bengali.

### Open-ended answer scoring — LLM with confidence routing + deterministic fallback
- **Judge problem it answers:** "Free-text is messy — is your scoring of it robust and
  trustworthy, or a brittle keyword hack?"
- **Pitch line:** "We score open answers for *financial responsibility, not mood* — an
  LLM does the nuanced read, and when it's unsure or unavailable we fall back to a
  deterministic scorer instead of guessing."
- **Differentiator:** Confidence-aware routing + a reproducible fallback (cached,
  seeded) — graceful degradation, not a single fragile call. Scores stance, not
  sentiment (a stressed-but-responsible answer still scores high). The fallback is
  *genuinely multilingual*: a curated Hindi/Bengali lexicon with negation handling, so
  an offline Hindi or Bengali borrower gets a real score — not a neutral default that
  an English-only fallback (or an English-only tool like VADER) would hand them.
- **Demo moment:** Show the same answer scoring identically twice (deterministic), the
  system deferring to the safe fallback on a low-confidence read, and a Hindi/Bengali
  answer scoring as responsible-vs-avoidant even with the LLM switched off.
- **Honest caveat:** The offline fallback is a curated keyword+negation heuristic, not a
  language model — it reads clear stances well; the LLM remains the primary, nuanced
  path when available.

### Application throttle (anti-gaming the psychometric assessment)
- **Judge problem it answers:** "A psychometric questionnaire is only predictive the
  first time — what stops a rejected borrower from re-applying repeatedly, memorising the
  items, and rehearsing the 'right' answers until they pass?"
- **Pitch line:** "We cap applications per borrower per rolling window — a repeat
  applicant already knows the questionnaire, so unlimited retries would let them game the
  behavioural signal. One honest sitting, not a coached retake."
- **Differentiator:** Most teams treat the assessment as replayable; we recognise the
  psychometric items as a *finite, learnable* instrument and protect their validity with
  a server-side limit (default 1 per 30 days, keyed on borrower identity, configurable).
  The refusal is explicit — a 429 with the date they can re-apply — not a silent failure.
- **Demo moment:** Complete an assessment, immediately try to start another for the same
  borrower, and show the block with the concrete "apply again on or after <date>" message.
- **Honest caveat:** The limit is keyed on the borrower's identifier; a determined actor
  forging fresh identities is an identity/KYC problem upstream, not one this control
  claims to solve.

### Five-facet sub-scores + thin-file confidence indicator
- **Judge problem it answers:** "A single number is opaque — what is the borrower
  actually strong or weak on?"
- **Pitch line:** "We break the score into five readable facets, normalised against the
  population, with an explicit confidence flag when the file is thin."
- **Demo moment:** The facet radar on the dashboard + the thin-file confidence badge.

### Multi-dimension Fairness Monitor
- **Judge problem it answers:** "Alternate data can encode bias — how do you know you're
  not discriminating, and against which groups?"
- **Pitch line:** "We monitor approval-rate parity across five slices simultaneously —
  borrower category, gender, geography, income bracket, and social category — with a
  live 80% rule check on each."
- **Differentiator:** Most teams check one protected attribute. We built a configurable
  dimension framework: adding a new slice is a one-line entry, and the dashboard selector
  lets the loan officer or regulator switch views in one click. Demographic fields are
  monitoring-only — they are never model inputs. Default view is borrower category
  (Individual vs MSME) — the slice a loan officer reasons about — rather than leading
  with a sensitive attribute.
- **Demo moment:** Open the Fairness Monitor, switch between dimensions live — point out
  that geography passes the 80% rule (rural borrowers approved at comparable rates) while
  income bracket flags a disparity worth investigating, which is exactly the kind of
  signal a responsible lending programme should surface and act on.
- **Honest caveat:** Demographic fields are synthetic — distributions approximate
  realistic proportions but are not derived from real borrower data. The monitoring
  framework is what to demonstrate, not the specific ratios.

### Adverse-action reason codes
- **Judge problem it answers:** "If you decline someone, can you tell them why — as
  regulation requires?"
- **Pitch line:** "Every decision produces plain-language reasons a borrower can act on,
  derived from the model's own additive terms."
- **Demo moment:** Open a declined/review borrower and read the human-readable reasons.

### Loan-officer-readable dashboard labels (no psychometric jargon)
- **Judge problem it answers:** "A loan officer isn't a psychologist — will they actually
  understand what the score is telling them?"
- **Pitch line:** "The dashboard speaks the loan officer's language: 'Sense of financial
  control' and 'Tendency to spend impulsively', not 'locus of control' and 'present bias'."
- **Differentiator:** Construct names stay intact under the hood (auditable, in the item
  bank and model features), but every term the reviewer sees in the Signal Trace, the
  five-facet profile ("Character & Money Mindset"), and the reason codes is rephrased in
  plain English — so the explainability is usable, not just present.
- **Demo moment:** Point at the psychometric signals in the trace and read them aloud —
  they need no translation for the panel.

### Consent & data-protection posture (Data Fiduciary / DPDP-aligned)
- **Judge problem it answers:** "You're using alternate personal data — is this lawful
  and consented?"
- **Pitch line:** "Consent-first by design: the borrower is a data principal, we act as a
  Data Fiduciary, with scoped, revocable consent tracked end to end."
- **Demo moment:** The consent flow and revocation / scope-tracking screen.

---

## Cross-cutting narrative (the one-paragraph story)

> *"Most teams build a black box and bolt an explanation onto it. We did the opposite:
> a transparent decider whose curves a regulator can audit, wrapped in a champion–
> challenger panel that turns model disagreement into an early-warning system, with a
> conformal layer that abstains rather than guessing — all serving thin-file borrowers
> in their own language, consent-first. We replaced an unexplainable decider with a
> transparent one at zero accuracy cost, and turned the cases the models argue about
> into a human-review safety net instead of silent auto-approvals."*
