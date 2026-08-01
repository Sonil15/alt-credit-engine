# Demo Script: ACE Live Product Walkthrough

This picks up exactly where [`speech_script.md`](speech_script.md)'s closing segue leaves off — **Sonil** drives the whole demo hands-on-keyboard, live, immediately after the slides. Order: **Score Explainer first** (to demystify the mechanics before showing either surface built on top of it), **then the Apply Flow** (borrower side), **then the Loan Dashboard** (lender side). Budget **~5–6 minutes** total: a short framing beat, ~2 minutes on the Explainer, ~90–120s on the apply flow, ~2 minutes on the dashboard. If you're running long, cut inside the Explainer's 8 steps first — the dashboard's shape-function viewer is the single strongest visual in the whole demo and shouldn't get rushed.

> **What's real vs. simulated — say this once, early, plainly, rather than let a judge catch it.**
> DigiLocker "verification," the CIBIL-history dropdown, OTP (hardcoded `123456`), the animated "Bureau Check" sequence, and the Udyam registry lookup are all **UI theater** — timed `setTimeout` sequences with no real government/bureau API behind them. Everything else is **real and live**: the LLM business-profile extraction, the psychometric item bank and its LLM-scored open-ended question, voice STT/TTS, the full econometric + EBM scoring pipeline, the conformal/anomaly/panel integrity gates, the fairness monitor, and the decision-letter sign-off.

> **Setup, before you're on stage:**
> 1. Pre-register and log in a demo borrower account beforehand — don't burn demo time on the register/OTP theater. Pick **"Micro Enterprise / Vendor"** as the cohort: it's the richest walkthrough (free-text business description → AI extraction, Udyam field, cohort-specific consent scopes) and it's the MSME thin-file story the deck leads with.
> 2. Open three tabs in advance, all idle and ready, **in the order you'll use them**: `/explanation` (with cohort pre-set to Vendor), `/apply`, and `/dashboard`. Don't navigate to any of them cold during the demo.
> 3. Know your pre-seeded data going in: which cohort's seeded applicant has full vault coverage (so the Explainer's step 2 tabs aren't empty), and at least one **pending** decision letter in the dashboard's review queue so the sign-off moment isn't hunting for a row.
> 4. Have your fallback ready (a recorded backup clip, or a second seeded account) *before* you switch off the slides — see the segue's preemptive questions in `speech_script.md`.

---

## Opening Frame (~15–20s)

**Screen:** stay on the last slide, or a blank browser tab — this is spoken before you click anything.

* **Sonil:**
  "Everything we build here has two sides. There's the borrower side — the apply flow, where someone actually applies for credit. And there's the lender side — the bank's loan dashboard, where an officer works the queue. We'll show you both.

  But before we show you either one, we want you to actually trust what's happening in between — because a score is only as good as your ability to audit it. So let's start under the hood."

---

## Part 1: The Score Explainer (~2 minutes)

**Screen:** `/explanation` (pre-opened tab, cohort dropdown set to Vendor)

**Demo-mechanics note — know this before you're on stage:** this page doesn't depend on any application submitted live — it's driven entirely by a **cohort dropdown** and a **"simulate gamed applicant" toggle**, and shows a real, fully-scored seeded exemplar of whichever cohort you pick. That's actually a clean fit for going first: you're not pretending this is "our" applicant, you're deliberately showing a representative, fully-scored trail so the mechanics are legible before anyone's live data exists.

* **Sonil:**
  "I've set this to a vendor — a small shop owner, the kind of thin-file borrower this whole system exists for. This page is the full computation trail, start to finish, for a real scored applicant.

  Step one is intake — the raw statement and the structured extraction our AI pulled from it. Step two is the vault — every raw alt-data payload that actually entered the scoring engine, source by source, decrypted for this view. Step three is feature engineering — every one of those raw numbers turned into a model-ready feature, tagged by which engine computed it: econometric or statistical. Step four is hard policy — two named rule checks, transience-without-income and telecom repayment history, pass or reject, before the model even runs.

  Step five is the part I actually want you to read closely." *[Pause on this step.]* "This is the EBM scorecard — literally base points, plus the sum of every feature's own point contribution, equals the final score. Not an approximation of the model's reasoning — this *is* the model's reasoning, printed as arithmetic.

  Step six is where we catch what the model alone might miss." *[Toggle "Simulate gamed applicant."]* "Watch what happens when I feed it an implausible combination — income pushed way up, every delay and volatility signal zeroed out, psychometrics maxed. A naive model would just approve this. Ours doesn't — the anomaly gate flags the *combination* as statistically implausible, even though every individual number looks great, and routes it to a human instead of auto-approving."

  *[Toggle back off.]* "Step seven is the affordability gate — a separate check on whether the requested amount is something this borrower can actually service, kept deliberately outside the risk model itself. And step eight shows how all of this becomes the radar chart and reason codes you'll see on the lender side in a minute.

  So — that's the machine. Now let's see the two doors people actually walk through to reach it."

**Preemptive Questions:**
1. *"Why show a pre-scored example instead of a live applicant?"* — Say it plainly: showing the mechanics first, on a representative fully-scored trail, is deliberate — it means the audience understands what they're looking at before we walk a fresh application through it.
2. *"Is the 'gamed applicant' toggle a real red-team test, or a scripted demo trick?"* — It's a genuine server-side feature overlay hitting the real anomaly/OOD gate — say so, and be ready to explain the actual distance metric (Mahalanobis distance against the training distribution) if pushed.

---

## Part 2: The Apply Flow — Borrower Side (~90–120s)

**Screens:** `/apply` → `/onboard` → `/consent` → `/assessment` → `/borrower`

* **Sonil:**
  "First door: the borrower. This is the apply portal — already signed in, so we skip past identity verification, which, to be upfront, is simulated for the demo: DigiLocker, OTP, the bureau-check animation, all theater. What starts here is real."

  *[Click "Start New Application" → lands on `/onboard`.]*

  "Every borrower starts by telling us who they are — not their identity, their *economic shape*. I'll pick Micro Enterprise, or Vendor, same as the example we just walked through."

  *[Select cohort "Micro Enterprise / Vendor," pick a loan purpose, enter an amount.]*

  "And instead of a rigid form, a vendor just describes their business in their own words."

  *[Type a short free-text business description into the textarea — e.g. "I run a small grocery shop, been open about 3 years, do around 40,000 to 60,000 rupees a month, busier around festivals." Click "Analyze my business."]*

  "That went to an LLM, not a keyword matcher — it comes back having filled in structured fields on its own: business type, years in business, turnover, seasonality, headcount. That's the exact 'Step 1' extraction you just saw on the Explainer, happening live."

  *[Point out the pre-filled fields, then click "Continue to Consent →" → lands on `/consent`.]*

  "Consent is RBI Account-Aggregator style — granular, scope by scope, and cohort-aware. Because I picked Vendor, it's asking for merchant UPI volumes and ONDC ratings specifically — a farmer would see mandi receipts instead. One scope can't be opted out of — the psychometric survey — because without it there's no thin-file signal at all."

  *[Grant consent → click through to `/assessment`.]*

  "Last stop before scoring: the Psychometric Questionnaire. Mostly forced-choice, quick, and hard to game." *[Answer one forced-choice item.]* "Under the hood, this questionnaire calculates 10 distinct behavioral traits — like planning discipline, impulse spending, and debt attitude — scored 0 to 1. We take a population-normalized weighted average to form the Psychometric Questionnaire facet score, which feeds directly as additive point contributions into our credit model. The open-ended question is the one that matters most, and you can answer it by voice."

  *[Click the mic button, speak a short answer to the open-ended prompt — e.g., describing how you covered an unexpected expense — let the transcript populate, edit if needed, submit.]*

  "Live speech-to-text, and the open answer itself gets scored by a language model for financial-responsibility signal. All 10 traits combine into that single auditable facet score feeding the score engine."

  *[Let it finish → redirects to `/borrower`.]*

  "And there's the result — score, decision, drivers, in plain language. You already know exactly how this number was built, because we just showed you."

**Demo-mechanics note:** the psychometric page's own result panel is dead code — it always redirects straight to `/borrower` before showing anything, so don't narrate expecting a score to appear mid-assessment. The "processing" spinner is real work (feature pipeline + EBM scoring) for a few seconds, even though its step labels are cosmetic.

**Preemptive Questions:**
1. *"Is that LLM extraction reliable, or does a borrower have to word things exactly right?"* — Know your fallback behavior: the UI reports whether extraction used the LLM or a fallback offline analyzer, and either way the borrower sees and can review the structured fields before continuing.
2. *"What stops someone from just describing a business that doesn't exist?"* — Be honest about what's checked today (Udyam format, not a real registry lookup yet) versus what real deployment would need.

---

## Part 3: The Loan Dashboard — Lender Side (~2 minutes)

**Screen:** `/dashboard` (pre-opened tab)

* **Sonil:**
  "Second door: the lender. This is what a loan officer actually works from. Up top, the portfolio view: total borrowers, approval rate, expected default rate, and a score distribution across our reject, review, and approve bands.

  This is the fairness monitor." *[Point to it, optionally switch the "Group by" dropdown.]* "Approval-rate parity by group, checked against the standard eighty-percent disparate-impact rule — live, continuously monitored, right here on the officer's own screen, not a claim we make once and forget.

  Let's pull up one borrower." *[Search or click a borrower from the list.]* "Score gauge, decision, and — this badge matters — whether they were routed through traditional bureau fast-track or through our alternative engine, so an officer always knows *which* path produced this number.

  Here's the model panel." *[Scroll to it.]* "Our glass-box EBM is the champion — it decides. CatBoost and logistic regression sit alongside as challengers, auditing it for agreement, never overriding it. When they disagree, that disagreement itself is the signal, and it routes to manual review instead of a silent auto-decision.

  And here's the one I'd ask you to remember." *[Scroll to the shape-function viewer.]* "This is not an explanation of the model. It *is* the model. Every feature has its own curve, learned directly from data, and this blue marker shows exactly where this borrower sits on it and how many points that's worth. No SHAP, no approximation — a loan officer, or a regulator, can read the risk straight off the axis, and if it ever looks wrong, we can point at the exact curve responsible and correct it by hand. This is the same math you saw broken down step by step on the Explainer — here it's the officer's actual working tool.

  Last thing — decision letters." *[Open the review queue, click "Review & Sign" on a pending letter.]* "Approvals go out automatically. Anything that needs a human — review or reject — queues here, an officer reads the actual letter, signs it, and it's released to the borrower with their name and timestamp attached. That's not a UI mockup — signing this writes a real record and unlocks the borrower's own letter view."

  *[Close the modal.]* "So that's the full loop — the mechanics first, then a vendor describing their shop in their own words, then a loan officer reading the same glass-box math and signing off. Back to Gauri and me for questions."

**Preemptive Questions:**
1. *"Can a loan officer override the model's decision, and is that logged?"* — Yes — the "Change Decision" control on the borrower panel requires a written justification and writes to an audit log; know this and offer to show it live if there's time.
2. *"Is the fairness monitor computed live, or is that a static chart for the demo?"* — It's computed from the real portfolio behind the dashboard (seeded data today, same code path as production); be ready to say plainly that the *population* is seeded/synthetic even though the *computation* is real.
