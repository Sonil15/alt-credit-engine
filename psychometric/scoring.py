"""Deterministic psychometric scoring from fixed item bank."""

from __future__ import annotations

import re
from typing import Any

from psychometric.bank import Item, load_item_bank, load_items

CONSTRUCTS = [
    "conscientiousness",
    "locus_of_control",
    "financial_self_efficacy",
    "present_bias",
    "debt_attitude",
    "risk_tolerance",
    "delayed_gratification",
    "honesty",
    "cognitive_reflection",
    "resourcefulness",
]

# --- Open-ended fallback lexicon (multilingual, curated) --------------------
#
# This is the *deterministic* scorer used when Groq is unavailable or its read is
# rejected (see psychometric.session.extract_open_ended_score). It must stay
# pure, offline, and dependency-free: no VADER (English-only, would regress
# hi/bn), no transformers.
#
# English terms are matched with word boundaries (so "pay" never fires inside
# "display"). Devanagari (Hindi) and Bengali terms are matched as *substrings*
# of a token, because those scripts attach vowel signs, suffixes, and
# postpositions ("बिल" → "बिलों", "সঞ্চয়" → "সঞ্চয়ের") that a strict word-boundary
# match would miss. Negation particles, by contrast, are matched as whole tokens
# so the standalone "ना" / "না" does not light up inside "करना" / "করি".

# Paying essentials first, saving, budgeting, prioritising, planning.
RESPONSIBLE_EN = re.compile(
    r"\b("
    r"pays?|paid|paying|repay(?:s|ed|ing|ment|ments)?|"
    r"bills?|rent|"
    r"sav(?:e|es|ed|ing|ings)|"
    r"budget(?:s|ed|ing)?|"
    r"priorit(?:y|ies|ise|ize|ised|ized|ising|izing)|"
    r"first|before|essentials?|necessit(?:y|ies)|"
    r"utilit(?:y|ies)|electricity|"
    r"grocer(?:y|ies)|"
    r"plan(?:s|ned|ning)?|"
    r"emergenc(?:y|ies)|responsib(?:le|ility)|ontime"
    r")\b",
    re.IGNORECASE,
)
RESPONSIBLE_INDIC = (
    # Hindi
    "भर",        # bhar-na, to pay (a bill)
    "चुक",       # chukana, pay off / settle
    "बिल",       # bill
    "किराय",     # rent (किराया / किराये)
    "बचत",       # savings
    "बचा",       # bacha-na, to save
    "बजट",       # budget
    "ज़रूरी",     # essential / necessary
    "जरूरी",
    "प्राथमिकता",  # priority
    "पहले",      # first / before
    "योजना",     # plan
    "बिजली",     # electricity
    "राशन",      # rations / groceries
    # Bengali
    "পরিশোধ",    # poriśodh, repayment / pay off
    "বিল",       # bill
    "ভাড়া",      # rent
    "সঞ্চয়",     # savings
    "বাজেট",     # budget
    "প্রথম",      # first
    "আগে",       # before / first
    "অগ্রাধিকার",  # priority
    "পরিকল্পনা",  # plan
    "প্রয়োজনীয়",  # essential / necessary
    "জরুরি",     # urgent / essential
    "বিদ্যুৎ",    # electricity
)

# Avoiding, delaying, ignoring debt and bills (plus the worry/stress around it).
AVOIDANT_EN = re.compile(
    r"\b("
    r"avoid(?:s|ed|ing)?|ignor(?:e|es|ed|ing)|neglect(?:s|ed|ing)?|"
    r"delay(?:s|ed|ing)?|postpone(?:s|d|ing)?|defer(?:s|red|ring)?|"
    r"procrastinat(?:e|es|ed|ing)|"
    r"skip(?:s|ped|ping)?|miss(?:es|ed|ing)?|forget|forgot|forgotten|"
    r"late|overdue|"
    r"debts?|"
    r"stress(?:ed|ful)?|worr(?:y|ies|ied)|anxious|anxiety"
    r")\b",
    re.IGNORECASE,
)
AVOIDANT_INDIC = (
    # Hindi
    "टाल",        # taal-na, postpone / put off
    "टालमटोल",     # procrastination
    "नजरअंदाज",   # ignore
    "अनदेखा",     # overlook / ignore
    "देर",        # late / delay (देरी)
    "भूल",        # forget
    "चिंता",       # worry / anxiety
    "तनाव",       # stress
    "क़र्ज़",       # debt / loan
    "कर्ज",
    "उधार",       # borrowing / loan
    "बकाया",      # arrears / overdue
    # Bengali
    "এড়ি",        # eriye, avoid
    "উপেক্ষা",     # ignore
    "অগ্রাহ্য",     # disregard / ignore
    "দেরি",       # delay / late
    "বিলম্ব",      # delay
    "ভুল",        # forget / mistake (ভুলে)
    "দুশ্চিন্তা",    # worry / anxiety
    "চিন্তা",       # worry
    "উদ্বেগ",      # anxiety
    "চাপ",        # stress / pressure
    "ঋণ",         # debt / loan
    "ধার",        # borrow / loan
    "বকেয়া",      # arrears / overdue
)

# Negation particles: matched as whole tokens (see note above). English
# contractions are caught generically via an ``n't`` suffix check.
NEGATION_EN = frozenset(
    {
        "not", "no", "never", "none", "nor", "neither", "without", "cannot",
        "cant", "dont", "doesnt", "didnt", "wont", "wouldnt", "shouldnt",
        "couldnt", "isnt", "arent", "wasnt", "werent", "havent", "hasnt",
    }
)
NEGATION_INDIC = frozenset(
    {
        "नहीं", "नही", "ना", "मत", "बिना",   # Hindi: no / not / don't / without
        "না", "নেই", "নয়", "নি", "ছাড়া",     # Bengali: no / not / is-not / without
    }
)

# How many tokens on either side of a keyword a negation particle reaches.
# Small, because we only want to catch the local "do not <verb>" / "<noun> नहीं"
# pattern, not negate a whole sentence.
_NEGATION_WINDOW = 2

# Splits on whitespace and common ASCII/Indic punctuation (including the
# Devanagari/Bengali danda । ॥) while keeping apostrophes so contractions like
# "don't" survive as a single token.
_TOKEN_RE = re.compile(r"[^\s।॥,.;:!?()\[\]{}\"“”`/\\|—–…]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _is_negation(token: str) -> bool:
    return (
        token in NEGATION_EN
        or token.endswith("n't")
        or token in NEGATION_INDIC
    )


def _label(token: str) -> str | None:
    """Classify a token as 'responsible', 'avoidant', or None."""
    if RESPONSIBLE_EN.search(token) or any(t in token for t in RESPONSIBLE_INDIC):
        return "responsible"
    if AVOIDANT_EN.search(token) or any(t in token for t in AVOIDANT_INDIC):
        return "avoidant"
    return None


def score_likert_answer(item: Item, answer: str) -> float | None:
    normalized = answer.strip()
    if normalized not in item.scoring_key:
        return None
    return float(item.scoring_key[normalized])


def score_forced_choice_answer(item: Item, answer: str) -> dict[str, float] | None:
    normalized = answer.strip()
    if normalized not in item.scoring_key:
        return None
    return item.scoring_key[normalized]


def score_open_ended_answer(text: str) -> float:
    """Deterministic, offline keyword score for open-ended responses (en/hi/bn).

    Curated multilingual lexicon (see module top): responsible cues push the
    score up, avoidant cues push it down, starting from a neutral 0.5. A nearby
    negation particle flips a cue's polarity, so "I do NOT save" / "बचत नहीं" /
    "সঞ্চয় করি না" reads as avoidant rather than responsible. Pure function, same
    text always yields the same score in [0, 1]; no network, no heavy deps.
    """
    if not text.strip():
        return 0.5

    tokens = _tokenize(text)
    negation_positions = {i for i, tok in enumerate(tokens) if _is_negation(tok)}

    responsible = 0
    avoidant = 0
    for i, token in enumerate(tokens):
        label = _label(token)
        if label is None:
            continue
        negated = any(
            (i - d) in negation_positions or (i + d) in negation_positions
            for d in range(1, _NEGATION_WINDOW + 1)
        )
        if negated:
            label = "avoidant" if label == "responsible" else "responsible"
        if label == "responsible":
            responsible += 1
        else:
            avoidant += 1

    base = 0.5 + (responsible * 0.12) - (avoidant * 0.1)
    return max(0.0, min(1.0, base))


def compute_response_validity(answers: dict[str, str], items: list[Item]) -> float:
    """
    Consistency between paired items. Returns 1.0 when answers align, lower when contradictory.
    """
    item_map = {item.id: item for item in items}
    pairs_checked = 0
    consistency_sum = 0.0

    for item in items:
        if not item.consistency_pair:
            continue
        pair_item = item_map.get(item.consistency_pair)
        if pair_item is None:
            continue
        if item.id not in answers or pair_item.id not in answers:
            continue

        score_a = score_likert_answer(item, answers[item.id])
        score_b = score_likert_answer(pair_item, answers[pair_item.id])
        if score_a is None or score_b is None:
            continue

        pairs_checked += 1
        diff = abs(score_a - score_b)
        consistency_sum += max(0.0, 1.0 - diff * 2)

    if pairs_checked == 0:
        return 1.0
    return round(consistency_sum / pairs_checked, 4)


def compute_trait_scores(
    answers: dict[str, str],
    open_ended_scores: dict[str, float] | None = None,
    items: list[Item] | None = None,
) -> dict[str, float]:
    """
    Map answers to per-construct sub-scores in [0, 1] plus response_validity.
    Pure function: same inputs always produce same outputs.
    """
    item_list = items or load_items()
    open_scores = open_ended_scores or {}
    construct_values: dict[str, list[float]] = {c: [] for c in CONSTRUCTS}

    for item in item_list:
        if item.id not in answers:
            continue
        if item.type == "likert":
            score = score_likert_answer(item, answers[item.id])
            if score is not None:
                construct_values[item.construct].append(score)
        elif item.type == "forced_choice":
            scores = score_forced_choice_answer(item, answers[item.id])
            if scores is not None:
                for construct in item.presented_constructs:
                    val = scores.get(construct, 0.0)
                    construct_values[construct].append(val)
        elif item.type == "open_ended":
            score = open_scores.get(item.id)
            if score is None:
                score = score_open_ended_answer(answers[item.id])
            construct_values[item.construct].append(score)

    traits: dict[str, float] = {}
    for construct in CONSTRUCTS:
        values = construct_values[construct]
        traits[construct] = round(sum(values) / len(values), 4) if values else 0.5

    traits["response_validity"] = compute_response_validity(answers, item_list)
    return traits


def traits_to_ml_features(traits: dict[str, float]) -> dict[str, float]:
    """Return feature dict suitable for ml_features storage."""
    features = {key: float(traits.get(key, 0.5)) for key in CONSTRUCTS + ["response_validity"]}
    for key in [
        "cohort_code",
        "upi_spend_consistency",
        "small_dues_payment_promptness",
        "e_wallet_topup_frequency",
        "daily_transaction_count",
        "average_ticket_size",
        "harvest_income_spike",
        "input_purchase_consistency",
        "utility_payment_consistency",
        "grocery_spend_stability",
        "business_vintage_years",
        "turnover_income_consistency",
    ]:
        if key in traits:
            features[key] = float(traits[key])
    return features


def format_assessment_payload(
    user_id: str,
    language: str,
    cohort: str,
    answers: dict[str, str],
    transcript: list[dict[str, Any]],
    traits: dict[str, float],
) -> dict[str, Any]:
    """Assemble structured survey payload for ingestion pipeline."""
    return {
        "user_id": user_id,
        "language": language,
        "cohort": cohort,
        "assessment_version": load_item_bank().get("version", "1.0"),
        "answers": answers,
        "transcript": transcript,
        "traits": traits,
        "conscientiousness": traits.get("conscientiousness", 0.5),
        "locus_of_control": traits.get("locus_of_control", 0.5),
        "financial_self_efficacy": traits.get("financial_self_efficacy", 0.5),
        "present_bias": traits.get("present_bias", 0.5),
        "debt_attitude": traits.get("debt_attitude", 0.5),
        "response_validity": traits.get("response_validity", 1.0),
    }
