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
]

# Keyword heuristics for open-ended extraction (multilingual hints)
RESPONSIBLE_KEYWORDS = re.compile(
    r"\b(pay|bills|rent|save|budget|priority|first|utility|groceries|"
    r"भर|बिल|बचत|बजट|প্রথম|বিল|সঞ্চয়)\b",
    re.IGNORECASE,
)
AVOIDANT_KEYWORDS = re.compile(
    r"\b(avoid|ignore|delay|stress|worried|debt|"
    r"टाल|चिंता|দেরি|উদ্বেগ)\b",
    re.IGNORECASE,
)


def score_likert_answer(item: Item, answer: str) -> float | None:
    normalized = answer.strip()
    if normalized not in item.scoring_key:
        return None
    return float(item.scoring_key[normalized])


def score_open_ended_answer(text: str) -> float:
    """Deterministic keyword-based score for open-ended responses."""
    if not text.strip():
        return 0.5
    responsible = len(RESPONSIBLE_KEYWORDS.findall(text))
    avoidant = len(AVOIDANT_KEYWORDS.findall(text))
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
    return {key: float(traits.get(key, 0.5)) for key in CONSTRUCTS + ["response_validity"]}


def format_assessment_payload(
    user_id: str,
    language: str,
    answers: dict[str, str],
    transcript: list[dict[str, Any]],
    traits: dict[str, float],
) -> dict[str, Any]:
    """Assemble structured survey payload for ingestion pipeline."""
    return {
        "user_id": user_id,
        "language": language,
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
