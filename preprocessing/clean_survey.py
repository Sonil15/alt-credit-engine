"""Extract categorical survey features and Groq-derived stress/intent scores."""

import logging
from typing import Any

from psychometric.scoring import compute_trait_scores, traits_to_ml_features

logger = logging.getLogger(__name__)

PSYCHOMETRIC_TRAITS = [
    "conscientiousness",
    "locus_of_control",
    "financial_self_efficacy",
    "present_bias",
    "debt_attitude",
    "response_validity",
]


async def clean_survey(raw_data: dict[str, Any]) -> dict[str, float]:
    """
    Extract psychometric trait features from structured assessment payload.
    Supports new agentic assessment format and legacy survey format.
    """
    if "traits" in raw_data and isinstance(raw_data["traits"], dict):
        return traits_to_ml_features(raw_data["traits"])

    if all(key in raw_data for key in PSYCHOMETRIC_TRAITS):
        return traits_to_ml_features(raw_data)

    if "answers" in raw_data and isinstance(raw_data["answers"], dict):
        traits = compute_trait_scores(raw_data["answers"])
        return traits_to_ml_features(traits)

    # Legacy fallback: map old fields to approximate traits
    logger.warning("Legacy survey payload detected; mapping to approximate psychometric traits")
    risk = str(raw_data.get("risk_appetite", "medium")).lower()
    savings = str(raw_data.get("savings_freq", "monthly")).lower()
    stress_text = str(raw_data.get("stress_response_text", ""))

    risk_map = {"low": 0.8, "medium": 0.5, "high": 0.2}
    savings_map = {"weekly": 0.9, "monthly": 0.7, "quarterly": 0.5, "rarely": 0.3, "never": 0.1}

    from psychometric.scoring import score_open_ended_answer

    open_score = score_open_ended_answer(stress_text)
    return {
        "conscientiousness": savings_map.get(savings, 0.5),
        "locus_of_control": risk_map.get(risk, 0.5),
        "financial_self_efficacy": open_score,
        "present_bias": 1.0 - risk_map.get(risk, 0.5),
        "debt_attitude": open_score,
        "response_validity": 0.8,
    }
