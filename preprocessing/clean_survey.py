import asyncio
import json
import re
from typing import Any

from groq import Groq

from core.config import get_settings

RISK_APPETITE_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}
SAVINGS_FREQ_MAP = {"weekly": 1.0, "monthly": 0.7, "quarterly": 0.4, "rarely": 0.1, "never": 0.0}
INTENT_LABEL_MAP = {"responsible": 0.0, "avoidant": 0.5, "impulsive": 1.0}

SURVEY_PROMPT = """Analyze the following financial stress response text from a credit applicant.
Return ONLY valid JSON with this exact schema:
{"financial_stress_score": <float 0.0 to 1.0>, "intent_label": "<responsible|avoidant|impulsive>"}

Text:
{text}
"""


def _encode_categorical(field_name: str, value: str) -> float:
    normalized = value.strip().lower()
    if field_name == "risk_appetite":
        return RISK_APPETITE_MAP.get(normalized, 0.5)
    if field_name == "savings_freq":
        return SAVINGS_FREQ_MAP.get(normalized, 0.5)
    return 0.5


def _parse_groq_json(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _fallback_stress_analysis(text: str) -> dict[str, float]:
    lowered = text.lower()
    stress_keywords = ["stress", "worried", "debt", "late", "struggle", "delay"]
    responsible_keywords = ["pay", "bills", "save", "budget", "priority", "first"]

    stress_hits = sum(1 for word in stress_keywords if word in lowered)
    responsible_hits = sum(1 for word in responsible_keywords if word in lowered)

    score = min(1.0, 0.3 + (stress_hits * 0.15))
    if responsible_hits >= 2:
        intent = "responsible"
        score = max(0.0, score - 0.2)
    elif stress_hits >= 3:
        intent = "avoidant"
    else:
        intent = "impulsive"

    return {
        "financial_stress_score": float(score),
        "intent_label_score": INTENT_LABEL_MAP[intent],
    }


async def clean_survey(raw_data: dict[str, Any]) -> dict[str, float]:
    """Extract categorical survey features and Groq-derived stress/intent scores."""
    features: dict[str, float] = {
        "risk_appetite": _encode_categorical("risk_appetite", str(raw_data.get("risk_appetite", "medium"))),
        "savings_freq": _encode_categorical("savings_freq", str(raw_data.get("savings_freq", "monthly"))),
    }

    stress_text = str(raw_data.get("stress_response_text", "")).strip()
    if not stress_text:
        features["financial_stress_score"] = 0.5
        features["intent_label_score"] = 0.5
        return features

    settings = get_settings()
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
        fallback = _fallback_stress_analysis(stress_text)
        features.update(fallback)
        return features

    def _call_groq() -> dict[str, Any]:
        client = Groq(api_key=settings.GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial risk analyst. Respond with JSON only.",
                },
                {"role": "user", "content": SURVEY_PROMPT.format(text=stress_text)},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        return _parse_groq_json(completion.choices[0].message.content or "{}")

    try:
        parsed = await asyncio.to_thread(_call_groq)
        stress_score = float(parsed.get("financial_stress_score", 0.5))
        intent_label = str(parsed.get("intent_label", "avoidant")).lower()
        features["financial_stress_score"] = max(0.0, min(1.0, stress_score))
        features["intent_label_score"] = INTENT_LABEL_MAP.get(intent_label, 0.5)
    except Exception:
        features.update(_fallback_stress_analysis(stress_text))

    return features
