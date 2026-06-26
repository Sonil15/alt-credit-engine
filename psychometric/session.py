"""Agentic psychometric assessment session manager and Groq extraction."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from groq import Groq

from core.config import get_settings
from psychometric.bank import SUPPORTED_LANGUAGES, Item, load_item_bank, load_items
from psychometric.scoring import compute_trait_scores, format_assessment_payload, score_open_ended_answer

GREETINGS = {
    "en": "Hello! As a {cohort}, we have tailored this assessment for you. I'll ask you a few short questions about your financial habits. Please answer honestly.",
    "hi": "नमस्ते! एक {cohort} के रूप में, हमने यह मूल्यांकन आपके लिए तैयार किया है। मैं आपसे आपकी वित्तीय आदतों के बारे में कुछ छोटे प्रश्न पूछूँगा/पूछूँगी। कृपया ईमानदारी से उत्तर दें।",
    "bn": "নমস্কার! একজন {cohort} হিসেবে, আমরা আপনার জন্য এই মূল্যায়নটি তৈরি করেছি। আমি আপনার আর্থিক অভ্যাস সম্পর্কে কয়েকটি ছোট প্রশ্ন করব। অনুগ্রহ করে সৎভাবে উত্তর দিন।",
}

LIKERT_HINT = {
    "en": "Reply with a number from 1 (strongly disagree) to 5 (strongly agree).",
    "hi": "1 (पूरी तरह असहमत) से 5 (पूरी तरह सहमत) तक संख्या से उत्तर दें।",
    "bn": "1 (সম্পূর্ণ অসম্মত) থেকে 5 (সম্পূর্ণ সম্মত) পর্যন্ত সংখ্যা দিয়ে উত্তর দিন।",
}

FORCED_CHOICE_HINT = {
    "en": "Reply with A, B, or C to select the statement that describes you best.",
    "hi": "A, B, या C लिखकर वह कथन चुनें जो आपका सबसे अच्छा वर्णन करता है।",
    "bn": "আপনার ক্ষেত্রে সবচেয়ে প্রযোজ্য কথাটি বেছে নিতে A, B, অথবা C দিয়ে উত্তর দিন।",
}

COMPLETION_MSG = {
    "en": "Thank you! Your responses have been recorded securely.",
    "hi": "धन्यवाद! आपके उत्तर सुरक्षित रूप से दर्ज कर लिए गए हैं।",
    "bn": "ধন্যবাদ! আপনার উত্তর নিরাপদে সংরক্ষিত হয়েছে।",
}

TIMEOUT_MSG = {
    "en": "Assessment time limit exceeded. Your partial responses have been recorded securely.",
    "hi": "मूल्यांकन की समय सीमा समाप्त हो गई है। आपके आंशिक उत्तर सुरक्षित रूप से दर्ज कर लिए गए हैं।",
    "bn": "মূল্যায়নের সময়সীমা অতিক্রম করেছে। আপনার আংশিক প্রতিক্রিয়া নিরাপদে রেকর্ড করা হয়েছে।",
}


@dataclass
class AssessmentSession:
    session_id: str
    user_id: str
    language: str
    item_ids: list[str]
    cohort: str = "Salaried"
    current_index: int = 0
    answers: dict[str, str] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False
    traits: dict[str, float] = field(default_factory=dict)
    awaiting_clarification: str | None = None
    start_time: float | None = None
    has_extended: bool = False

    @property
    def progress(self) -> float:
        if not self.item_ids:
            return 1.0
        return round(min(self.current_index, len(self.item_ids)) / len(self.item_ids), 2)


_sessions: dict[str, AssessmentSession] = {}


def _parse_groq_json(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


async def extract_open_ended_score(text: str, language: str) -> float:
    """LLM extraction with deterministic keyword fallback."""
    settings = get_settings()
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
        return score_open_ended_answer(text)

    prompt = (
        "Analyze this financial behavior response. Return ONLY JSON:\n"
        '{"responsibility_score": <float 0.0 to 1.0>}\n'
        f"Language hint: {language}\nText:\n{text}"
    )

    def _call() -> float:
        client = Groq(api_key=settings.GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial psychometric analyst. JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=100,
        )
        parsed = _parse_groq_json(completion.choices[0].message.content or "{}")
        return max(0.0, min(1.0, float(parsed.get("responsibility_score", 0.5))))

    try:
        return await asyncio.to_thread(_call)
    except Exception:
        return score_open_ended_answer(text)


def _render_item(item: Item, language: str) -> dict[str, Any]:
    bank = load_item_bank()
    payload = {
        "item_id": item.id,
        "construct": item.construct,
        "type": item.type,
        "prompt": item.prompt(language),
    }
    if item.type == "likert":
        labels = item.option_labels(language, bank["likert_labels"])
        payload["options"] = [
            {"value": opt, "label": labels[i] if i < len(labels) else opt}
            for i, opt in enumerate(item.options)
        ]
        payload["hint"] = LIKERT_HINT.get(language, LIKERT_HINT["en"])
    elif item.type == "forced_choice":
        option_texts_for_lang = item.option_texts.get(language, item.option_texts.get("en", {}))
        payload["options"] = [
            {"value": opt, "label": option_texts_for_lang.get(opt, opt)}
            for opt in item.options
        ]
        payload["hint"] = FORCED_CHOICE_HINT.get(language, FORCED_CHOICE_HINT["en"])
    return payload


def _validate_closed_answer(item: Item, answer: str, language: str) -> tuple[bool, str | None]:
    normalized = answer.strip().upper()
    if normalized in item.options:
        return True, None
    if item.type == "likert":
        if normalized.isdigit() and normalized in item.scoring_key:
            return True, None
        return False, LIKERT_HINT.get(language, LIKERT_HINT["en"])
    elif item.type == "forced_choice":
        return False, FORCED_CHOICE_HINT.get(language, FORCED_CHOICE_HINT["en"])
    return True, None


def create_session(user_id: str | None, language: str, cohort: str = "Salaried") -> AssessmentSession:
    lang = language if language in SUPPORTED_LANGUAGES else "en"
    items = load_items()
    session = AssessmentSession(
        session_id=str(uuid.uuid4()),
        user_id=user_id or str(uuid.uuid4()),
        language=lang,
        cohort=cohort,
        item_ids=[item.id for item in items],
        start_time=None,
    )
    # Determine translated cohort string
    translations = {
        "Salaried": {"en": "Salaried person", "hi": "वेतनभोगी", "bn": "বেতনভুক্ত ব্যক্তি"},
        "GigWorker": {"en": "Gig Worker", "hi": "गिग वर्कर", "bn": "গিগ কর্মী"},
        "Student": {"en": "Student", "hi": "छात्र", "bn": "শিক্ষার্থী"},
        "Vendor": {"en": "Micro Enterprise / Vendor", "hi": "विक्रेता", "bn": "ক্ষুদ্র বিক্রেতা"},
        "Farmer": {"en": "Farmer", "hi": "किसान", "bn": "কৃষক"},
        "Homemaker": {"en": "Homemaker", "hi": "गृहिणी", "bn": "গৃহিণী"},
    }
    cohort_translated = translations.get(cohort, translations["Salaried"]).get(lang, cohort)
    greeting_text = GREETINGS[lang].format(cohort=cohort_translated)

    session.transcript.append(
        {"role": "agent", "type": "greeting", "text": greeting_text, "language": lang}
    )
    _sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> AssessmentSession | None:
    return _sessions.get(session_id)


def start_response(session: AssessmentSession) -> dict[str, Any]:
    if session.completed:
        return _completion_response(session)

    first_item = load_items()[0]
    item_payload = _render_item(first_item, session.language)

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "language": session.language,
        "message": session.transcript[0]["text"],
        "item": item_payload,
        "progress": session.progress,
        "completed": False,
    }


def begin_timer(session_id: str) -> dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise ValueError("Session not found")
    if session.completed:
        return _completion_response(session)

    session.start_time = time.time()
    first_item = load_items()[0]
    item_payload = _render_item(first_item, session.language)
    session.transcript.append({"role": "agent", "type": "item", **item_payload})
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "completed": False,
        "needs_clarification": False,
        "message": item_payload["prompt"],
        "item": item_payload,
        "progress": session.progress,
        "traits": session.traits,
        "survey_payload": None,
    }


async def submit_answer(session_id: str, item_id: str, answer: str) -> dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise ValueError("Session not found")
    if session.completed:
        return _completion_response(session)

    # Backend timeout enforcement
    settings = get_settings()
    limit = settings.PSYCHOMETRIC_TIME_LIMIT_SECONDS
    if session.has_extended:
        limit += settings.PSYCHOMETRIC_EXTENSION_SECONDS

    buffer = 15  # 15s buffer for network latency
    if session.start_time is not None:
        elapsed = time.time() - session.start_time
        if elapsed > (limit + buffer):
            return await force_timeout_session(session_id)

    item = next((i for i in load_items() if i.id == item_id), None)
    if item is None:
        raise ValueError(f"Unknown item {item_id}")

    normalized_answer = answer.strip()
    if item.type == "forced_choice":
        normalized_answer = normalized_answer.upper()

    session.transcript.append(
        {"role": "user", "item_id": item_id, "text": normalized_answer, "language": session.language}
    )

    if item.type in ("likert", "forced_choice"):
        valid, hint = _validate_closed_answer(item, normalized_answer, session.language)
        if not valid:
            session.awaiting_clarification = item_id
            clarify = hint or (LIKERT_HINT[session.language] if item.type == "likert" else FORCED_CHOICE_HINT[session.language])
            session.transcript.append({"role": "agent", "type": "clarify", "text": clarify})
            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "completed": False,
                "needs_clarification": True,
                "message": clarify,
                "item": _render_item(item, session.language),
                "progress": session.progress,
            }

    if item.type == "open_ended" and len(normalized_answer) < 5:
        clarify = {
            "en": "Please share a bit more detail (at least one full sentence).",
            "hi": "कृपया थोड़ा और विस्तार से बताएं (कम से कम एक पूरा वाक्य)।",
            "bn": "অনুগ্রহ করে আরও একটু বিস্তারিত লিখুন (অন্তত একটি সম্পূর্ণ বাক্য)।",
        }[session.language]
        session.awaiting_clarification = item_id
        session.transcript.append({"role": "agent", "type": "clarify", "text": clarify})
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "completed": False,
            "needs_clarification": True,
            "message": clarify,
            "item": _render_item(item, session.language),
            "progress": session.progress,
        }

    session.answers[item_id] = normalized_answer
    session.awaiting_clarification = None
    session.current_index += 1

    if session.current_index >= len(session.item_ids):
        return await _finalize_session(session)

    next_item = load_items()[session.current_index]
    item_payload = _render_item(next_item, session.language)
    session.transcript.append({"role": "agent", "type": "item", **item_payload})
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "completed": False,
        "needs_clarification": False,
        "message": item_payload["prompt"],
        "item": item_payload,
        "progress": session.progress,
    }


async def force_timeout_session(session_id: str) -> dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise ValueError("Session not found")
    if session.completed:
        return _completion_response(session)

    # In case of timeout, extract scores from whatever answers are present.
    # Unanswered constructs will automatically receive 0.5 default in compute_trait_scores.
    open_scores: dict[str, float] = {}
    for item in load_items():
        if item.type == "open_ended" and item.id in session.answers:
            open_scores[item.id] = await extract_open_ended_score(
                session.answers[item.id],
                session.language,
            )

    traits = compute_trait_scores(session.answers, open_scores)
    session.traits = traits
    session.completed = True
    session.transcript.append(
        {
            "role": "agent",
            "type": "completion",
            "text": TIMEOUT_MSG.get(session.language, TIMEOUT_MSG["en"]),
        }
    )
    return _completion_response(session)


def extend_session(session_id: str) -> dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise ValueError("Session not found")
    if not session.has_extended:
        session.has_extended = True
    return {
        "session_id": session.session_id,
        "has_extended": session.has_extended,
    }


async def _finalize_session(session: AssessmentSession) -> dict[str, Any]:
    open_scores: dict[str, float] = {}
    for item in load_items():
        if item.type == "open_ended" and item.id in session.answers:
            open_scores[item.id] = await extract_open_ended_score(
                session.answers[item.id],
                session.language,
            )

    traits = compute_trait_scores(session.answers, open_scores)
    session.traits = traits
    session.completed = True
    session.transcript.append(
        {
            "role": "agent",
            "type": "completion",
            "text": COMPLETION_MSG[session.language],
        }
    )
    return _completion_response(session)


def _completion_response(session: AssessmentSession, message: str | None = None) -> dict[str, Any]:
    payload = None
    if session.completed and session.traits:
        payload = format_assessment_payload(
            session.user_id,
            session.language,
            session.cohort,
            session.answers,
            session.transcript,
            session.traits,
        )

    if message is None:
        message = COMPLETION_MSG.get(session.language, COMPLETION_MSG["en"])
        if session.transcript:
            last = session.transcript[-1]
            if last.get("role") == "agent" and last.get("type") == "completion" and "text" in last:
                message = last["text"]

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "completed": session.completed,
        "needs_clarification": False,
        "message": message,
        "progress": 1.0,
        "traits": session.traits,
        "survey_payload": payload,
    }


def build_survey_payload_for_ingest(session: AssessmentSession) -> dict[str, Any]:
    if not session.completed:
        raise ValueError("Session not completed")
    return format_assessment_payload(
        session.user_id,
        session.language,
        session.cohort,
        session.answers,
        session.transcript,
        session.traits,
    )
