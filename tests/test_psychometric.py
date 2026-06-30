"""Tests for psychometric item bank, scoring, and extraction."""

import pytest
import time
from core.config import get_settings
from psychometric.bank import load_items, validate_item_bank
from psychometric.scoring import (
    compute_response_validity,
    compute_trait_scores,
    score_likert_answer,
    score_open_ended_answer,
)
from psychometric import session as session_mod
from psychometric.session import (
    create_session,
    get_session,
    submit_answer,
    force_timeout_session,
    extend_session,
    begin_timer,
    extract_open_ended_score,
    _resolve_open_ended_score,
)


def test_item_bank_valid():
    errors = validate_item_bank()
    assert errors == []


def test_all_items_have_three_translations():
    for item in load_items():
        assert "en" in item.text and item.text["en"].strip()
        assert "hi" in item.text and item.text["hi"].strip()
        assert "bn" in item.text and item.text["bn"].strip()


def test_scoring_deterministic():
    answers = {
        "fc_1": "A",
        "fc_2": "B",
        "fc_3": "C",
        "fc_4": "A",
        "fc_5": "B",
        "open_1": "I pay my bills and rent first before any other spending.",
    }
    t1 = compute_trait_scores(answers)
    t2 = compute_trait_scores(answers)
    assert t1 == t2


def test_forced_choice_scoring():
    item = next(i for i in load_items() if i.id == "fc_1")
    assert item.scoring_key["A"]["conscientiousness"] == 1.0
    assert item.scoring_key["A"]["locus_of_control"] == 0.0
    assert item.scoring_key["A"]["financial_self_efficacy"] == 0.0


def test_consistency_pair_validity():
    # Since forced choice doesn't use consistency pairs, response_validity should default to 1.0
    answers = {
        "fc_1": "A",
        "fc_2": "B",
    }
    v = compute_response_validity(answers, load_items())
    assert v == 1.0


def test_open_ended_extraction_fallback():
    responsible = score_open_ended_answer("I pay bills and save money every month.")
    avoidant = score_open_ended_answer("I avoid checking debt and delay payments.")
    assert responsible > avoidant


def test_open_ended_fallback_hindi():
    # Pays bills/rent first and saves vs. ignores debt, delays payment, worries.
    responsible = score_open_ended_answer(
        "मैं हर महीने बिल और किराया पहले भरता हूँ और बचत करता हूँ।"
    )
    avoidant = score_open_ended_answer(
        "मैं कर्ज नजरअंदाज करता हूँ, भुगतान में देरी करता हूँ और चिंता करता हूँ।"
    )
    # A Hindi answer must produce real signal, not collapse to neutral 0.5.
    assert responsible > 0.5
    assert avoidant < 0.5
    assert responsible > avoidant


def test_open_ended_fallback_bengali():
    # Pays bills/rent first and saves vs. ignores debt, delays, worries.
    responsible = score_open_ended_answer(
        "আমি প্রতি মাসে প্রথমে বিল ও ভাড়া পরিশোধ করি এবং সঞ্চয় করি।"
    )
    avoidant = score_open_ended_answer(
        "আমি ঋণ উপেক্ষা করি, সব কিছুতে দেরি করি এবং দুশ্চিন্তা করি।"
    )
    assert responsible > 0.5
    assert avoidant < 0.5
    assert responsible > avoidant


def test_open_ended_fallback_negation():
    # Negated responsible cues must not read as responsible in any language.
    en_plain = score_open_ended_answer("I pay bills and save money every month.")
    en_neg = score_open_ended_answer("I do not pay bills and I do not save money.")
    assert en_neg < 0.5
    assert en_neg < en_plain

    hi_plain = score_open_ended_answer("मैं बचत करता हूँ और बिल भरता हूँ।")
    hi_neg = score_open_ended_answer("मैं बचत नहीं करता और बिल नहीं भरता।")
    assert hi_neg < 0.5
    assert hi_neg < hi_plain

    bn_plain = score_open_ended_answer("আমি সঞ্চয় করি এবং বিল পরিশোধ করি।")
    bn_neg = score_open_ended_answer("আমি সঞ্চয় করি না এবং বিল পরিশোধ করি না।")
    assert bn_neg < 0.5
    assert bn_neg < bn_plain


def test_resolve_uses_score_when_confident():
    text = "I pay rent and utilities first, then groceries."
    assert _resolve_open_ended_score(
        {"responsibility_score": 0.82, "confidence": 0.9}, text
    ) == 0.82


def test_resolve_trusts_score_when_confidence_missing():
    # The score is the primary signal; absent/garbled confidence is treated as confident.
    text = "I always clear my bills first."
    assert _resolve_open_ended_score({"responsibility_score": 0.7}, text) == 0.7


def test_resolve_falls_back_on_low_confidence():
    # Low self-reported confidence must defer to the deterministic scorer, not 0.5.
    text = "I avoid checking debt and delay payments."
    resolved = _resolve_open_ended_score(
        {"responsibility_score": 0.95, "confidence": 0.1}, text
    )
    assert resolved == score_open_ended_answer(text)


def test_resolve_falls_back_on_missing_or_invalid_score():
    text = "I pay bills and save money every month."
    fallback = score_open_ended_answer(text)
    assert _resolve_open_ended_score({}, text) == fallback  # missing field (was silent 0.5)
    assert _resolve_open_ended_score({"responsibility_score": "abc"}, text) == fallback
    assert _resolve_open_ended_score({"responsibility_score": 1.7}, text) == fallback  # out of range


@pytest.mark.asyncio
async def test_extract_open_ended_score_caches(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")  # force deterministic fallback path
    session_mod._open_ended_cache.clear()
    text = "I always pay rent and utilities first."

    first = await extract_open_ended_score(text, "en")
    assert (settings.GROQ_MODEL, "en", text.strip()) in session_mod._open_ended_cache
    second = await extract_open_ended_score(text, "en")
    assert first == second == score_open_ended_answer(text)


@pytest.mark.asyncio
async def test_session_time_tracking():
    session = create_session(user_id="test-time-user", language="en")
    begin_timer(session.session_id)
    assert session.start_time is not None
    assert session.start_time <= time.time()
    assert session.has_extended is False


@pytest.mark.asyncio
async def test_force_timeout_session():
    session = create_session(user_id="test-timeout-user", language="en")
    first_item_id = session.item_ids[0]
    await submit_answer(session.session_id, first_item_id, "A")
    
    assert session.completed is False
    
    result = await force_timeout_session(session.session_id)
    assert result["completed"] is True
    assert "Assessment time limit exceeded" in result["message"]
    assert session.completed is True
    assert "conscientiousness" in session.traits
    assert session.traits["conscientiousness"] == 1.0
    assert session.traits["present_bias"] == 0.5


@pytest.mark.asyncio
async def test_extend_session_and_backend_enforcement(monkeypatch):
    session = create_session(user_id="test-extend-user", language="en")
    assert session.has_extended is False
    
    extend_result = extend_session(session.session_id)
    assert extend_result["has_extended"] is True
    assert session.has_extended is True
    
    settings = get_settings()
    monkeypatch.setattr(settings, "PSYCHOMETRIC_TIME_LIMIT_SECONDS", 1)
    monkeypatch.setattr(settings, "PSYCHOMETRIC_EXTENSION_SECONDS", 1)
    
    # 30 seconds ago is greater than the 2s limit + 15s latency buffer = 17s
    session.start_time = time.time() - 30
    
    first_item_id = session.item_ids[0]
    result = await submit_answer(session.session_id, first_item_id, "A")
    assert result["completed"] is True
    assert "Assessment time limit exceeded" in result["message"]
