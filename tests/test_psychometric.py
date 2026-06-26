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
from psychometric.session import (
    create_session,
    get_session,
    submit_answer,
    force_timeout_session,
    extend_session,
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


@pytest.mark.asyncio
async def test_session_time_tracking():
    session = create_session(user_id="test-time-user", language="en")
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
