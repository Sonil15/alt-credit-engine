"""Tests for psychometric item bank, scoring, and extraction."""

from psychometric.bank import load_items, validate_item_bank
from psychometric.scoring import (
    compute_response_validity,
    compute_trait_scores,
    score_likert_answer,
    score_open_ended_answer,
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
        "con_1": "5",
        "con_2": "5",
        "loc_1": "5",
        "loc_2": "1",
        "fse_1": "4",
        "fse_2": "4",
        "pb_1": "1",
        "pb_2": "1",
        "debt_1": "5",
        "debt_2": "1",
        "open_1": "I pay my bills and rent first before any other spending.",
    }
    t1 = compute_trait_scores(answers)
    t2 = compute_trait_scores(answers)
    assert t1 == t2


def test_reverse_key_scoring():
    item = next(i for i in load_items() if i.id == "loc_2")
    assert score_likert_answer(item, "1") == 1.0
    assert score_likert_answer(item, "5") == 0.0


def test_consistency_pair_validity():
    consistent = {
        "con_1": "5",
        "con_2": "5",
        "pb_1": "1",
        "pb_2": "1",
    }
    inconsistent = {
        "con_1": "5",
        "con_2": "1",
        "pb_1": "1",
        "pb_2": "5",
    }
    v_good = compute_response_validity(consistent, load_items())
    v_bad = compute_response_validity(inconsistent, load_items())
    assert v_good > v_bad


def test_open_ended_extraction_fallback():
    responsible = score_open_ended_answer("I pay bills and save money every month.")
    avoidant = score_open_ended_answer("I avoid checking debt and delay payments.")
    assert responsible > avoidant
