"""Deterministic fallback extraction + confidence routing for the business profiler."""

import pytest

from core.business_profile import (
    EXTRACTION_CONFIDENCE_THRESHOLD,
    PURPOSES_BY_COHORT,
    _resolve_extraction,
    business_features_applicable,
    extract_business_profile,
    fallback_extract_business_profile,
    turnover_income_consistency,
)


def test_fallback_extracts_english_description():
    profile = fallback_extract_business_profile(
        "I have run a vegetable stall in the market for 8 years, "
        "earning about 40,000 rupees a month with 2 helpers."
    )
    assert profile["sector"] == "retail"
    assert profile["years_in_business"] == 8.0
    assert profile["monthly_turnover"] == 40000.0
    assert profile["employees"] == 2


def test_fallback_extracts_hindi_with_lakh():
    profile = fallback_extract_business_profile(
        "मैं 12 साल से किराना दुकान चलाता हूँ, महीने में करीब 1 लाख की बिक्री होती है"
    )
    assert profile["sector"] == "retail"
    assert profile["years_in_business"] == 12.0
    assert profile["monthly_turnover"] == 100000.0


def test_fallback_extracts_bengali_farming():
    profile = fallback_extract_business_profile(
        "আমি ৫ বছর ধরে ধান চাষ করি, ফসল কাটার মৌসুমে আয় বেশি হয়"
    )
    assert profile["sector"] == "agriculture"
    assert profile["years_in_business"] == 5.0
    assert profile["seasonality"] == "high"


def test_fallback_ignores_amount_without_income_context():
    # A bare number with no earn/turnover context must not become turnover.
    profile = fallback_extract_business_profile("I want a loan of 2 lakh for my shop")
    assert profile["monthly_turnover"] is None


def test_fallback_is_deterministic():
    text = "chai stall for 3 years, kamai 15,000 per month"
    assert fallback_extract_business_profile(text) == fallback_extract_business_profile(text)


def test_low_confidence_routes_to_fallback():
    parsed = {
        "sector": "retail",
        "years_in_business": 8,
        "monthly_turnover": 40000,
        "confidence": EXTRACTION_CONFIDENCE_THRESHOLD - 0.1,
    }
    profile, confidence, method = _resolve_extraction(parsed, "8 years vegetable stall, earn 40000")
    assert method == "fallback"
    assert profile["years_in_business"] == 8.0  # fallback re-reads the text


def test_confident_llm_response_is_used_and_sanitized():
    parsed = {
        "sector": "Retail ",
        "years_in_business": 8,
        "monthly_turnover": 40000,
        "seasonality": "HIGH",
        "employees": 2,
        "confidence": 0.9,
    }
    profile, confidence, method = _resolve_extraction(parsed, "whatever")
    assert method == "llm"
    assert confidence == 0.9
    assert profile["sector"] == "retail"
    assert profile["seasonality"] == "high"


def test_garbage_llm_values_become_none():
    parsed = {
        "sector": 42,
        "years_in_business": -5,
        "monthly_turnover": "lots",
        "seasonality": "sometimes",
        "employees": 999999,
        "confidence": 0.9,
    }
    profile, _, method = _resolve_extraction(parsed, "no signal here whatsoever xyz")
    # Every field rejected -> falls back rather than returning an empty llm read.
    assert method == "fallback"


@pytest.mark.asyncio
async def test_extract_without_api_key_uses_fallback(monkeypatch):
    from core.config import get_settings

    monkeypatch.setattr(get_settings(), "GROQ_API_KEY", "", raising=False)
    profile, confidence, method = await extract_business_profile(
        "tailoring shop for 4 years, income 20,000 monthly", "en"
    )
    assert method == "fallback"
    assert profile["sector"] == "services"
    assert profile["years_in_business"] == 4.0


def test_turnover_income_consistency_bounds():
    # Default / Salaried (expected ratio = 0.90)
    assert turnover_income_consistency(40000, 40000) == 1.0
    assert turnover_income_consistency(40000, 36000) == 1.0
    assert turnover_income_consistency(40000, 18000) == 0.5
    assert turnover_income_consistency(40000, 80000) == 0.5
    assert turnover_income_consistency(0, 40000) == 0.0
    assert turnover_income_consistency(40000, 0) == 0.0

    # Farmer (expected ratio = 0.20)
    assert turnover_income_consistency(100000, 20000, "Farmer") == 1.0
    assert turnover_income_consistency(100000, 10000, "Farmer") == 0.5
    assert turnover_income_consistency(40000, 80000, "Farmer") == 0.5

    # Vendor (expected ratio = 0.40)
    assert turnover_income_consistency(100000, 40000, "Vendor") == 1.0
    assert turnover_income_consistency(100000, 20000, "Vendor") == 0.5

    # GigWorker (expected ratio = 0.80)
    assert turnover_income_consistency(100000, 80000, "GigWorker") == 1.0
    assert turnover_income_consistency(100000, 40000, "GigWorker") == 0.5

    # Tier 1 Vintage (< 0.5 years): always fully consistent
    assert turnover_income_consistency(40000, 0, "Vendor", business_vintage_years=0.2) == 1.0
    assert turnover_income_consistency(40000, 1000, "Vendor", business_vintage_years=0.4) == 1.0

    # Tier 2 Vintage (0.5 <= vintage < 1.5 years): grace factor of 50%
    # Vendor expected digital ratio = 0.40. With vintage=1.0, it is adjusted to 0.20.
    assert turnover_income_consistency(100000, 20000, "Vendor", business_vintage_years=1.0) == 1.0
    assert turnover_income_consistency(100000, 10000, "Vendor", business_vintage_years=1.0) == 0.5
    # Default/Salaried expected digital ratio = 0.90. With vintage=0.8, it is adjusted to 0.45.
    assert turnover_income_consistency(100000, 45000, "Salaried", business_vintage_years=0.8) == 1.0
    assert turnover_income_consistency(100000, 22500, "Salaried", business_vintage_years=0.8) == 0.5


def test_purposes_map_covers_all_cohorts():
    for cohort in ("Salaried", "GigWorker", "Student", "Vendor", "Farmer", "Homemaker"):
        assert PURPOSES_BY_COHORT[cohort], cohort


def test_business_features_not_applicable_for_student_laptop():
    assert business_features_applicable("Student", "device_equipment") is False


def test_business_features_applicable_for_vendor():
    assert business_features_applicable("Vendor", "working_capital") is True


def test_business_features_applicable_for_homemaker_home_business():
    assert business_features_applicable("Homemaker", "small_home_business") is True
    assert business_features_applicable("Homemaker", "household") is False


def test_business_features_applicable_when_profile_submitted():
    assert business_features_applicable("Student", "device_equipment", has_business_profile=True) is True
