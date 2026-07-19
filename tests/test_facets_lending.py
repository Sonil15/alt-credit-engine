import pandas as pd

from convergence.lending import interest_rate_for_pd, recommend_terms, tenure_for_score
from convergence.facets import (
    FACETS,
    compute_confidence,
    compute_norm_stats,
    compute_facet_scores,
    feature_goodness,
)


def _population() -> pd.DataFrame:
    # A safe borrower and a risky borrower so percentiles have spread.
    return pd.DataFrame(
        [
            {
                "user_id": "safe",
                "avg_days_late": 0.0,
                "missed_payments_count": 0.0,
                "necessity_ratio": 0.8,
                "avg_merchant_rating": 4.8,
                "monthly_spend_volatility": 1000.0,
                "spatial_variance_score": 0.5,
                "anchor_count": 2.0,
                "monthly_income_mean": 50000.0,
                "cashflow_volatility": 1000.0,
                "resilience_coefficient": 0.9,
                "is_stationary": 1.0,
                "trend_slope": 0.05,
                "conscientiousness": 0.9,
                "locus_of_control": 0.9,
                "financial_self_efficacy": 0.9,
                "present_bias": 0.1,
                "debt_attitude": 0.9,
            },
            {
                "user_id": "risky",
                "avg_days_late": 18.0,
                "missed_payments_count": 6.0,
                "necessity_ratio": 0.3,
                "avg_merchant_rating": 2.6,
                "monthly_spend_volatility": 40000.0,
                "spatial_variance_score": 45.0,
                "anchor_count": 1.0,
                "monthly_income_mean": 9000.0,
                "cashflow_volatility": 40000.0,
                "resilience_coefficient": 0.1,
                "is_stationary": 0.0,
                "trend_slope": -0.05,
                "conscientiousness": 0.2,
                "locus_of_control": 0.2,
                "financial_self_efficacy": 0.2,
                "present_bias": 0.9,
                "debt_attitude": 0.2,
            },
        ]
    )


def test_feature_goodness_direction():
    assert feature_goodness(10, 0, 10, "high") == 1.0
    assert feature_goodness(10, 0, 10, "low") == 0.0
    assert feature_goodness(5, 0, 10, "high") == 0.5
    # no spread -> neutral
    assert feature_goodness(5, 3, 3, "high") == 0.5


def test_safe_borrower_outscores_risky_on_every_facet():
    pop = _population()
    stats = compute_norm_stats(pop)
    safe = compute_facet_scores(pop.iloc[0], stats)
    risky = compute_facet_scores(pop.iloc[1], stats)
    assert len(safe) == 5
    for s, r in zip(safe, risky, strict=True):
        assert s["score"] >= r["score"]
        assert isinstance(s["score"], float)  # JSON-safe, not numpy


def test_confidence_detects_thin_file():
    row = pd.Series({"conscientiousness": 0.7})  # only psychometric present
    stats = compute_norm_stats(pd.DataFrame([row]))
    facets = compute_facet_scores(row, stats)
    conf = compute_confidence(facets)
    assert conf["thin_file"] is True
    assert conf["confidence_pct"] < 100.0
    assert conf["facets_with_data"] >= 1


def test_interest_rate_rises_with_risk():
    assert interest_rate_for_pd(0.05) < interest_rate_for_pd(0.5)
    assert interest_rate_for_pd(0.0) >= 11.0
    assert interest_rate_for_pd(1.0) <= 26.0


def test_tenure_bands_by_score():
    assert tenure_for_score(800) == 36
    assert tenure_for_score(400) == 12


def test_rejected_applicant_gets_no_offer():
    row = pd.Series({"monthly_income_mean": 30000.0})
    terms = recommend_terms(0.8, 420, "REJECT", row)
    assert terms["eligible"] is False
    assert terms["max_loan_amount"] == 0.0


def test_approved_applicant_gets_priced_offer():
    row = pd.Series({"monthly_income_mean": 40000.0})
    terms = recommend_terms(0.08, 780, "APPROVE", row)
    assert terms["eligible"] is True
    assert terms["max_loan_amount"] > 0
    assert 11.0 <= terms["interest_rate_pct"] <= 26.0
    assert terms["monthly_emi"] > 0


def test_review_offer_is_more_conservative_than_approve():
    row = pd.Series({"monthly_income_mean": 40000.0})
    approve = recommend_terms(0.2, 760, "APPROVE", row)
    review = recommend_terms(0.2, 700, "REVIEW", row)
    assert review["max_loan_amount"] < approve["max_loan_amount"]


def test_cohort_aware_scoring():
    pop = _population()
    stats = compute_norm_stats(pop)

    # 1. Student Cohort
    row_student = pd.Series({
        "cohort_code": 2.0,
        "upi_spend_consistency": 0.9,
        "small_dues_payment_promptness": 8.5
    })
    student_facets = compute_facet_scores(row_student, stats)
    assert len(student_facets) == 3
    assert any(p["key"] == "campus_transaction_behavior" for p in student_facets)
    conf_student = compute_confidence(student_facets, cohort="Student")
    assert conf_student["thin_file"] is True  # still missing location/survey data in this Series

    # 2. Homemaker Cohort
    row_homemaker = pd.Series({"cohort_code": 5.0, "utility_payment_consistency": 0.95})
    homemaker_facets = compute_facet_scores(row_homemaker, stats)
    assert len(homemaker_facets) == 4
    assert any(p["key"] == "household_reliability" for p in homemaker_facets)

    # 3. Farmer Cohort
    row_farmer = pd.Series({"cohort_code": 4.0, "harvest_income_spike": 5.0})
    farmer_facets = compute_facet_scores(row_farmer, stats)
    assert len(farmer_facets) == 4
    assert any(p["key"] == "agricultural_seasonality" for p in farmer_facets)
    assert any(p["key"] == "business_credentials" for p in farmer_facets)


def test_optional_facets_and_confidence_offset():
    pop = _population()
    stats = compute_norm_stats(pop)

    # Farmer cohort: expected facets include agricultural_seasonality (3), location_stability (2), psychometric_character (10), and business_credentials (5) -> 20 features
    # Let's provide a subset: only 12 features out of 20 (missing location_stability, business_credentials, and enam_receipt_volume)
    row_farmer_no_opt = pd.Series({
        "cohort_code": 4.0,  # Farmer
        "harvest_income_spike": 5.0,
        "input_purchase_consistency": 2.0,
        "conscientiousness": 0.9,
        "locus_of_control": 0.9,
        "financial_self_efficacy": 0.9,
        "present_bias": 0.1,
        "debt_attitude": 0.9,
        "risk_tolerance": 0.1,
        "delayed_gratification": 0.9,
        "honesty": 0.9,
        "cognitive_reflection": 0.9,
        "resourcefulness": 0.9,
        # missing location_stability features (spatial_variance_score, anchor_count)
        # missing business_credentials features
        # missing enam_receipt_volume
    })
    farmer_facets_no_opt = compute_facet_scores(row_farmer_no_opt, stats)
    # Should not include spending_behaviour since it's not expected and not provided
    assert not any(p["key"] == "spending_behaviour" for p in farmer_facets_no_opt)

    conf_no_opt = compute_confidence(farmer_facets_no_opt, cohort="Farmer")
    # Features with data = 13 (agricultural_seasonality: 3 due to hybrid logic, psychometric_character: 10)
    # Expected features = 20 (including 3 from agricultural_seasonality + 2 from location_stability + 5 from business_credentials)
    # confidence pct should be round(100 * 13 / 20) = 65.0%
    assert conf_no_opt["confidence_pct"] == 65.0
    assert conf_no_opt["features_total"] == 20

    # Farmer cohort with optional data: same expected features, but now also provides e-commerce (spending_behaviour)
    # spending_behaviour has 4 features now (necessity_ratio, avg_merchant_rating, monthly_spend_volatility, sms_spend_total)
    # let's provide necessity_ratio, avg_merchant_rating, monthly_spend_volatility (3 features)
    row_farmer_with_opt = pd.Series({
        "cohort_code": 4.0,  # Farmer
        "harvest_income_spike": 5.0,
        "input_purchase_consistency": 2.0,
        "conscientiousness": 0.9,
        "locus_of_control": 0.9,
        "financial_self_efficacy": 0.9,
        "present_bias": 0.1,
        "debt_attitude": 0.9,
        "risk_tolerance": 0.1,
        "delayed_gratification": 0.9,
        "honesty": 0.9,
        "cognitive_reflection": 0.9,
        "resourcefulness": 0.9,
        # missing location_stability features
        # provided optional features: spending_behaviour (3 features out of 4)
        "necessity_ratio": 0.8,
        "avg_merchant_rating": 4.5,
        "monthly_spend_volatility": 1500.0,
    })
    farmer_facets_with_opt = compute_facet_scores(row_farmer_with_opt, stats)
    # Should dynamically include spending_behaviour because data is present
    assert any(p["key"] == "spending_behaviour" for p in farmer_facets_with_opt)

    conf_with_opt = compute_confidence(farmer_facets_with_opt, cohort="Farmer")
    # Features with data = 13 (expected) + 4 (optional due to hybrid logic) = 17
    # Expected features (denominator) = 20
    # confidence pct = round(100 * 17 / 20, 1) = 85.0%
    assert conf_with_opt["confidence_pct"] == 85.0
    assert conf_with_opt["features_total"] == 20


def test_hybrid_confidence_logic():
    pop = _population()
    stats = compute_norm_stats(pop)

    # Student Cohort: expected facets are location_stability (2), psychometric_character (10), campus_transaction_behavior (3) -> total 15 features.
    # Provide only 1 feature of campus_transaction_behavior (upi_spend_consistency = 0.9)
    # Location stability has no data, psychometric has no data.
    # Since campus_transaction_behavior has data, the hybrid logic should count all 3 of its features as having data.
    row = pd.Series({
        "cohort_code": 2.0,  # Student
        "upi_spend_consistency": 0.9,
    })
    facets = compute_facet_scores(row, stats)
    conf = compute_confidence(facets, cohort="Student")

    # campus_transaction_behavior is non-psychometric, so it gets full credit of 3 features.
    # psychometric: 0 features.
    # location_stability: 0 features.
    # features_with_data should be 3, not 1.
    assert conf["features_with_data"] == 3
    # confidence_pct = 100 * 3 / 15 = 20.0
    assert conf["confidence_pct"] == 20.0

    # Let's also check that psychometric_character is still feature-by-feature.
    # Provide only conscientiousness (1 feature of psychometric) and no other data.
    # Expected: psychometric_character has data, but only conscientiousness is counted.
    row_psych = pd.Series({
        "cohort_code": 2.0,  # Student
        "conscientiousness": 0.8,
    })
    facets_psych = compute_facet_scores(row_psych, stats)
    conf_psych = compute_confidence(facets_psych, cohort="Student")

    # psychometric: 1 feature (conscientiousness).
    # location_stability: 0 features.
    # campus_transaction_behavior: 0 features.
    # features_with_data should be 1.
    assert conf_psych["features_with_data"] == 1
    # confidence_pct = 100 * 1 / 15 = 6.7
    assert conf_psych["confidence_pct"] == 6.7


def test_homemaker_dynamic_business_expectation():
    pop = _population()
    stats = compute_norm_stats(pop)

    # 1. Homemaker without business purpose:
    # expected facets: telecom_reliability (3), location_stability (2), psychometric_character (10), household_reliability (2) -> 17 features.
    row = pd.Series({
        "cohort_code": 5.0,  # Homemaker
        "utility_payment_consistency": 0.95,
        "grocery_spend_stability": 0.95,
        # location stability
        "spatial_variance_score": 0.1,
        "anchor_count": 5.0,
        # psychometric
        "conscientiousness": 0.9,
        "locus_of_control": 0.9,
        "financial_self_efficacy": 0.9,
        "present_bias": 0.1,
        "debt_attitude": 0.9,
        "risk_tolerance": 0.1,
        "delayed_gratification": 0.9,
        "honesty": 0.9,
        "cognitive_reflection": 0.9,
        "resourcefulness": 0.9,
        # telecom
        "avg_days_late": 1.0,
        "missed_payments_count": 0.0,
        "sms_bill_delay": 0.0,
    })

    # expect_business_profile = False
    facets = compute_facet_scores(row, stats, cohort="Homemaker", expect_business_profile=False)
    conf = compute_confidence(facets, cohort="Homemaker", expect_business_profile=False)
    assert conf["features_total"] == 17
    assert conf["confidence_pct"] == 100.0

    # 2. Homemaker with business purpose:
    # expected facets should include business_credentials (5) -> 22 features.
    # Because we didn't provide business credentials, confidence should drop.
    facets_biz = compute_facet_scores(row, stats, cohort="Homemaker", expect_business_profile=True)
    conf_biz = compute_confidence(facets_biz, cohort="Homemaker", expect_business_profile=True)
    assert conf_biz["features_total"] == 22
    assert conf_biz["confidence_pct"] == round(100 * 17 / 22, 1)  # 77.3%


def test_msme_dynamic_capacity_multipliers():
    from convergence.lending import get_msme_capacity_multiplier

    # Verify Farmer: digital ratio 0.20 -> 1/0.20 = 5.0 -> capped at 3.0
    assert get_msme_capacity_multiplier("Farmer") == 3.0

    # Verify Vendor: digital ratio 0.40 -> 1/0.40 = 2.5
    assert get_msme_capacity_multiplier("Vendor") == 2.5

    # Verify Homemaker: same expected digital ratio as Vendor -> 0.40 -> 2.5
    assert get_msme_capacity_multiplier("Homemaker") == 2.5

    # Verify GigWorker: digital ratio 0.80 -> 1/0.80 = 1.25
    assert get_msme_capacity_multiplier("GigWorker") == 1.25

    # Verify Default/Unknown cohort (e.g. None or Salaried)
    assert get_msme_capacity_multiplier(None) == 1.5
    assert get_msme_capacity_multiplier("Salaried") == 1.5

    # Test recommendation logic incorporates this
    row_farmer = pd.Series({"monthly_income_mean": 20000.0, "borrower_type": 1.0})
    terms_farmer = recommend_terms(0.1, 700, "APPROVE", row_farmer, cohort="Farmer")
    
    row_gig = pd.Series({"monthly_income_mean": 20000.0, "borrower_type": 1.0})
    terms_gig = recommend_terms(0.1, 700, "APPROVE", row_gig, cohort="GigWorker")

    # Farmer has higher capacity due to lower digital ratio (3x vs 1.25x), so higher max loan amount
    assert terms_farmer["max_loan_amount"] > terms_gig["max_loan_amount"]

