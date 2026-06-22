import pandas as pd

from convergence.lending import interest_rate_for_pd, recommend_terms, tenure_for_score
from convergence.pillars import (
    PILLARS,
    compute_confidence,
    compute_norm_stats,
    compute_pillar_scores,
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


def test_safe_borrower_outscores_risky_on_every_pillar():
    pop = _population()
    stats = compute_norm_stats(pop)
    safe = compute_pillar_scores(pop.iloc[0], stats)
    risky = compute_pillar_scores(pop.iloc[1], stats)
    assert len(safe) == len(PILLARS) == 5
    for s, r in zip(safe, risky, strict=True):
        assert s["score"] >= r["score"]
        assert isinstance(s["score"], float)  # JSON-safe, not numpy


def test_confidence_detects_thin_file():
    row = pd.Series({"conscientiousness": 0.7})  # only psychometric present
    stats = compute_norm_stats(pd.DataFrame([row]))
    pillars = compute_pillar_scores(row, stats)
    conf = compute_confidence(pillars)
    assert conf["thin_file"] is True
    assert conf["confidence_pct"] < 100.0
    assert conf["pillars_with_data"] >= 1


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
