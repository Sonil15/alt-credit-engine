import pandas as pd

from convergence.score_engine import check_red_flags
from convergence.scorecard import pd_to_credit_score, score_from_pd_and_features


def test_scorecard_monotonicity():
    low_pd_score = pd_to_credit_score(0.1)
    high_pd_score = pd_to_credit_score(0.9)
    assert low_pd_score > high_pd_score
    assert 300 <= low_pd_score <= 900
    assert 300 <= high_pd_score <= 900


def test_scorecard_factor_points():
    row = pd.Series({"missed_payments_count": 3.0, "resilience_coefficient": 0.8})
    result = score_from_pd_and_features(0.3, row)
    assert result.credit_score >= 300
    assert isinstance(result.factor_points, dict)


def test_red_flag_geo_income():
    row = pd.Series({"spatial_variance_score": 60.0, "monthly_income_mean": 0.0, "missed_payments_count": 0})
    flagged, reason = check_red_flags(row)
    assert flagged is True
    assert reason is not None


def test_red_flag_missed_payments():
    row = pd.Series({"spatial_variance_score": 10.0, "monthly_income_mean": 5000.0, "missed_payments_count": 5.0})
    flagged, _ = check_red_flags(row)
    assert flagged is True


def test_no_red_flag():
    row = pd.Series({"spatial_variance_score": 10.0, "monthly_income_mean": 5000.0, "missed_payments_count": 1.0})
    flagged, reason = check_red_flags(row)
    assert flagged is False
    assert reason is None
