import pandas as pd

from convergence.score_engine import check_red_flags
from convergence.scorecard import (
    SCORE_OFFSET,
    PDO_FACTOR,
    expected_value_to_base_points,
    pd_to_credit_score,
    pd_to_log_odds,
    shap_to_points,
)


def test_scorecard_monotonicity():
    low_pd_score = pd_to_credit_score(0.1)
    high_pd_score = pd_to_credit_score(0.9)
    assert low_pd_score > high_pd_score
    assert 300 <= low_pd_score <= 900
    assert 300 <= high_pd_score <= 900


def test_shap_points_reconcile_with_score():
    """base_points + Σ feature_points must reconstruct the headline score.

    With SHAP in log-odds space: base_value + Σ shap = log-odds(PD). The PDO
    transform of that sum must equal the score (pre-clamp).
    """
    base_value = -2.0
    shap = [0.4, -0.3, 0.1, 0.05]
    total_log_odds = base_value + sum(shap)
    pd_value = 1.0 / (1.0 + pow(2.718281828, -total_log_odds))

    reconstructed = expected_value_to_base_points(base_value) + sum(shap_to_points(s) for s in shap)
    expected = SCORE_OFFSET - PDO_FACTOR * pd_to_log_odds(pd_value)
    assert abs(reconstructed - expected) < 1e-6


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
