"""Log-odds to points scorecard with PDO calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.json_utils import safe_float

# Calibrated for 300-900 range: base score 600 at 50:1 odds, PDO=50
BASE_SCORE = 600
BASE_ODDS = 50.0  # good:bad = 50:1 at base score
PDO = 50  # points to double the odds
SCORE_MIN = 300
SCORE_MAX = 900

# Factor -> points per unit (WOE-style linear scaling for demo scorecard)
FACTOR_WEIGHTS: dict[str, float] = {
    "avg_days_late": -8.0,
    "missed_payments_count": -25.0,
    "necessity_ratio": 40.0,
    "avg_merchant_rating": 15.0,
    "monthly_spend_volatility": -0.002,
    "spatial_variance_score": -0.5,
    "anchor_count": 5.0,
    "monthly_income_mean": 0.0008,
    "monthly_expense_mean": -0.0004,
    "cashflow_volatility": -0.001,
    "conscientiousness": 45.0,
    "locus_of_control": 40.0,
    "financial_self_efficacy": 35.0,
    "present_bias": -50.0,
    "debt_attitude": 40.0,
    "response_validity": 30.0,
    "resilience_coefficient": 60.0,
    "is_stationary": 20.0,
}


@dataclass
class ScorecardResult:
    credit_score: int
    probability_of_default: float
    base_score: int
    factor_points: dict[str, float]


def _clamp_score(score: float) -> int:
    return int(round(max(SCORE_MIN, min(SCORE_MAX, score))))


def pd_to_log_odds(probability_of_default: float) -> float:
    pd_clamped = min(max(probability_of_default, 1e-6), 1 - 1e-6)
    return math.log(pd_clamped / (1 - pd_clamped))


def log_odds_to_pd(log_odds: float) -> float:
    return 1.0 / (1.0 + math.exp(-log_odds))


def pd_to_credit_score(probability_of_default: float) -> int:
    """Map PD to credit score using PDO scorecard formula."""
    factor = PDO / math.log(2)
    offset = BASE_SCORE - factor * math.log(BASE_ODDS)
    log_odds = pd_to_log_odds(probability_of_default)
    score = offset - factor * log_odds
    return _clamp_score(score)


def compute_factor_points(row: pd.Series, feature_names: list[str] | None = None) -> dict[str, float]:
    """Compute per-factor point contributions for explainability."""
    names = feature_names or list(FACTOR_WEIGHTS.keys())
    points: dict[str, float] = {}
    for name in names:
        weight = FACTOR_WEIGHTS.get(name, 0.0)
        value = safe_float(row.get(name, 0.0))
        points[name] = round(weight * value, 2)
    return points


def score_from_pd_and_features(
    probability_of_default: float,
    feature_row: pd.Series,
) -> ScorecardResult:
    """Build full scorecard result with factor-level point breakdown."""
    credit_score = pd_to_credit_score(probability_of_default)
    factor_points = compute_factor_points(feature_row)
    return ScorecardResult(
        credit_score=credit_score,
        probability_of_default=probability_of_default,
        base_score=BASE_SCORE,
        factor_points=factor_points,
    )


def population_stats(scores: list[int]) -> dict[str, float]:
    if not scores:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(scores)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
