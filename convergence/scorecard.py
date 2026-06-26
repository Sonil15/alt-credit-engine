"""Log-odds to points scorecard with PDO calibration.

The headline score is a deterministic PDO transform of the model's probability of
default. Explainability points are derived from the model's *own* SHAP
contributions (which live in log-odds space) using the same PDO factor, so the
per-feature breakdown reconciles to the score:

    credit_score ≈ base_points + Σ feature_points          (before clamping)

This replaces the previous hand-tuned weight table, which never reconciled with
the CatBoost-driven score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Calibrated for the 300-900 range, anchored to the population's *actual* odds.
# The base score 600 sits at ~10:1 good:bad (≈9% PD, the dataset's base default
# rate). A 50:1 anchor was previously used, but that only mapped sensibly when the
# model was over-confident (CatBoost pushing good borrowers to near-0 PD). The
# glass-box EBM champion is honestly calibrated (median PD ≈ base rate), so we
# anchor to real population odds: this spreads calibrated PDs across the band
# (strong ≈5% PD → ~646, base rate ≈9% → ~600, weak ≈17% PD → ~548).
# Decision cutoffs (APPROVE/REVIEW) live in convergence.panel.
BASE_SCORE = 600
BASE_ODDS = 10.0  # good:bad ≈ 10:1 at base score (≈9% PD, the population base rate)
PDO = 50  # points to double the odds
SCORE_MIN = 300
SCORE_MAX = 900

# Points per unit of log-odds, and the offset that anchors the band.
PDO_FACTOR = PDO / math.log(2)
SCORE_OFFSET = BASE_SCORE - PDO_FACTOR * math.log(BASE_ODDS)


@dataclass
class ScorecardResult:
    credit_score: int
    probability_of_default: float
    base_score: int
    base_points: float
    factor_points: dict[str, float] = field(default_factory=dict)


def _clamp_score(score: float) -> int:
    return int(round(max(SCORE_MIN, min(SCORE_MAX, score))))


def pd_to_log_odds(probability_of_default: float) -> float:
    pd_clamped = min(max(probability_of_default, 1e-6), 1 - 1e-6)
    return math.log(pd_clamped / (1 - pd_clamped))


def log_odds_to_pd(log_odds: float) -> float:
    return 1.0 / (1.0 + math.exp(-log_odds))


def pd_to_credit_score(probability_of_default: float) -> int:
    """Map PD to credit score using the PDO scorecard formula."""
    log_odds = pd_to_log_odds(probability_of_default)
    return _clamp_score(SCORE_OFFSET - PDO_FACTOR * log_odds)


def shap_to_points(shap_value: float) -> float:
    """Translate one SHAP log-odds contribution into credit-score points.

    Positive SHAP raises default log-odds, which *lowers* the score, hence the
    sign flip. Summed over all features (plus base_points) this reconstructs the
    score up to the 300-900 clamp.
    """
    return -PDO_FACTOR * float(shap_value)


def expected_value_to_base_points(expected_value: float) -> float:
    """Score the model would assign at its average prediction (SHAP base value)."""
    return SCORE_OFFSET - PDO_FACTOR * float(expected_value)


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
