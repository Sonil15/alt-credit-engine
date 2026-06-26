"""Split conformal abstention behavior."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models_ai.conformal import (
    DEFAULT_ALPHA,
    apply_conformal_gate,
    conformal_report,
    fit_calibration,
    prediction_set_from_pd,
)
from models_ai.constants import FEATURE_COLUMNS


class _ProbModel:
    """Minimal sklearn-like wrapper for unit tests."""

    def __init__(self, model: LogisticRegression):
        self._model = model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X[FEATURE_COLUMNS])


def _synthetic_model():
    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    logits = X[FEATURE_COLUMNS[:3]].sum(axis=1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(int)
    model = LogisticRegression(max_iter=500)
    model.fit(X[FEATURE_COLUMNS], y)
    return _ProbModel(model), X, pd.Series(y)


def test_prediction_set_unambiguous_low_pd():
    cal = {"threshold_q": 0.5, "alpha": DEFAULT_ALPHA, "coverage_target": 0.9}
    report = conformal_report(0.01, cal)
    assert report["prediction_set"] == ["no_default"]
    assert report["abstain"] is False


def test_prediction_set_ambiguous_mid_pd():
    cal = {"threshold_q": 0.5, "alpha": DEFAULT_ALPHA, "coverage_target": 0.9}
    report = conformal_report(0.5, cal)
    assert set(report["prediction_set"]) == {"no_default", "default"}
    assert report["abstain"] is True


def test_fit_calibration_empirical_coverage():
    model, X, y = _synthetic_model()
    cal_idx = np.arange(0, len(y), 2)
    calibration = fit_calibration(model, X.iloc[cal_idx], y.iloc[cal_idx], alpha=0.10)
    assert calibration["n_calibration"] == len(cal_idx)
    assert calibration["empirical_coverage"] >= 0.80
    assert 0.0 <= calibration["threshold_q"] <= 1.0


def test_apply_conformal_gate_routes_contested_approve():
    conformal = {"abstain": True}
    assert apply_conformal_gate("APPROVE", False, conformal) == "REVIEW"


def test_apply_conformal_gate_keeps_reject():
    conformal = {"abstain": True}
    assert apply_conformal_gate("REJECT", False, conformal) == "REJECT"


def test_apply_conformal_gate_red_flag_always_rejects():
    conformal = {"abstain": True}
    assert apply_conformal_gate("APPROVE", True, conformal) == "REJECT"


def test_prediction_set_from_pd_threshold_boundary():
    assert prediction_set_from_pd(0.05, 0.5) == ["no_default"]
    assert prediction_set_from_pd(0.95, 0.5) == ["default"]
    assert set(prediction_set_from_pd(0.5, 0.5)) == {"no_default", "default"}
