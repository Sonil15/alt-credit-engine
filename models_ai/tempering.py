"""Post-hoc temperature scaling for the champion/challenger panel.

With ~60 training rows and 23 features every panel model saturates its logits:
good borrowers get PD ≈ 0, bad ones PD ≈ 1, and the PDO scorecard clamps most of
the portfolio to the 900 ceiling. Temperature scaling divides each model's
feature-driven margin (its logit minus the prior-corrected intercept) by a single
T ≥ 1 fitted on the held-out calibration slice by log-loss. Ranking (AUC) is
unchanged — only confidence is damped — and because the transform is a uniform
scale on the additive terms, the champion's per-feature contributions still
reconcile exactly to the score.

Together with :func:`models_ai.constants.prior_correction_log_odds` this gives
PDs an honest center *and* an honest spread.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# T=1 (no damping) is always in the grid, so tempering can only help calibration.
# The ceiling is deliberately moderate (6.0): an unbounded search on saturated
# small-sample margins drives T toward "flatten every PD to the base rate", which
# collapses the whole portfolio onto one score. Capping T keeps damped-but-real
# confidence, so calibrated PDs still spread across the 300-900 band.
TEMPERATURE_GRID = np.geomspace(1.0, 6.0, 60)


def fit_temperature(margins: np.ndarray, labels: pd.Series | np.ndarray, center: float) -> float:
    """Pick T minimizing the Brier score of sigmoid(center + margins / T).

    Brier (a proper scoring rule, like log-loss) is used deliberately: the raw
    margins are saturated, so the handful of confidently-wrong points carry
    unbounded log-loss and would push T to "flatten everything to the base rate".
    Brier bounds each error's cost at 1, so T settles where the bulk of the
    portfolio is well calibrated instead of where the worst mistake is hidden.
    """
    y = np.asarray(labels, dtype=float)
    m = np.asarray(margins, dtype=float)
    best_t, best_loss = 1.0, np.inf
    for t in TEMPERATURE_GRID:
        z = center + m / t
        p = 1.0 / (1.0 + np.exp(-z))
        loss = float(np.mean((p - y) ** 2))
        if loss < best_loss - 1e-12:
            best_loss, best_t = loss, float(t)
    return best_t


def temper_catboost(model, X_cal: pd.DataFrame, y_cal: pd.Series) -> float:
    """Scale CatBoost's raw-score margin by 1/T via its scale/bias hooks."""
    scale, bias = model.get_scale_and_bias()
    raw = np.asarray(model.predict(X_cal, prediction_type="RawFormulaVal"), dtype=float)
    margins = raw - bias
    t = fit_temperature(margins, y_cal, bias)
    model.set_scale_and_bias(scale / t, bias)
    logger.info("CatBoost temperature-scaled with T=%.2f", t)
    return t


def temper_logistic(pipeline, X_cal: pd.DataFrame, y_cal: pd.Series) -> float:
    """Scale the logistic challenger's coefficients by 1/T (intercept unchanged)."""
    clf = pipeline.named_steps["clf"]
    center = float(np.ravel(clf.intercept_)[0])
    margins = np.asarray(pipeline.decision_function(X_cal), dtype=float) - center
    t = fit_temperature(margins, y_cal, center)
    clf.coef_ = clf.coef_ / t
    logger.info("Logistic temperature-scaled with T=%.2f", t)
    return t
