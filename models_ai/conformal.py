"""Split conformal prediction for statistically-guaranteed abstention.

Uses the EBM champion's default probability on a held-out calibration set to
learn a nonconformity threshold. At scoring time, ambiguous cases (prediction
set contains both default and no_default) abstain to manual REVIEW.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models_ai.constants import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.10
CALIBRATION_PATH = Path(__file__).parent / "artifacts" / "conformal_calibration.json"


def _nonconformity_score(prob_default: float, true_label: int) -> float:
    """Score = 1 - p(true class). Lower is more conforming."""
    p_default = float(prob_default)
    if int(true_label) == 1:
        return 1.0 - p_default
    return p_default


def prediction_set_from_pd(prob_default: float, threshold_q: float) -> list[str]:
    """Return labels included in the conformal prediction set."""
    p_default = float(prob_default)
    p_no_default = 1.0 - p_default
    labels: list[str] = []
    if p_no_default >= 1.0 - threshold_q:
        labels.append("no_default")
    if p_default >= 1.0 - threshold_q:
        labels.append("default")
    return labels


def fit_calibration(
    model: Any,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Fit split-conformal threshold on a calibration split (exchangeable holdout)."""
    probs = model.predict_proba(X_cal[FEATURE_COLUMNS])[:, 1]
    y_arr = y_cal.values.astype(int)
    n = len(y_arr)
    if n == 0:
        raise ValueError("Calibration set is empty")

    scores = np.array([_nonconformity_score(probs[i], y_arr[i]) for i in range(n)])
    q_level = min(math.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    threshold = float(np.quantile(scores, q_level, method="higher"))

    covered = sum(
        1
        for i in range(n)
        if ("default" if y_arr[i] == 1 else "no_default")
        in prediction_set_from_pd(probs[i], threshold)
    )

    return {
        "alpha": alpha,
        "coverage_target": round(1.0 - alpha, 4),
        "threshold_q": round(threshold, 6),
        "n_calibration": n,
        "empirical_coverage": round(covered / n, 4),
        "model": "ebm_champion",
    }


def conformal_report(prob_default: float, calibration: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-friendly conformal report for one borrower."""
    threshold = float(calibration["threshold_q"])
    prediction_set = prediction_set_from_pd(prob_default, threshold)
    return {
        "prediction_set": prediction_set,
        "abstain": len(prediction_set) > 1,
        "alpha": calibration.get("alpha", DEFAULT_ALPHA),
        "coverage_target": calibration.get("coverage_target"),
        "threshold_q": round(threshold, 6),
        "nonconformity_default": round(1.0 - float(prob_default), 4),
        "nonconformity_no_default": round(float(prob_default), 4),
    }


def apply_conformal_gate(decision: str, auto_reject: bool, conformal: dict[str, Any]) -> str:
    """Route ambiguous conformal cases away from silent auto-approval."""
    if auto_reject:
        return "REJECT"
    if conformal.get("abstain") and decision == "APPROVE":
        return "REVIEW"
    return decision


def save_calibration(calibration: dict[str, Any], path: Path | None = None) -> Path:
    target = path or CALIBRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    logger.info("Saved conformal calibration to %s", target)
    return target


def load_calibration(path: Path | None = None) -> dict[str, Any] | None:
    target = path or CALIBRATION_PATH
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
