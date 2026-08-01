"""Out-of-distribution abstention gate (multivariate anomaly detection).

The champion is an *additive* glass box: ``logit(PD) = β₀ + Σ fᵢ(xᵢ) + Σ f_ij(xᵢ,xⱼ)``.
Additive terms extrapolate independently, so a *joint* feature vector that never
occurred in training - each coordinate individually plausible, the combination
impossible - is scored by summing curves in a region the model never saw. That is
exactly the shape of a gamed applicant: push one or two features to flattering
extremes while the rest stay mediocre.

This gate measures the applicant's Mahalanobis distance from the training manifold
(shrunk covariance, so the ~40-feature / partly-collinear matrix stays invertible)
and, above a calibrated distance threshold, routes the case to manual ``REVIEW``.
It is an *integrity filter that sits outside the score*: it never edits a feature,
never touches PD, and never turns a rejection into an approval - it only declines to
auto-approve a statistically anomalous joint profile. The EBM stays a clean glass box.

Persistence is plain JSON (mean vector + precision matrix + threshold): the gate is
itself auditable - a risk officer can read the matrix - and needs no pickled estimator
at serve time.

Caveat (documented, not hidden): the training population is a *mixture* of cohorts,
many features being structurally 0 for cohorts they don't apply to. A single global
Mahalanobis distance treats that mixture as one cloud; a legitimate cohort whose
combination is rare in the overall mix can sit far from the centroid. That is why the
gate abstains to human review rather than rejecting, and why per-cohort OOD is listed
as future work. The false-positive budget is explicit: at ``quantile=0.99`` roughly
1% of *training* applicants would themselves be routed to review.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from models_ai.constants import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Distance quantile above which a joint feature vector is treated as out-of-distribution.
# Named + configurable on purpose: this is the abstention budget, not a magic number.
# 0.99 => ~1% of the training population itself would route to manual review.
DEFAULT_OOD_QUANTILE = 0.99
# Below this many training rows a shrunk covariance is too unstable to trust; disable.
MIN_TRAIN_ROWS = 50
CALIBRATION_PATH = Path(__file__).parent / "artifacts" / "ood_calibration.json"


def _feature_matrix(X: pd.DataFrame) -> np.ndarray:
    """Numeric FEATURE_COLUMNS matrix with non-finite entries zeroed (serve-time guard)."""
    arr = X[FEATURE_COLUMNS].to_numpy(dtype=float, copy=True)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _mahalanobis_sq(mat: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of each row of ``mat`` from ``mean`` under ``precision``."""
    centered = mat - mean
    # row-wise (x-μ)ᵀ P (x-μ) without forming the full Gram matrix
    return np.einsum("ij,jk,ik->i", centered, precision, centered)


def fit_ood(
    X_train: pd.DataFrame,
    *,
    quantile: float = DEFAULT_OOD_QUANTILE,
) -> dict[str, Any]:
    """Learn the training manifold (location + shrunk precision) and a distance threshold.

    Fitted on the same rows the champion trained on, so "distance from the training
    distribution" means distance from what the model actually saw.
    """
    mat = _feature_matrix(X_train)
    n = mat.shape[0]
    if n < MIN_TRAIN_ROWS:
        logger.warning("OOD gate disabled: only %d training rows (< %d)", n, MIN_TRAIN_ROWS)
        return {"enabled": False, "reason": "insufficient_training_rows", "n_train": int(n)}

    lw = LedoitWolf().fit(mat)
    mean = np.asarray(lw.location_, dtype=float)
    precision = np.asarray(lw.precision_, dtype=float)  # shrunk, always invertible

    train_dsq = _mahalanobis_sq(mat, mean, precision)
    threshold = float(np.quantile(train_dsq, quantile))

    return {
        "enabled": True,
        "quantile": quantile,
        "threshold_dsq": round(threshold, 6),
        "n_train": int(n),
        "n_features": len(FEATURE_COLUMNS),
        "shrinkage": round(float(lw.shrinkage_), 6),
        "feature_columns": list(FEATURE_COLUMNS),
        "mean": [round(float(v), 8) for v in mean],
        "precision": [[round(float(v), 8) for v in row] for row in precision],
        "model": "ebm_champion",
    }


def ood_report(feature_row: pd.DataFrame, calibration: dict[str, Any] | None) -> dict[str, Any]:
    """Build a JSON-friendly OOD report for one applicant's joint feature vector."""
    if not calibration or not calibration.get("enabled"):
        return {"ood": False, "enabled": False}

    mean = np.asarray(calibration["mean"], dtype=float)
    precision = np.asarray(calibration["precision"], dtype=float)
    threshold = float(calibration["threshold_dsq"])

    mat = _feature_matrix(feature_row)
    dsq = float(_mahalanobis_sq(mat[:1], mean, precision)[0])

    return {
        "enabled": True,
        "ood": dsq > threshold,
        "distance_sq": round(dsq, 4),
        "threshold_dsq": round(threshold, 4),
        # how far past the boundary, as a multiple of the threshold (1.0 == on the boundary)
        "severity": round(dsq / threshold, 3) if threshold > 0 else None,
        "quantile": calibration.get("quantile", DEFAULT_OOD_QUANTILE),
    }


def apply_ood_gate(decision: str, auto_reject: bool, ood: dict[str, Any]) -> str:
    """Route an anomalous joint profile away from silent auto-approval.

    Mirrors ``apply_conformal_gate``: only ever APPROVE -> REVIEW. Never overrides an
    auto-reject, never manufactures an approval.
    """
    if auto_reject:
        return "REJECT"
    if ood.get("ood") and decision == "APPROVE":
        return "REVIEW"
    return decision


def save_calibration(calibration: dict[str, Any], path: Path | None = None) -> Path:
    target = path or CALIBRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    logger.info("Saved OOD calibration to %s", target)
    return target


def load_calibration(path: Path | None = None) -> dict[str, Any] | None:
    target = path or CALIBRATION_PATH
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
