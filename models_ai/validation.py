"""Model validation metrics and model card persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from core.seeds import CATBOOST_RANDOM_SEED
from convergence.panel import APPROVE_SCORE, REVIEW_SCORE, pd_cutoff_for_score
from models_ai.constants import FEATURE_COLUMNS, LABEL_COLUMN, fill_missing_features

logger = logging.getLogger(__name__)

MODEL_CARD_PATH = Path(__file__).parent / "artifacts" / "model_card.json"
MODEL_VERSION = "1.0.0"

APPROVE_PD_THRESHOLD = pd_cutoff_for_score(APPROVE_SCORE)
REVIEW_PD_THRESHOLD = pd_cutoff_for_score(REVIEW_SCORE)


def _gini(auc: float) -> float:
    return 2 * auc - 1


def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for binary classifier."""
    df = pd.DataFrame({"y": y_true, "prob": y_prob})
    df = df.sort_values("prob")
    df["cum_good"] = (1 - df["y"]).cumsum() / max((1 - df["y"]).sum(), 1)
    df["cum_bad"] = df["y"].cumsum() / max(df["y"].sum(), 1)
    return float((df["cum_bad"] - df["cum_good"]).abs().max())


def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    bins = np.linspace(0, 1, n_bins + 1)
    calibration = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not mask.any():
            continue
        calibration.append(
            {
                "bin_start": float(bins[i]),
                "bin_end": float(bins[i + 1]),
                "predicted_mean": float(y_prob[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return calibration


def _decision_from_pd(pd_value: float) -> str:
    if pd_value <= APPROVE_PD_THRESHOLD:
        return "APPROVE"
    if pd_value <= REVIEW_PD_THRESHOLD:
        return "REVIEW"
    return "REJECT"


def _confusion_at_cutoffs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    decisions = [_decision_from_pd(p) for p in y_prob]
    true_decisions = [_decision_from_pd(0.0 if y == 0 else 1.0) for y in y_true]
    labels = ["APPROVE", "REVIEW", "REJECT"]
    matrix = confusion_matrix(true_decisions, decisions, labels=labels)
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "accuracy": float(accuracy_score(true_decisions, decisions)),
    }


def evaluate_model(model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Compute credit-risk validation metrics on holdout data."""
    probs = model.predict_proba(X)[:, 1]
    y_arr = y.values.astype(int)

    if len(np.unique(y_arr)) < 2:
        return {
            "auc": 0.5,
            "gini": 0.0,
            "ks": 0.0,
            "accuracy": float((probs >= 0.5).astype(int).mean()),
            "default_rate": float(y_arr.mean()) if len(y_arr) else 0.0,
            "calibration": [],
            "confusion": {},
            "roc_curve": {"fpr": [], "tpr": []},
        }

    auc = float(roc_auc_score(y_arr, probs))
    fpr, tpr, _ = roc_curve(y_arr, probs)

    return {
        "auc": round(auc, 4),
        "gini": round(_gini(auc), 4),
        "ks": round(_ks_statistic(y_arr, probs), 4),
        "accuracy": round(float(((probs >= 0.5).astype(int) == y_arr).mean()), 4),
        "default_rate": round(float(y_arr.mean()), 4),
        "calibration": _calibration_bins(y_arr, probs),
        "confusion": _confusion_at_cutoffs(y_arr, probs),
        "roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr[:20]],
            "tpr": [round(float(v), 4) for v in tpr[:20]],
        },
    }


def cross_validate_metrics(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict[str, float]:
    """Stratified k-fold CV for robust metric reporting."""
    if len(y) < n_splits * 2 or y.nunique() < 2:
        return {"cv_auc_mean": 0.0, "cv_auc_std": 0.0, "cv_gini_mean": 0.0, "cv_ks_mean": 0.0}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CATBOOST_RANDOM_SEED)
    aucs, ginis, kss = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        from models_ai.catboost_model import train_catboost

        fold_model = train_catboost(
            X.iloc[train_idx].reset_index(drop=True),
            label_series=y.iloc[train_idx].reset_index(drop=True),
        )
        metrics = evaluate_model(fold_model, X.iloc[test_idx], y.iloc[test_idx])
        aucs.append(metrics["auc"])
        ginis.append(metrics["gini"])
        kss.append(metrics["ks"])

    return {
        "cv_auc_mean": round(float(np.mean(aucs)), 4),
        "cv_auc_std": round(float(np.std(aucs)), 4),
        "cv_gini_mean": round(float(np.mean(ginis)), 4),
        "cv_ks_mean": round(float(np.mean(kss)), 4),
    }


def build_model_card(
    metrics: dict[str, Any],
    *,
    users_trained: int,
    feature_columns: list[str],
    cv_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    card = {
        "model_version": MODEL_VERSION,
        "model_type": "CatBoostClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "users_trained": users_trained,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "cv_metrics": cv_metrics or {},
        "scorecard": {
            "range": [300, 900],
            "base_score": 600,
            "base_odds": 50,
            "pdo": 50,
        },
        "decision_thresholds": {
            "approve_pd_max": APPROVE_PD_THRESHOLD,
            "review_pd_max": REVIEW_PD_THRESHOLD,
        },
    }
    return card


def save_model_card(card: dict[str, Any], path: Path | None = None) -> Path:
    target = path or MODEL_CARD_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(card, indent=2), encoding="utf-8")
    logger.info("Saved model card to %s", target)
    return target


def load_model_card(path: Path | None = None) -> dict[str, Any] | None:
    target = path or MODEL_CARD_PATH
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def train_test_split_data(
    df: pd.DataFrame,
    labels: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        df,
        labels,
        test_size=test_size,
        random_state=CATBOOST_RANDOM_SEED,
        stratify=labels if labels.nunique() > 1 else None,
    )
