"""Train the champion + challenger panel and write a combined model card.

Champion  : EBM (glass-box, model of record, drives the score and explanation).
Challengers: CatBoost + logistic regression (audit the champion for agreement).

All three are trained on the same split so their holdout metrics are comparable.
The headline ``metrics`` in the model card are the CHAMPION's; that's the model
that actually decides, with challenger AUCs recorded alongside for transparency.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sqlalchemy.ext.asyncio import AsyncSession

from convergence.panel import decision_thresholds, pd_cutoff_for_score
from core.feature_store import fetch_features_wide
from core.seeds import CATBOOST_RANDOM_SEED
from models_ai.catboost_model import extract_labels, save_model as save_catboost, train_catboost
from models_ai.conformal import DEFAULT_ALPHA, fit_calibration, save_calibration
from models_ai.constants import FEATURE_COLUMNS, fill_missing_features
from models_ai.imputation import build_imputation_stats, save_imputation_stats
from models_ai.ebm_model import save_ebm, train_ebm
from models_ai.ebm_model import MODEL_PATH as EBM_PATH
from models_ai.logistic_model import save_logistic, train_logistic
from models_ai.validation import (
    MODEL_VERSION,
    evaluate_model,
    save_model_card,
    train_test_split_data,
)

logger = logging.getLogger(__name__)


def _champion_cv_auc(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict[str, float]:
    """Stratified k-fold AUC for the EBM champion (honest small-data metric)."""
    if len(y) < n_splits * 2 or y.nunique() < 2:
        return {"cv_auc_mean": 0.0, "cv_auc_std": 0.0}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CATBOOST_RANDOM_SEED)
    aucs: list[float] = []
    for tr, te in skf.split(X, y):
        model = train_ebm(X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True))
        proba = model.predict_proba(X.iloc[te])[:, 1]
        if y.iloc[te].nunique() > 1:
            aucs.append(roc_auc_score(y.iloc[te], proba))
    return {
        "cv_auc_mean": round(float(np.mean(aucs)), 4),
        "cv_auc_std": round(float(np.std(aucs)), 4),
    }


def _holdout_auc(model, X: pd.DataFrame, y: pd.Series) -> float:
    if y.nunique() < 2:
        return 0.5
    return round(float(roc_auc_score(y, model.predict_proba(X)[:, 1])), 4)


async def train_all_from_db(session: AsyncSession) -> dict[str, Any]:
    """Train champion + challengers, persist artifacts, write the combined card."""
    wide = await fetch_features_wide(session)
    if wide.empty:
        raise ValueError("No ml_features available for training")

    labels = extract_labels(wide)
    # Learn the cohort-aware typical-applicant profile from the raw (pre-fill) training
    # population, so serve-time imputation of absent sources reflects this cohort mix.
    save_imputation_stats(build_imputation_stats(wide))
    features = fill_missing_features(wide.copy())
    X_train, X_test, y_train, y_test = train_test_split_data(features, labels)
    y_train = y_train.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)

    # Hold out a calibration slice from train for split conformal (not used in fitting).
    stratify = y_train if y_train.nunique() > 1 else None
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=CATBOOST_RANDOM_SEED,
        stratify=stratify,
    )
    X_fit = X_fit.reset_index(drop=True)
    X_cal = X_cal.reset_index(drop=True)
    y_fit = y_fit.reset_index(drop=True)
    y_cal = y_cal.reset_index(drop=True)

    # --- champion: EBM ---
    ebm = train_ebm(X_fit, y_fit)
    conformal_calibration = fit_calibration(ebm, X_cal, y_cal, alpha=DEFAULT_ALPHA)
    save_calibration(conformal_calibration)
    champion_metrics = evaluate_model(ebm, X_test, y_test)  # only uses predict_proba
    cv_metrics = _champion_cv_auc(features, labels)
    save_ebm(ebm)

    # --- challengers: CatBoost + logistic ---
    catboost = train_catboost(X_fit, label_series=y_fit)
    logistic = train_logistic(X_fit, y_fit)
    save_catboost(catboost)
    save_logistic(logistic)

    challenger_metrics = {
        "catboost": {"auc": _holdout_auc(catboost, X_test, y_test)},
        "logistic": {"auc": _holdout_auc(logistic, X_test, y_test)},
    }

    card = {
        "model_version": MODEL_VERSION,
        "model_type": "ExplainableBoostingClassifier (champion) + CatBoost/Logistic (challengers)",
        "champion": "ebm",
        "challengers": ["catboost", "logistic"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "users_trained": int(len(wide)),
        "feature_columns": FEATURE_COLUMNS,
        "metrics": champion_metrics,
        "cv_metrics": cv_metrics,
        "challenger_metrics": challenger_metrics,
        "scorecard": {"range": [300, 900], "base_score": 600, "base_odds": 50, "pdo": 50},
        "decision_thresholds": {
            **decision_thresholds(),
            "approve_pd_max": round(pd_cutoff_for_score(decision_thresholds()["approve_score"]), 4),
            "review_pd_max": round(pd_cutoff_for_score(decision_thresholds()["review_score"]), 4),
        },
        "conformal": conformal_calibration,
    }
    save_model_card(card)

    logger.info(
        "Panel trained: champion EBM AUC=%.3f | challengers %s",
        champion_metrics.get("auc", 0.0),
        challenger_metrics,
    )
    return {
        "users_trained": int(len(wide)),
        "default_rate": float(labels.mean()),
        "model_path": str(EBM_PATH),
        "feature_count": len(FEATURE_COLUMNS),
        "model_version": MODEL_VERSION,
        "metrics": champion_metrics,
        "cv_metrics": cv_metrics,
        "challenger_metrics": challenger_metrics,
    }
