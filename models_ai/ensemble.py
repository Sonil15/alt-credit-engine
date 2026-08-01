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
from models_ai.ood import fit_ood, save_calibration as save_ood_calibration
from models_ai.constants import FEATURE_COLUMNS, fill_missing_features
from models_ai.imputation import build_imputation_stats, save_imputation_stats
from models_ai.ebm_model import save_ebm, train_ebm
from models_ai.ebm_model import MODEL_PATH as EBM_PATH
from models_ai.logistic_model import save_logistic, train_logistic
from models_ai.tempering import fit_temperature, temper_catboost, temper_logistic
from models_ai.validation import (
    MODEL_VERSION,
    evaluate_model,
    save_model_card,
    train_test_split_data,
)

logger = logging.getLogger(__name__)


def _champion_cv_diagnostics(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Stratified k-fold AUC for the EBM champion (honest small-data metric).

    Also returns the out-of-fold margins (fold-model logit minus that fold's
    intercept) and labels: the honest inputs for temperature scaling. The held-out
    calibration slice can't play that role here - at n≈20 the champion usually
    separates it perfectly, and log-loss then says "don't damp anything" (T=1)
    even when cross-validation shows the confidence is not real.
    """
    if len(y) < n_splits * 2 or y.nunique() < 2:
        return {"cv_auc_mean": 0.0, "cv_auc_std": 0.0}, np.array([]), np.array([])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CATBOOST_RANDOM_SEED)
    aucs: list[float] = []
    oof_margins: list[float] = []
    oof_labels: list[int] = []
    for tr, te in skf.split(X, y):
        model = train_ebm(X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True))
        proba = model.predict_proba(X.iloc[te])[:, 1]
        if y.iloc[te].nunique() > 1:
            aucs.append(roc_auc_score(y.iloc[te], proba))
        clipped = np.clip(proba, 1e-9, 1.0 - 1e-9)
        fold_intercept = float(np.ravel(model.intercept_)[0])
        oof_margins.extend(np.log(clipped / (1.0 - clipped)) - fold_intercept)
        oof_labels.extend(y.iloc[te].astype(int).tolist())
    cv_metrics = {
        "cv_auc_mean": round(float(np.mean(aucs)), 4),
        "cv_auc_std": round(float(np.std(aucs)), 4),
    }
    return cv_metrics, np.asarray(oof_margins), np.asarray(oof_labels)


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
    # 25% (not 20%): at alpha=0.1 the conformal quantile needs n_cal >= 20 before
    # ceil((n+1)(1-alpha))/n drops below the maximum - with a smaller slice a single
    # noisy calibration point (a lucky defaulter scored as safe) forces the threshold
    # to ~1.0 and the gate abstains on the whole portfolio.
    stratify = y_train if y_train.nunique() > 1 else None
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=CATBOOST_RANDOM_SEED,
        stratify=stratify,
    )
    X_fit = X_fit.reset_index(drop=True)
    X_cal = X_cal.reset_index(drop=True)
    y_fit = y_fit.reset_index(drop=True)
    y_cal = y_cal.reset_index(drop=True)

    # --- champion: EBM ---
    ebm = train_ebm(X_train, y_train)
    # Honest confidence: damp the champion's saturated small-sample logits with a
    # temperature fitted on OUT-OF-FOLD predictions (see _champion_cv_diagnostics),
    # BEFORE fitting conformal, so the abstention threshold is learned on the same
    # PD scale that serves scores.
    cv_metrics, oof_margins, oof_labels = _champion_cv_diagnostics(X_train, y_train)
    ebm_intercept = float(np.ravel(ebm.intercept_)[0])
    t_ebm = (
        fit_temperature(oof_margins, oof_labels, ebm_intercept) if len(oof_margins) else 1.0
    )
    ebm.term_scores_ = [np.asarray(scores) / t_ebm for scores in ebm.term_scores_]
    temperatures = {"ebm": round(t_ebm, 2)}
    conformal_calibration = fit_calibration(ebm, X_cal, y_cal, alpha=DEFAULT_ALPHA)
    save_calibration(conformal_calibration)
    # OOD integrity gate: learn the joint training manifold the champion actually saw
    # (X_train), so anomalous feature *combinations* at serve time abstain to REVIEW.
    # It never touches PD - purely an eligibility filter sitting outside the glass box.
    ood_calibration = fit_ood(X_train)
    save_ood_calibration(ood_calibration)
    champion_metrics = evaluate_model(ebm, X_test, y_test)  # only uses predict_proba
    save_ebm(ebm)

    # --- challengers: CatBoost + logistic (tempered on the same slice so the
    # panel's decision bands stay on a comparable PD scale) ---
    catboost = train_catboost(X_fit, label_series=y_fit)
    logistic = train_logistic(X_fit, y_fit)
    temperatures["catboost"] = round(temper_catboost(catboost, X_cal, y_cal), 2)
    temperatures["logistic"] = round(temper_logistic(logistic, X_cal, y_cal), 2)
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
        "ood": {k: v for k, v in ood_calibration.items() if k not in ("mean", "precision")},
        "temperatures": temperatures,
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
