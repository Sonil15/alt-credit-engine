"""Fuse econometric + AI outputs into interpretable credit scores."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from sqlalchemy.ext.asyncio import AsyncSession

from core.feature_store import fetch_features_wide
from models_ai.catboost_model import (
    FEATURE_COLUMNS,
    get_feature_matrix_for_user,
    load_model,
    predict_pd,
)

logger = logging.getLogger(__name__)

SPATIAL_VARIANCE_THRESHOLD = 50.0
SCORE_MIN = 300
SCORE_MAX = 900


def pd_to_credit_score(probability_of_default: float) -> int:
    """Map PD (0-1) to alternate credit score (300-900)."""
    score = SCORE_MAX - (probability_of_default * (SCORE_MAX - SCORE_MIN))
    return int(round(max(SCORE_MIN, min(SCORE_MAX, score))))


def check_red_flags(row: pd.Series) -> tuple[bool, str | None]:
    spatial = float(row.get("spatial_variance_score", 0.0) or 0.0)
    income = float(row.get("monthly_income_mean", 0.0) or 0.0)

    if spatial > SPATIAL_VARIANCE_THRESHOLD and income <= 0:
        return True, "Auto-reject: high geographic instability with zero baseline income"

    missed = float(row.get("missed_payments_count", 0.0) or 0.0)
    if missed >= 5:
        return True, "Auto-reject: excessive missed telecom payments"

    return False, None


def _decision_from_score(credit_score: int, auto_reject: bool) -> str:
    if auto_reject:
        return "REJECT"
    if credit_score >= 750:
        return "APPROVE"
    if credit_score >= 550:
        return "REVIEW"
    return "REJECT"


def extract_top_shap_drivers(
    model,
    feature_row: pd.DataFrame,
    top_n: int = 3,
) -> list[dict[str, float]]:
    """Return top N SHAP feature drivers for a single user."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_row[FEATURE_COLUMNS])

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    contributions = [
        {"feature": name, "shap_value": float(val)}
        for name, val in zip(FEATURE_COLUMNS, values, strict=True)
    ]
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return contributions[:top_n]


async def score_user(session: AsyncSession, user_id: str) -> dict[str, Any]:
    """Compute full credit score payload for a single user."""
    wide = await fetch_features_wide(session)
    if wide.empty:
        raise ValueError("No ml_features available. Ingest data and run training first.")

    wide["user_id"] = wide["user_id"].astype(str)
    user_row = wide[wide["user_id"] == str(user_id)]
    if user_row.empty:
        raise ValueError(f"User {user_id} not found in ml_features")

    auto_reject, reject_reason = check_red_flags(user_row.iloc[0])

    model = load_model()
    pd_df = predict_pd(model, user_row)
    probability_of_default = float(pd_df.iloc[0]["probability_of_default"])

    if auto_reject:
        probability_of_default = 1.0

    credit_score = pd_to_credit_score(probability_of_default)
    decision = _decision_from_score(credit_score, auto_reject)

    feature_row = get_feature_matrix_for_user(wide, user_id)
    shap_drivers = extract_top_shap_drivers(model, feature_row)

    return {
        "user_id": str(user_id),
        "credit_score": credit_score,
        "probability_of_default": round(probability_of_default, 4),
        "decision": decision,
        "auto_reject": auto_reject,
        "reject_reason": reject_reason,
        "shap_drivers": shap_drivers,
    }


async def score_all_users(session: AsyncSession) -> list[dict[str, Any]]:
    """Score every user with features in the database."""
    wide = await fetch_features_wide(session)
    if wide.empty:
        return []

    results = []
    for user_id in wide["user_id"].astype(str).unique():
        try:
            results.append(await score_user(session, user_id))
        except Exception:
            logger.exception("Failed to score user %s", user_id)
    return results
