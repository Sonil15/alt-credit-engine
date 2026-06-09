"""Fuse econometric + AI outputs into interpretable credit scores."""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from convergence.fairness import compute_fairness_report
from convergence.reason_codes import format_reason_codes, shap_to_reason_codes
from convergence.scorecard import score_from_pd_and_features
from core.feature_store import fetch_features_wide, fetch_user_features_wide
from core.json_utils import safe_float, safe_round, sanitize_for_json
from core.model_cache import get_cached_explainer, get_cached_model, get_model_version
from models.db_models import ScoreDecision
from models_ai.catboost_model import get_feature_matrix_for_user, predict_pd
from models_ai.constants import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

SPATIAL_VARIANCE_THRESHOLD = 50.0


def check_red_flags(row: pd.Series) -> tuple[bool, str | None]:
    spatial = safe_float(row.get("spatial_variance_score", 0.0))
    income = safe_float(row.get("monthly_income_mean", 0.0))

    if spatial > SPATIAL_VARIANCE_THRESHOLD and income <= 0:
        return True, "Auto-reject: high geographic instability with zero baseline income"

    missed = safe_float(row.get("missed_payments_count", 0.0))
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


def extract_top_shap_drivers(model, explainer, feature_row: pd.DataFrame, top_n: int = 3) -> list[dict[str, float]]:
    """Return top N SHAP feature drivers for a single user."""
    shap_values = explainer.shap_values(feature_row[FEATURE_COLUMNS])

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    contributions = [
        {"feature": name, "shap_value": safe_float(val)}
        for name, val in zip(FEATURE_COLUMNS, values, strict=True)
    ]
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return contributions[:top_n]


def _finalize_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure all floats in a score payload are strict-JSON safe."""
    return sanitize_for_json(payload)


async def _persist_decision(session: AsyncSession, payload: dict[str, Any]) -> None:
    session.add(
        ScoreDecision(
            user_id=UUID(str(payload["user_id"])),
            credit_score=int(payload["credit_score"]),
            probability_of_default=float(payload["probability_of_default"]),
            decision=str(payload["decision"]),
            model_version=str(payload.get("model_version", "unknown")),
            auto_reject=1 if payload.get("auto_reject") else 0,
            reject_reason=payload.get("reject_reason"),
            reason_codes_json=json.dumps(payload.get("reason_codes", [])),
        )
    )
    await session.commit()


async def score_user(session: AsyncSession, user_id: str, *, persist: bool = True) -> dict[str, Any]:
    """Compute full credit score payload for a single user."""
    user_row = await fetch_user_features_wide(session, user_id)
    if user_row.empty:
        raise ValueError(f"User {user_id} not found in ml_features")

    auto_reject, reject_reason = check_red_flags(user_row.iloc[0])

    model = get_cached_model()
    explainer = get_cached_explainer()
    pd_df = predict_pd(model, user_row)
    probability_of_default = safe_float(pd_df.iloc[0]["probability_of_default"])

    if auto_reject:
        probability_of_default = 1.0

    feature_row = get_feature_matrix_for_user(user_row, user_id)
    scorecard = score_from_pd_and_features(probability_of_default, user_row.iloc[0])
    credit_score = scorecard.credit_score
    decision = _decision_from_score(credit_score, auto_reject)

    shap_drivers = extract_top_shap_drivers(model, explainer, feature_row)
    reason_codes = shap_to_reason_codes(shap_drivers)
    if auto_reject and reject_reason:
        reason_codes.insert(0, reject_reason)

    result = _finalize_score_payload(
        {
            "user_id": str(user_id),
            "credit_score": credit_score,
            "probability_of_default": safe_round(probability_of_default, 4),
            "decision": decision,
            "auto_reject": auto_reject,
            "reject_reason": reject_reason,
            "shap_drivers": shap_drivers,
            "reason_codes": reason_codes,
            "reason_codes_text": format_reason_codes(reason_codes),
            "factor_points": scorecard.factor_points,
            "model_version": get_model_version(),
        }
    )

    if persist:
        await _persist_decision(session, result)

    return result


async def score_all_users(session: AsyncSession) -> list[dict[str, Any]]:
    """Score every user with features in the database."""
    wide = await fetch_features_wide(session)
    if wide.empty:
        return []

    model = get_cached_model()
    explainer = get_cached_explainer()
    pd_df = predict_pd(model, wide)

    results = []
    for _, row in wide.iterrows():
        user_id = str(row["user_id"])
        try:
            user_row = wide[wide["user_id"].astype(str) == user_id]
            auto_reject, reject_reason = check_red_flags(user_row.iloc[0])
            pd_row = pd_df[pd_df["user_id"] == user_id]
            probability_of_default = safe_float(pd_row.iloc[0]["probability_of_default"])
            if auto_reject:
                probability_of_default = 1.0

            feature_row = get_feature_matrix_for_user(wide, user_id)
            scorecard = score_from_pd_and_features(probability_of_default, user_row.iloc[0])
            credit_score = scorecard.credit_score
            decision = _decision_from_score(credit_score, auto_reject)
            shap_drivers = extract_top_shap_drivers(model, explainer, feature_row)
            reason_codes = shap_to_reason_codes(shap_drivers)
            if auto_reject and reject_reason:
                reason_codes.insert(0, reject_reason)

            result = _finalize_score_payload(
                {
                    "user_id": user_id,
                    "credit_score": credit_score,
                    "probability_of_default": safe_round(probability_of_default, 4),
                    "decision": decision,
                    "auto_reject": auto_reject,
                    "reject_reason": reject_reason,
                    "shap_drivers": shap_drivers,
                    "reason_codes": reason_codes,
                    "reason_codes_text": format_reason_codes(reason_codes),
                    "factor_points": scorecard.factor_points,
                    "model_version": get_model_version(),
                }
            )
            await _persist_decision(session, result)
            results.append(result)
        except Exception:
            logger.exception("Failed to score user %s", user_id)
    return results


async def portfolio_summary(session: AsyncSession) -> dict[str, Any]:
    """Aggregate portfolio metrics for bank dashboard."""
    scores = await score_all_users(session)
    wide = await fetch_features_wide(session)
    if not scores:
        return {
            "total_users": 0,
            "approval_rate": 0.0,
            "review_rate": 0.0,
            "reject_rate": 0.0,
            "expected_default_rate": 0.0,
            "avg_score": 0.0,
            "score_distribution": {"300-549": 0, "550-749": 0, "750-900": 0},
            "fairness": compute_fairness_report([], wide),
        }

    total = len(scores)
    approvals = sum(1 for s in scores if s["decision"] == "APPROVE")
    reviews = sum(1 for s in scores if s["decision"] == "REVIEW")
    rejects = sum(1 for s in scores if s["decision"] == "REJECT")
    avg_pd = sum(s["probability_of_default"] for s in scores) / total
    avg_score = sum(s["credit_score"] for s in scores) / total

    distribution = {"300-549": 0, "550-749": 0, "750-900": 0}
    for s in scores:
        if s["credit_score"] < 550:
            distribution["300-549"] += 1
        elif s["credit_score"] < 750:
            distribution["550-749"] += 1
        else:
            distribution["750-900"] += 1

    return {
        "total_users": total,
        "approval_rate": round(approvals / total, 4),
        "review_rate": round(reviews / total, 4),
        "reject_rate": round(rejects / total, 4),
        "expected_default_rate": round(avg_pd, 4),
        "avg_score": round(avg_score, 2),
        "score_distribution": distribution,
        "fairness": compute_fairness_report(scores, wide),
    }
