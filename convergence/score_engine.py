"""Fuse econometric + AI outputs into interpretable credit scores."""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from convergence.fairness import compute_fairness_report
from convergence.feature_meta import build_feature_trace
from convergence.lending import recommend_terms
from convergence.pillars import compute_confidence, compute_norm_stats, compute_pillar_scores
from convergence.reason_codes import format_reason_codes, shap_to_reason_codes
from convergence.scorecard import (
    expected_value_to_base_points,
    pd_to_credit_score,
    shap_to_points,
)
from core.feature_store import fetch_features_wide
from core.json_utils import safe_float, safe_round, sanitize_for_json
from core.model_cache import get_cached_explainer, get_cached_model, get_model_version
from models.db_models import ScoreDecision
from models_ai.catboost_model import get_feature_matrix_for_user, predict_pd
from models_ai.constants import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

SPATIAL_VARIANCE_THRESHOLD = 50.0
MISSED_PAYMENTS_THRESHOLD = 5
APPROVE_SCORE = 750
REVIEW_SCORE = 550
# Below this confidence we never silently auto-approve a thin file.
LOW_CONFIDENCE_PCT = 60.0


def check_red_flags(row: pd.Series) -> tuple[bool, str | None]:
    spatial = safe_float(row.get("spatial_variance_score", 0.0))
    income = safe_float(row.get("monthly_income_mean", 0.0))

    if spatial > SPATIAL_VARIANCE_THRESHOLD and income <= 0:
        return True, "Auto-reject: high geographic instability with zero baseline income"

    missed = safe_float(row.get("missed_payments_count", 0.0))
    if missed >= MISSED_PAYMENTS_THRESHOLD:
        return True, "Auto-reject: excessive missed telecom payments"

    return False, None


def _decision_from_score(credit_score: int, auto_reject: bool, confidence_pct: float) -> str:
    if auto_reject:
        return "REJECT"
    if credit_score >= APPROVE_SCORE:
        # A thin file should be human-reviewed rather than auto-approved.
        return "APPROVE" if confidence_pct >= LOW_CONFIDENCE_PCT else "REVIEW"
    if credit_score >= REVIEW_SCORE:
        return "REVIEW"
    return "REJECT"


def _shap_contributions_for_row(explainer, feature_row: pd.DataFrame) -> tuple[list[dict[str, float]], float]:
    """Return per-feature SHAP contributions (+score points) and the base value."""
    shap_values = explainer.shap_values(feature_row[FEATURE_COLUMNS])
    expected = explainer.expected_value

    if isinstance(shap_values, list):  # [class0, class1]
        values = np.asarray(shap_values[1])[0]
        base = float(np.ravel(expected)[1]) if np.ndim(expected) else float(expected)
    else:
        values = np.asarray(shap_values)[0]
        base = float(np.ravel(expected)[0]) if np.ndim(expected) else float(expected)

    contributions = [
        {
            "feature": name,
            "shap_value": safe_float(val),
            "points": safe_round(shap_to_points(val), 1),
        }
        for name, val in zip(FEATURE_COLUMNS, values, strict=True)
    ]
    return contributions, base


def _top_drivers(contributions: list[dict[str, float]], top_n: int = 3) -> list[dict[str, float]]:
    ordered = sorted(contributions, key=lambda item: abs(item["shap_value"]), reverse=True)
    return ordered[:top_n]


def _finalize_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
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


def _build_payload(
    *,
    user_id: str,
    user_row: pd.Series,
    feature_row: pd.DataFrame,
    probability_of_default: float,
    explainer,
    norm_stats: dict,
) -> dict[str, Any]:
    """Assemble the full score payload for a single user (no DB I/O)."""
    auto_reject, reject_reason = check_red_flags(user_row)
    if auto_reject:
        probability_of_default = 1.0

    pillar_scores = compute_pillar_scores(user_row, norm_stats)
    confidence = compute_confidence(pillar_scores)

    credit_score = pd_to_credit_score(probability_of_default)
    decision = _decision_from_score(credit_score, auto_reject, confidence["confidence_pct"])

    contributions, base_value = _shap_contributions_for_row(explainer, feature_row)
    base_points = safe_round(expected_value_to_base_points(base_value), 1)
    factor_points = {item["feature"]: item["points"] for item in contributions}

    shap_drivers = _top_drivers(contributions)
    feature_trace = build_feature_trace(user_row, factor_points)
    reason_codes = shap_to_reason_codes(shap_drivers)
    if auto_reject and reject_reason:
        reason_codes.insert(0, reject_reason)
    if confidence["thin_file"]:
        reason_codes.append(
            "Thin-file applicant: limited alternative data ("
            + ", ".join(confidence["missing_sources"])
            + ")"
        )

    lending = recommend_terms(probability_of_default, credit_score, decision, user_row)

    return _finalize_score_payload(
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
            "base_points": base_points,
            "factor_points": factor_points,
            "feature_trace": feature_trace,
            "pillar_scores": pillar_scores,
            "confidence": confidence,
            "confidence_pct": confidence["confidence_pct"],
            "thin_file": confidence["thin_file"],
            "lending": lending,
            "model_version": get_model_version(),
        }
    )


async def score_user(session: AsyncSession, user_id: str, *, persist: bool = True) -> dict[str, Any]:
    """Compute full credit score payload for a single user."""
    wide = await fetch_features_wide(session)
    if wide.empty:
        raise ValueError(f"User {user_id} not found in ml_features")
    user_mask = wide["user_id"].astype(str) == str(user_id)
    if not user_mask.any():
        raise ValueError(f"User {user_id} not found in ml_features")

    norm_stats = compute_norm_stats(wide)
    model = get_cached_model()
    explainer = get_cached_explainer()

    user_wide = wide[user_mask]
    pd_df = predict_pd(model, user_wide)
    probability_of_default = safe_float(pd_df.iloc[0]["probability_of_default"])
    feature_row = get_feature_matrix_for_user(wide, user_id)

    result = _build_payload(
        user_id=str(user_id),
        user_row=user_wide.iloc[0],
        feature_row=feature_row,
        probability_of_default=probability_of_default,
        explainer=explainer,
        norm_stats=norm_stats,
    )

    if persist:
        await _persist_decision(session, result)
    return result


async def score_all_users(session: AsyncSession) -> list[dict[str, Any]]:
    """Score every user with features in the database."""
    wide = await fetch_features_wide(session)
    if wide.empty:
        return []

    norm_stats = compute_norm_stats(wide)
    model = get_cached_model()
    explainer = get_cached_explainer()
    pd_df = predict_pd(model, wide)

    results = []
    for _, row in wide.iterrows():
        user_id = str(row["user_id"])
        try:
            user_wide = wide[wide["user_id"].astype(str) == user_id]
            pd_row = pd_df[pd_df["user_id"] == user_id]
            probability_of_default = safe_float(pd_row.iloc[0]["probability_of_default"])
            feature_row = get_feature_matrix_for_user(wide, user_id)

            result = _build_payload(
                user_id=user_id,
                user_row=user_wide.iloc[0],
                feature_row=feature_row,
                probability_of_default=probability_of_default,
                explainer=explainer,
                norm_stats=norm_stats,
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
