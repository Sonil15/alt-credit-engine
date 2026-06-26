"""Fuse econometric + AI outputs into interpretable credit scores."""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from convergence.fairness import compute_fairness_report
from convergence.feature_meta import build_feature_trace
from convergence.lending import recommend_terms
from convergence.panel import APPROVE_SCORE, REVIEW_SCORE, compute_agreement, decision_thresholds
from convergence.pillars import BorrowerCohort, COHORT_CODE_MAP, compute_confidence, compute_norm_stats, compute_pillar_scores
from convergence.reason_codes import format_reason_codes, shap_to_reason_codes
from convergence.scorecard import (
    expected_value_to_base_points,
    pd_to_credit_score,
    shap_to_points,
)
from core.feature_store import fetch_features_wide
from core.json_utils import safe_float, safe_round, sanitize_for_json
from core.model_cache import (
    get_cached_challengers,
    get_cached_champion,
    get_cached_conformal_calibration,
    get_model_version,
)
from models_ai.conformal import apply_conformal_gate, conformal_report
from models.db_models import ScoreDecision
from models_ai.catboost_model import get_feature_matrix_for_user
from models_ai.constants import FEATURE_COLUMNS
from models_ai.ebm_model import ebm_contributions
from models_ai.ebm_model import predict_pd as champion_predict_pd

logger = logging.getLogger(__name__)

SPATIAL_VARIANCE_THRESHOLD = 50.0
MISSED_PAYMENTS_THRESHOLD = 5
# Below this confidence we never silently auto-approve a thin file.
LOW_CONFIDENCE_PCT = 60.0


def check_red_flags(row: pd.Series) -> tuple[bool, str | None]:
    spatial = safe_float(row.get("spatial_variance_score", 0.0))
    income = safe_float(row.get("monthly_income_mean", 0.0))

    if spatial > SPATIAL_VARIANCE_THRESHOLD and income <= 0:
        return True, "Auto-reject: high geographic instability with zero baseline income"

    if "cohort_code" in row:
        code = safe_float(row.get("cohort_code", 0.0))
        is_salaried = (code == 0.0)
    else:
        cohort = row.get("cohort", "Salaried")
        if isinstance(cohort, float) or pd.isna(cohort) or cohort is None:
            is_salaried = True
        else:
            is_salaried = (str(cohort) == "Salaried")

    if is_salaried:
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


def _champion_contributions(champion, feature_row: pd.DataFrame) -> tuple[list[dict[str, float]], float]:
    """Per-feature contributions (+score points) from the EBM champion's own terms.

    These are the model's additive terms in log-odds space — not a SHAP
    approximation — so ``base_points + Σ points`` reconstructs the score exactly.
    The ``shap_value`` key is retained for payload/schema compatibility; it now
    carries the EBM term contribution rather than a Shapley value.
    """
    contrib_map, base = ebm_contributions(champion, feature_row)
    contributions = [
        {
            "feature": name,
            "shap_value": safe_float(contrib_map.get(name, 0.0)),
            "points": safe_round(shap_to_points(contrib_map.get(name, 0.0)), 1),
        }
        for name in FEATURE_COLUMNS
    ]
    return contributions, base


def _challenger_pd(model, feature_row: pd.DataFrame) -> float:
    """Run one challenger on the prepared feature row and return its PD."""
    return safe_float(model.predict_proba(feature_row[FEATURE_COLUMNS])[:, 1][0])


def _apply_agreement_gate(champion_decision: str, auto_reject: bool, agreement: dict) -> str:
    """Auto-decisions require panel support; genuine disagreement routes to a human.

    Hard red-flag rejects always stand (rules override models). Otherwise we only
    overrule the champion when the panel *genuinely* conflicts — not for adjacent
    boundary scatter (e.g. REVIEW vs REJECT), which is noise on a strict scorecard:

      - hard conflict (one model would APPROVE while another would REJECT) -> REVIEW
      - a contested APPROVE (champion approves but the panel isn't unanimous) -> REVIEW
      - otherwise keep the champion's decision (challengers agree closely enough)
    """
    if auto_reject:
        return "REJECT"
    if agreement["hard_conflict"]:
        return "REVIEW"
    if champion_decision == "APPROVE" and not agreement["unanimous"]:
        return "REVIEW"
    return champion_decision


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
    champion,
    challengers: dict,
    norm_stats: dict,
) -> dict[str, Any]:
    """Assemble the full score payload for a single user (no DB I/O)."""
    auto_reject, reject_reason = check_red_flags(user_row)
    champion_pd = probability_of_default  # raw champion PD, used for panel agreement
    if auto_reject:
        probability_of_default = 1.0

    # Resolve cohort early
    if "cohort_code" in user_row:
        code = safe_float(user_row.get("cohort_code", 0.0))
        cohort_enum = COHORT_CODE_MAP.get(code, BorrowerCohort.SALARIED)
    else:
        cohort_str = user_row.get("cohort", "Salaried")
        if isinstance(cohort_str, float) or pd.isna(cohort_str) or cohort_str is None:
            cohort_enum = BorrowerCohort.SALARIED
        else:
            try:
                cohort_enum = BorrowerCohort(str(cohort_str))
            except ValueError:
                cohort_enum = BorrowerCohort.SALARIED

    cohort = cohort_enum.value if hasattr(cohort_enum, "value") else str(cohort_enum)

    pillar_scores = compute_pillar_scores(user_row, norm_stats, cohort=cohort_enum)
    confidence = compute_confidence(pillar_scores, cohort=cohort_enum)
    confidence["cohort"] = str(cohort)

    credit_score = pd_to_credit_score(probability_of_default)

    # Champion proposes; challengers audit. Disagreement -> manual review.
    challenger_pds = {name: _challenger_pd(model, feature_row) for name, model in challengers.items()}
    agreement = compute_agreement(champion_pd, challenger_pds)
    champion_decision = _decision_from_score(credit_score, auto_reject, confidence["confidence_pct"])
    decision = _apply_agreement_gate(champion_decision, auto_reject, agreement)

    conformal_calibration = get_cached_conformal_calibration()
    conformal = (
        conformal_report(champion_pd, conformal_calibration)
        if conformal_calibration
        else {"prediction_set": [], "abstain": False, "enabled": False}
    )
    pre_conformal_decision = decision
    decision = apply_conformal_gate(decision, auto_reject, conformal)

    contributions, base_value = _champion_contributions(champion, feature_row)
    base_points = safe_round(expected_value_to_base_points(base_value), 1)
    factor_points = {item["feature"]: item["points"] for item in contributions}

    shap_drivers = _top_drivers(contributions)
    feature_trace = build_feature_trace(user_row, factor_points)
    reason_codes = shap_to_reason_codes(shap_drivers)
    gated_to_review = decision == "REVIEW" and champion_decision != "REVIEW" and not auto_reject
    conformal_gated = decision == "REVIEW" and pre_conformal_decision == "APPROVE" and conformal.get("abstain")
    if auto_reject and reject_reason:
        reason_codes.insert(0, reject_reason)
    elif conformal_gated:
        coverage_pct = round(float(conformal.get("coverage_target", 0.9)) * 100)
        reason_codes.insert(
            0,
            f"Conformal abstention: champion cannot guarantee default/non-default at "
            f"{coverage_pct}% coverage — routed to manual review",
        )
    elif gated_to_review:
        reason_codes.insert(
            0,
            "Model panel disagreement: champion (EBM) and challengers did not reach "
            "consensus — routed to manual review",
        )
    if confidence["thin_file"]:
        reason_codes.append(
            "Thin-file applicant: limited alternative data ("
            + ", ".join(confidence["missing_sources"])
            + ")"
        )

    lending = recommend_terms(probability_of_default, credit_score, decision, user_row)
    is_simulated = bool(safe_float(user_row.get("is_simulated", 0.0)) == 1.0)

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
            "panel": agreement,
            "conformal": conformal,
            "explanation_method": "ebm-additive-terms",
            "model_version": get_model_version(),
            "is_simulated": is_simulated,
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
    champion = get_cached_champion()
    challengers = get_cached_challengers()

    user_wide = wide[user_mask]
    pd_df = champion_predict_pd(champion, user_wide)
    probability_of_default = safe_float(pd_df.iloc[0]["probability_of_default"])
    feature_row = get_feature_matrix_for_user(wide, user_id)

    result = _build_payload(
        user_id=str(user_id),
        user_row=user_wide.iloc[0],
        feature_row=feature_row,
        probability_of_default=probability_of_default,
        champion=champion,
        challengers=challengers,
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
    champion = get_cached_champion()
    challengers = get_cached_challengers()
    pd_df = champion_predict_pd(champion, wide)

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
                champion=champion,
                challengers=challengers,
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
            "score_distribution": _empty_score_distribution(),
            "decision_thresholds": decision_thresholds(),
            "fairness": compute_fairness_report([], wide),
        }

    total = len(scores)
    approvals = sum(1 for s in scores if s["decision"] == "APPROVE")
    reviews = sum(1 for s in scores if s["decision"] == "REVIEW")
    rejects = sum(1 for s in scores if s["decision"] == "REJECT")
    avg_pd = sum(s["probability_of_default"] for s in scores) / total
    avg_score = sum(s["credit_score"] for s in scores) / total

    distribution = _empty_score_distribution()
    reject_key, review_key, approve_key = _score_distribution_keys()
    for s in scores:
        cs = s["credit_score"]
        if cs < REVIEW_SCORE:
            distribution[reject_key] += 1
        elif cs < APPROVE_SCORE:
            distribution[review_key] += 1
        else:
            distribution[approve_key] += 1

    return {
        "total_users": total,
        "approval_rate": round(approvals / total, 4),
        "review_rate": round(reviews / total, 4),
        "reject_rate": round(rejects / total, 4),
        "expected_default_rate": round(avg_pd, 4),
        "avg_score": round(avg_score, 2),
        "score_distribution": distribution,
        "decision_thresholds": decision_thresholds(),
        "fairness": compute_fairness_report(scores, wide),
    }


def _score_distribution_keys() -> tuple[str, str, str]:
    return (
        f"300-{REVIEW_SCORE - 1}",
        f"{REVIEW_SCORE}-{APPROVE_SCORE - 1}",
        f"{APPROVE_SCORE}-900",
    )


def _empty_score_distribution() -> dict[str, int]:
    reject_key, review_key, approve_key = _score_distribution_keys()
    return {reject_key: 0, review_key: 0, approve_key: 0}
