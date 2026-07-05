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
from convergence.lending import evaluate_funding_gap, recommend_terms
from convergence.letter_store import upsert_letter_for_decision
from convergence.panel import APPROVE_SCORE, REVIEW_SCORE, compute_agreement, decision_thresholds
from convergence.facets import BorrowerCohort, COHORT_CODE_MAP, compute_confidence, compute_norm_stats, compute_facet_scores
from convergence.reason_codes import format_reason_codes, shap_to_reason_codes
from convergence.scorecard import (
    expected_value_to_base_points,
    pd_to_credit_score,
    shap_to_points,
)
from core.business_profile import (
    PURPOSES_BY_COHORT,
    fetch_all_latest_intakes,
    fetch_latest_intake,
    intake_to_dict,
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
from models_ai.ebm_model import ebm_contributions, ebm_mean_contributions
from models_ai.ebm_model import predict_pd as champion_predict_pd

logger = logging.getLogger(__name__)

SPATIAL_VARIANCE_THRESHOLD = 50.0
MISSED_PAYMENTS_THRESHOLD = 5
# Below this confidence we never silently auto-approve a thin file.
LOW_CONFIDENCE_PCT = 60.0

SCOPE_TO_FEATURES = {
    "telecom": ["avg_days_late", "missed_payments_count"],
    "ecommerce": ["necessity_ratio", "avg_merchant_rating", "monthly_spend_volatility"],
    "geo": ["spatial_variance_score", "anchor_count"],
    # Bank cash-flow, plus every econometric feature derived from the cash-flow series
    # (ECM + ADF). Revoking cash-flow must gate all of them or expense/stationarity
    # signals leak back into the model input and the driver explanation.
    "cashflow": [
        "monthly_income_mean", "monthly_expense_mean", "cashflow_volatility",
        "resilience_coefficient", "trend_slope", "is_stationary",
        "adf_statistic", "adf_pvalue",
    ],
    "survey": [
        "conscientiousness", "locus_of_control", "financial_self_efficacy",
        "present_bias", "debt_attitude", "risk_tolerance",
        "delayed_gratification", "honesty", "cognitive_reflection", "resourcefulness"
    ],
    "campus": ["upi_spend_consistency", "small_dues_payment_promptness", "e_wallet_topup_frequency"],
    "vendor": ["daily_transaction_count", "average_ticket_size"],
    "farmer": ["harvest_income_spike", "input_purchase_consistency"],
    "household": ["utility_payment_consistency", "grocery_spend_stability"]
}


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


def _champion_contributions(
    champion,
    feature_row: pd.DataFrame,
    baseline_contrib: dict[str, float] | None = None,
) -> tuple[list[dict[str, float]], float]:
    """Per-feature contributions (+score points) from the EBM champion's own terms,
    centered on the *typical applicant*.

    These are the model's additive terms in log-odds space — not a SHAP
    approximation. The raw terms are measured against the EBM intercept, which (because
    we train with balanced class weights) sits near a 50/50 coin-flip rather than the
    real ~9% applicant base rate. Against that intercept a typical low-risk borrower
    beats the baseline on nearly every feature, so the drivers come out all-positive
    and "needs work" signals never surface.

    We re-center each contribution on ``baseline_contrib`` — the population-average
    contribution per feature (the typical applicant, from
    :func:`models_ai.ebm_model.ebm_mean_contributions`). A driver is then positive only
    when the borrower beats a typical applicant on that signal, and negative when they
    fall short. The intercept is shifted to the typical applicant's log-odds in lock
    step, so ``base_points + Σ points`` still reconstructs the score exactly.

    The ``shap_value`` key is retained for payload/schema compatibility; it now
    carries the centered EBM term contribution rather than a Shapley value.
    """
    contrib_map, base = ebm_contributions(champion, feature_row)
    baseline_contrib = baseline_contrib or {}
    contributions = [
        {
            "feature": name,
            "shap_value": (
                centered := safe_float(contrib_map.get(name, 0.0))
                - safe_float(baseline_contrib.get(name, 0.0))
            ),
            "points": safe_round(shap_to_points(centered), 1),
        }
        for name in FEATURE_COLUMNS
    ]
    # Shift the baseline so base_points = the typical applicant's score; preserves
    # base_points + Σ centered_points == credit_score (PD/score itself untouched).
    typical_base = base + sum(safe_float(v) for v in baseline_contrib.values())
    return contributions, typical_base


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


def _top_negative_drivers(contributions: list[dict[str, float]], top_n: int = 3) -> list[dict[str, float]]:
    """The strongest *adverse* drivers — features that raised default risk (a positive
    centered contribution lowers the score). Selected on their own, not filtered out of
    a magnitude-ranked slice, so a rejected/marginal borrower always yields real
    adverse-action reason codes and targeted tips even when large positive drivers exist.
    """
    adverse = [item for item in contributions if safe_float(item["shap_value"]) > 0]
    adverse.sort(key=lambda item: item["shap_value"], reverse=True)
    return adverse[:top_n]


def _generate_actionable_insights(shap_drivers: list[dict[str, float]]) -> list[str]:
    mapping = {
        "avg_days_late": "Tip: Ensuring all telecom and utility bills are paid on time improves your score.",
        "missed_payments_count": "Tip: Ensuring all telecom and utility bills are paid on time improves your score.",
        "necessity_ratio": "Tip: Maintaining a healthy balance of essential spending helps your profile.",
        "avg_merchant_rating": "Tip: Higher merchant ratings on transactions indicate better digital reliability.",
        "monthly_spend_volatility": "Tip: Demonstrating stable spending patterns will improve your score.",
        "spatial_variance_score": "Tip: Showing location stability over time helps build your profile.",
        "anchor_count": "Tip: Establishing routine locations (like home/work) improves profile confidence.",
        "monthly_income_mean": "Tip: Increasing or stabilizing your monthly inflow strengthens your assessment.",
        "cashflow_volatility": "Tip: Demonstrating consistent monthly cash flow will improve your score.",
        "resilience_coefficient": "Tip: Maintaining a buffer in your account improves financial resilience.",
        "trend_slope": "Tip: A positive trend in your account balance over time will improve your score.",
        "is_stationary": "Tip: Reducing unpredictable spikes in cash flow will strengthen your profile.",
        "upi_spend_consistency": "Tip: Consistent digital payment habits build a stronger profile.",
        "small_dues_payment_promptness": "Tip: Prompt payment of small dues builds positive credit history.",
        "e_wallet_topup_frequency": "Tip: Regular usage of digital wallets can positively impact your score.",
        "daily_transaction_count": "Tip: Higher transaction volume indicates a healthier micro-enterprise.",
        "average_ticket_size": "Tip: Stable or growing average ticket sizes improve your business profile.",
        "harvest_income_spike": "Tip: Consistent cycles in farming income help validate agricultural profiles.",
        "input_purchase_consistency": "Tip: Regular purchases of agricultural inputs build a stronger profile.",
        "utility_payment_consistency": "Tip: Consistent payment of household utilities is a strong positive signal.",
        "grocery_spend_stability": "Tip: Stable household spending patterns improve your credit assessment."
    }
    
    insights = []
    # If a shap_value is positive, it means it increased Probability of Default (hurt the score)
    for driver in shap_drivers:
        feature = driver.get("feature", "")
        shap_val = driver.get("shap_value", 0.0)
        
        # Only provide tip if the feature was actually detrimental (shap_value > 0)
        if shap_val > 0.0 and feature in mapping:
            insights.append(mapping[feature])
            
    # Remove duplicates and limit to 3
    unique_insights = list(dict.fromkeys(insights))
    
    if not unique_insights:
        unique_insights.append("Tip: Continue building a consistent digital transaction history across all your accounts.")
        
    return unique_insights[:3]


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
            requested_amount=payload.get("requested_amount"),
            loan_purpose=payload.get("loan_purpose"),
            final_outcome=payload.get("final_outcome"),
        )
    )
    # Draft/refresh the borrower's decision letter alongside the audit row: approvals
    # issue automatically; rejections/reviews queue for officer sign-off.
    await upsert_letter_for_decision(session, payload)
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
    baseline_contrib: dict[str, float] | None = None,
    revoked_scopes: list[str] | None = None,
    intake: dict[str, Any] | None = None,
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

    facet_scores = compute_facet_scores(user_row, norm_stats, cohort=cohort_enum)
    confidence = compute_confidence(facet_scores, cohort=cohort_enum)
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

    contributions, base_value = _champion_contributions(champion, feature_row, baseline_contrib)
    base_points = safe_round(expected_value_to_base_points(base_value), 1)
    factor_points = {item["feature"]: item["points"] for item in contributions}

    # Never explain the score with data the borrower didn't consent to: drop every
    # feature belonging to a revoked scope from the driver lists and the lineage trace.
    revoked_features: set[str] = set()
    for scope in (revoked_scopes or []):
        revoked_features.update(SCOPE_TO_FEATURES.get(scope, []))
    visible_contributions = [c for c in contributions if c["feature"] not in revoked_features]

    shap_drivers = _top_drivers(visible_contributions)
    negative_drivers = _top_negative_drivers(visible_contributions)
    feature_trace = build_feature_trace(user_row, factor_points, exclude_features=revoked_features)
    reason_codes = shap_to_reason_codes(negative_drivers)
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
    if revoked_scopes:
        reason_codes.append(
            f"Consent withdrawn for data source(s): {', '.join(revoked_scopes)}"
        )

    lending = recommend_terms(probability_of_default, credit_score, decision, user_row)

    # Affordability gate — lending-policy overlay AFTER the model decision.
    # `decision` stays the model's call (fairness parity slices on it);
    # `final_outcome` is what the borrower is told.
    funding_gap = evaluate_funding_gap(decision, lending, intake)
    final_outcome = "REVIEW" if funding_gap.get("gated") else decision
    if funding_gap.get("gated"):
        reason_codes.insert(
            0,
            f"Affordability gate: requested amount ₹{round(funding_gap['requested_amount']):,} "
            f"exceeds maximum serviceable ₹{round(funding_gap['max_serviceable_amount']):,} — "
            "not approved as requested; routed for counter-offer review",
        )

    loan_purpose = intake.get("loan_purpose") if intake else None
    purpose_consistent = None
    if intake and loan_purpose:
        allowed = PURPOSES_BY_COHORT.get(str(intake.get("cohort", "")), [])
        purpose_consistent = loan_purpose in allowed

    is_simulated = bool(safe_float(user_row.get("is_simulated", 0.0)) == 1.0)
    
    approval_likelihood = "High" if decision == "APPROVE" else "Moderate" if decision == "REVIEW" else "Needs Review"
    actionable_insights = _generate_actionable_insights(negative_drivers)

    return _finalize_score_payload(
        {
            "user_id": str(user_id),
            "credit_score": credit_score,
            "probability_of_default": safe_round(probability_of_default, 4),
            "decision": decision,
            "approval_likelihood": approval_likelihood,
            "actionable_insights": actionable_insights,
            "auto_reject": auto_reject,
            "reject_reason": reject_reason,
            "shap_drivers": shap_drivers,
            "reason_codes": reason_codes,
            "reason_codes_text": format_reason_codes(reason_codes),
            "base_points": base_points,
            "factor_points": factor_points,
            "feature_trace": feature_trace,
            "facet_scores": facet_scores,
            "confidence": confidence,
            "confidence_pct": confidence["confidence_pct"],
            "thin_file": confidence["thin_file"],
            "lending": lending,
            "requested_amount": safe_float(intake.get("requested_amount")) if intake else None,
            "loan_purpose": loan_purpose,
            "purpose_consistent": purpose_consistent,
            "final_outcome": final_outcome,
            "funding_gap": funding_gap,
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
    # Typical-applicant reference for centering the driver explanation (see
    # _champion_contributions). Computed over the full applicant population.
    baseline_contrib = ebm_mean_contributions(champion, wide)

    user_wide = wide[user_mask].copy()

    # Mask features for revoked scopes
    from api.routes.consent import get_revoked_scopes
    revoked_scopes = get_revoked_scopes(user_id)
    if revoked_scopes:
        for scope in revoked_scopes:
            features_to_mask = SCOPE_TO_FEATURES.get(scope, [])
            for feat in features_to_mask:
                if feat in user_wide.columns:
                    user_wide[feat] = np.nan

    pd_df = champion_predict_pd(champion, user_wide)
    probability_of_default = safe_float(pd_df.iloc[0]["probability_of_default"])
    feature_row = get_feature_matrix_for_user(user_wide, user_id)
    intake = intake_to_dict(await fetch_latest_intake(session, user_id))

    result = _build_payload(
        user_id=str(user_id),
        user_row=user_wide.iloc[0],
        feature_row=feature_row,
        probability_of_default=probability_of_default,
        champion=champion,
        challengers=challengers,
        norm_stats=norm_stats,
        baseline_contrib=baseline_contrib,
        revoked_scopes=revoked_scopes,
        intake=intake,
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
    # Typical-applicant reference, computed once over the full population (see
    # _champion_contributions); shared across every borrower in the portfolio.
    baseline_contrib = ebm_mean_contributions(champion, wide)
    intakes = await fetch_all_latest_intakes(session)
    from api.routes.consent import get_revoked_scopes

    results = []
    for _, row in wide.iterrows():
        user_id = str(row["user_id"])
        try:
            user_wide = wide[wide["user_id"].astype(str) == user_id].copy()
            revoked_scopes = get_revoked_scopes(user_id)
            if revoked_scopes:
                for scope in revoked_scopes:
                    features_to_mask = SCOPE_TO_FEATURES.get(scope, [])
                    for feat in features_to_mask:
                        if feat in user_wide.columns:
                            user_wide[feat] = np.nan

            pd_row = champion_predict_pd(champion, user_wide)
            probability_of_default = safe_float(pd_row.iloc[0]["probability_of_default"])
            feature_row = get_feature_matrix_for_user(user_wide, user_id)

            result = _build_payload(
                user_id=user_id,
                user_row=user_wide.iloc[0],
                feature_row=feature_row,
                probability_of_default=probability_of_default,
                champion=champion,
                challengers=challengers,
                norm_stats=norm_stats,
                baseline_contrib=baseline_contrib,
                revoked_scopes=revoked_scopes,
                intake=intakes.get(user_id),
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
