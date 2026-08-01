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
from convergence.feature_meta import FEATURE_META, build_feature_trace
from convergence.lending import evaluate_funding_gap, recommend_terms
from convergence.letter_store import trim_pending_letter_queue, upsert_letter_for_decision
from convergence.panel import APPROVE_SCORE, REVIEW_SCORE, compute_agreement, decision_thresholds
from convergence.facets import BorrowerCohort, COHORT_CODE_MAP, compute_confidence, compute_norm_stats, compute_facet_scores
from convergence.reason_codes import format_reason_codes, drivers_to_reason_codes
from convergence.scorecard import (
    expected_value_to_base_points,
    pd_to_credit_score,
    ebm_to_points,
)
from core.business_profile import (
    BUSINESS_MODEL_FEATURES,
    PURPOSES_BY_COHORT,
    business_features_applicable,
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
    get_cached_ood_calibration,
    get_model_version,
)
from models_ai.conformal import apply_conformal_gate, conformal_report
from models_ai.ood import apply_ood_gate, ood_report
from models.db_models import ScoreDecision, BorrowerAccount
from sqlalchemy import select
from convergence.lending import _emi
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
    "telecom": ["missed_payments_count"],
    "ecommerce": ["monthly_spend_volatility"],
    "geo": ["anchor_count"],
    # Bank cash-flow, plus every econometric feature derived from the cash-flow series
    # (ECM + ADF). Revoking cash-flow must gate all of them or expense/stationarity
    # signals leak back into the model input and the driver explanation.
    "cashflow": [
        "cashflow_volatility", "cash_burn_rate",
        "resilience_coefficient", "trend_slope", "is_stationary",
        "adf_statistic", "adf_pvalue",
    ],
    "survey": [
        "conscientiousness", "locus_of_control", "financial_self_efficacy",
        "present_bias", "debt_attitude", "risk_tolerance",
        "delayed_gratification", "honesty", "cognitive_reflection", "resourcefulness",
        "response_validity"
    ],
    "campus": ["upi_spend_consistency"],
    "vendor": [],
    "farmer": ["harvest_income_spike"],
    "household": [],
    # Granular sub-scopes (2026-07)
    "upi_lite": ["upi_lite_txn_count", "upi_lite_average_ticket"],
    "dbt_logs": ["dbt_income_consistency"],
    "sms_parsing": ["sms_spend_total", "sms_bill_delay"],
    "enam_receipts": ["enam_receipt_volume"],
    "telco_postpaid_apis": ["avg_days_late"],
    "ondc_retail": ["avg_merchant_rating"],
    "amazon_flipkart_scraping": ["necessity_ratio"],
    "shipping_pin_codes": ["spatial_variance_score"],
    "aa_bank_statements": ["monthly_income_mean", "monthly_expense_mean"],
    "student_wallet_topups": ["e_wallet_topup_frequency"],
    "split_bill_history": ["small_dues_payment_promptness"],
    "ondc_merchant": ["average_ticket_size"],
    "upi_qr_dashboards": ["daily_transaction_count"],
    "kisan_credit_card": ["input_purchase_consistency"],
    "bbps_utility": ["utility_payment_consistency"],
    "grocery_pos": ["grocery_spend_stability"],
}


# The data scopes each cohort is actually scored on. ``geo`` and ``survey`` are
# universal; the remaining primary vault scopes (telecom / e-commerce / cash-flow)
# only apply where a cohort realistically has that footprint. Features from a scope a
# cohort does NOT use are dropped from the driver lists and the lineage trace (and
# their residual, near-zero contribution is folded into the baseline) so the audit
# trail never explains a student's score with a telecom bill the scorecard ignores.
# Each thin-file cohort's dedicated facet draws on more than the coarse scope name:
# its signals are spread across granular sub-scopes in ``SCOPE_TO_FEATURES`` (e.g. a
# vendor's ``daily_transaction_count`` lives under ``upi_qr_dashboards`` and
# ``average_ticket_size`` under ``ondc_merchant``). Every sub-scope a cohort's facet
# uses must be listed here, or the exclusion loop below folds those features into the
# baseline and the audit trail shows a Step 2 data source with no matching Step 3
# driver -- a student's split-bill or a vendor's QR settlements would vanish from the
# scorecard even though the facet scored them.
COHORT_EXPECTED_SCOPES = {
    "Salaried": ["telecom", "ecommerce", "geo", "cashflow", "survey"],
    "GigWorker": ["telecom", "ecommerce", "geo", "cashflow", "survey"],
    "Student": ["geo", "survey", "campus", "split_bill_history", "student_wallet_topups"],
    "Vendor": ["geo", "survey", "vendor", "upi_qr_dashboards", "ondc_merchant"],
    "Farmer": ["geo", "survey", "farmer", "kisan_credit_card", "enam_receipts"],
    "Homemaker": ["telecom", "geo", "survey", "household", "bbps_utility", "grocery_pos"],
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

    These are the model's additive terms in log-odds space, not a SHAP
    approximation. The raw terms are measured against the EBM intercept, which (because
    we train with balanced class weights) sits near a 50/50 coin-flip rather than the
    real ~9% applicant base rate. Against that intercept a typical low-risk borrower
    beats the baseline on nearly every feature, so the drivers come out all-positive
    and "needs work" signals never surface.

    We re-center each contribution on ``baseline_contrib``. The population-average
    contribution per feature (the typical applicant, from
    :func:`models_ai.ebm_model.ebm_mean_contributions`). A driver is then positive only
    when the borrower beats a typical applicant on that signal, and negative when they
    fall short. The intercept is shifted to the typical applicant's log-odds in lock
    step, so ``base_points + Σ points`` still reconstructs the score exactly.

    The ``contribution_value`` carries the centered EBM term contribution.
    """
    contrib_map, base = ebm_contributions(champion, feature_row)
    baseline_contrib = baseline_contrib or {}
    contributions = [
        {
            "feature": name,
            # Plain-language label from the single source of truth (FEATURE_META),
            # so the dashboard never surfaces raw feature names like
            # "locus_of_control" to a loan officer.
            "label": FEATURE_META.get(name, {}).get("label", name.replace("_", " ")),
            "contribution_value": (
                centered := safe_float(contrib_map.get(name, 0.0))
                - safe_float(baseline_contrib.get(name, 0.0))
            ),
            "points": safe_round(ebm_to_points(centered), 1),
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
    overrule the champion when the panel *genuinely* conflicts, not for adjacent
    boundary scatter (APPROVE vs REVIEW, or REVIEW vs REJECT), which is noise on a
    strict scorecard:

      - hard conflict (one model would APPROVE while another would REJECT) -> REVIEW
      - otherwise keep the champion's decision. A challenger sitting one band away
        (e.g. REVIEW while the champion APPROVES) is boundary scatter, not a veto:
        demoting every non-unanimous approval to review collapses the approve rate
        without adding signal, since an adjacent-band challenger raises no red flag
        the way a REJECT-vs-APPROVE split does.
    """
    if auto_reject:
        return "REJECT"
    if agreement["hard_conflict"]:
        return "REVIEW"
    return champion_decision


def _top_drivers(contributions: list[dict[str, float]], top_n: int = 3) -> list[dict[str, float]]:
    ordered = sorted(contributions, key=lambda item: abs(item["contribution_value"]), reverse=True)
    return ordered[:top_n]


def _top_negative_drivers(contributions: list[dict[str, float]], top_n: int = 3) -> list[dict[str, float]]:
    """The strongest *adverse* drivers, features that raised default risk (a positive
    centered contribution lowers the score). Selected on their own, not filtered out of
    a magnitude-ranked slice, so a rejected/marginal borrower always yields real
    adverse-action reason codes and targeted tips even when large positive drivers exist.
    """
    adverse = [item for item in contributions if safe_float(item["contribution_value"]) > 0]
    adverse.sort(key=lambda item: item["contribution_value"], reverse=True)
    return adverse[:top_n]


def _generate_actionable_insights(drivers: list[dict[str, float]]) -> list[str]:
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
        "cash_burn_rate": "Tip: Pacing your expenditures evenly throughout the month post-payday will improve your score.",
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
        "grocery_spend_stability": "Tip: Stable household spending patterns improve your credit assessment.",
        "business_vintage_years": "Tip: A longer business history indicates stability and helps improve your score.",
        "turnover_income_consistency": "Tip: Ensure your declared business turnover matches your actual bank transaction history.",
        # Granular sub-scope features (2026-07)
        "upi_lite_txn_count": "Tip: Regular use of UPI Lite for small transactions shows active digital usage.",
        "upi_lite_average_ticket": "Tip: Keeping a stable average ticket size for minor purchases builds credit history.",
        "dbt_income_consistency": "Tip: Receiving direct benefit transfers consistently confirms a stable baseline income.",
        "sms_spend_total": "Tip: Documenting regular e-commerce purchases via SMS alert records improves the credit profile.",
        "sms_bill_delay": "Tip: Paying bills quickly after receiving SMS alert alerts improves the telecom reliability score.",
        "enam_receipt_volume": "Tip: Linking e-NAM mandi verified crop sale receipts directly validates agricultural output."
    }
    
    insights = []
    # If a contribution_value is positive, it means it increased Probability of Default (hurt the score)
    for driver in drivers:
        feature = driver.get("feature", "")
        contrib_val = driver.get("contribution_value", 0.0)
        
        # Only provide tip if the feature was actually detrimental (contribution_value > 0)
        if contrib_val > 0.0 and feature in mapping:
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

    # Apply cohort score range bounds
    COHORT_SCORE_RANGES = {
        "Student": (330, 720),
        "Homemaker": (360, 730),
        "Farmer": (330, 800),
        "Vendor": (330, 850),
        "Salaried": (400, 850),
        "GigWorker": (350, 760),
    }
    raw_score = pd_to_credit_score(probability_of_default)
    min_score, max_score = COHORT_SCORE_RANGES.get(cohort, (300, 900))
    credit_score = max(min_score, min(max_score, raw_score))

    cohort_adjustment = credit_score - raw_score
    if cohort_adjustment != 0:
        from convergence.scorecard import SCORE_OFFSET, PDO_FACTOR, log_odds_to_pd
        log_odds_calibrated = (SCORE_OFFSET - credit_score) / PDO_FACTOR
        probability_of_default = log_odds_to_pd(log_odds_calibrated)

    auto_reject, reject_reason = check_red_flags(user_row)
    champion_pd = probability_of_default  # raw champion PD, used for panel agreement
    if auto_reject:
        probability_of_default = 1.0

    expect_business_profile = business_features_applicable(
        cohort,
        intake.get("loan_purpose") if intake else None,
        has_business_profile=bool(intake.get("has_business_profile")) if intake else False,
    )

    facet_scores = compute_facet_scores(
        user_row, norm_stats, cohort=cohort_enum, expect_business_profile=expect_business_profile
    )
    confidence = compute_confidence(
        facet_scores, cohort=cohort_enum, expect_business_profile=expect_business_profile
    )
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

    # OOD integrity gate: an anomalous *joint* feature vector (each coordinate plausible,
    # the combination absent from training) is where an additive model extrapolates
    # blindly - the signature of a gamed profile. Abstain to REVIEW, never reject.
    ood = ood_report(feature_row, get_cached_ood_calibration())
    pre_ood_decision = decision
    decision = apply_ood_gate(decision, auto_reject, ood)

    contributions, base_value = _champion_contributions(champion, feature_row, baseline_contrib)
    base_points = safe_round(expected_value_to_base_points(base_value), 1)
    factor_points = {item["feature"]: item["points"] for item in contributions}
    if cohort_adjustment != 0:
        factor_points["cohort_adjustment"] = cohort_adjustment

    # Never explain the score with data the borrower didn't consent to: drop every
    # feature belonging to a revoked scope from the driver lists and the lineage trace.
    # Also hide structurally-N/A business features (e.g. "years in business" for a
    # student laptop loan) even though the model may still consume them as 0.
    revoked_features: set[str] = set()
    for scope in (revoked_scopes or []):
        revoked_features.update(SCOPE_TO_FEATURES.get(scope, []))
    excluded_features = set(revoked_features)
    if not business_features_applicable(
        cohort,
        intake.get("loan_purpose") if intake else None,
        has_business_profile=bool(intake.get("has_business_profile")) if intake else False,
    ):
        excluded_features.update(BUSINESS_MODEL_FEATURES)

    # Exclude features that belong to scopes not applicable to this cohort
    expected_scopes = COHORT_EXPECTED_SCOPES.get(str(cohort), [])
    for scope_name, scope_features in SCOPE_TO_FEATURES.items():
        if scope_name not in expected_scopes:
            excluded_features.update(scope_features)

    visible_contributions = [c for c in contributions if c["feature"] not in excluded_features]

    feature_drivers = _top_drivers(visible_contributions)
    negative_drivers = _top_negative_drivers(visible_contributions)
    feature_trace = build_feature_trace(user_row, factor_points, exclude_features=excluded_features)

    # Retain all feature contributions in factor_points to reflect total EBM model
    # arithmetic in the scorecard. Individual feature traces filter unshared features.
    base_points = safe_round(
        credit_score - sum(safe_float(v) for v in factor_points.values()), 1
    )
    reason_codes = drivers_to_reason_codes(negative_drivers)
    gated_to_review = decision == "REVIEW" and champion_decision != "REVIEW" and not auto_reject
    conformal_gated = decision == "REVIEW" and pre_conformal_decision == "APPROVE" and conformal.get("abstain")
    ood_gated = decision == "REVIEW" and pre_ood_decision == "APPROVE" and ood.get("ood")
    if auto_reject and reject_reason:
        reason_codes.insert(0, reject_reason)
    elif ood_gated:
        reason_codes.insert(
            0,
            "Anomaly abstention: applicant's joint feature profile is statistically "
            "out-of-distribution versus the training population (Mahalanobis distance "
            f"{ood.get('severity', 0)}× the review threshold), routed to manual review",
        )
    elif conformal_gated:
        coverage_pct = round(float(conformal.get("coverage_target", 0.9)) * 100)
        reason_codes.insert(
            0,
            f"Conformal abstention: champion cannot guarantee default/non-default at "
            f"{coverage_pct}% coverage, routed to manual review",
        )
    elif gated_to_review:
        reason_codes.insert(
            0,
            "Model panel disagreement: champion (EBM) and challengers did not reach "
            "consensus, routed to manual review",
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

    lending = recommend_terms(probability_of_default, credit_score, decision, user_row, cohort=cohort)

    # Affordability gate, lending-policy overlay AFTER the model decision.
    # `decision` stays the model's call (fairness parity slices on it);
    # `final_outcome` is what the borrower is told.
    funding_gap = evaluate_funding_gap(decision, lending, intake)
    final_outcome = "REVIEW" if funding_gap.get("gated") else decision
    if funding_gap.get("gated"):
        reason_codes.insert(
            0,
            f"Affordability gate: requested amount ₹{round(funding_gap['requested_amount']):,} "
            f"exceeds maximum serviceable ₹{round(funding_gap['max_serviceable_amount']):,}, "
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
            "feature_drivers": feature_drivers,
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
            "loan_purpose_other_text": intake.get("loan_purpose_other_text") if intake else None,
            "purpose_consistent": purpose_consistent,
            "final_outcome": final_outcome,
            "funding_gap": funding_gap,
            "panel": agreement,
            "conformal": conformal,
            "ood": ood,
            "explanation_method": "ebm-additive-terms",
            "model_version": get_model_version(),
            "is_simulated": is_simulated,
            "cohort": cohort,
        }
    )


async def score_user(
    session: AsyncSession,
    user_id: str,
    *,
    pan: str | None = None,
    persist: bool = True,
    feature_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute full credit score payload for a single user, routing through CIBIL gate if set.

    ``feature_overrides`` (demo/red-team only): overwrite specific feature columns on the
    real feature vector *before* PD prediction, so a tampered profile is scored by the
    genuine engine - champion, panel, conformal, and the OOD gate all run for real. Never
    used on the persisted scoring path.
    """
    # First check the user's CIBIL score in BorrowerAccount
    result = await session.execute(
        select(BorrowerAccount).where(BorrowerAccount.user_id == UUID(str(user_id)))
    )
    account = result.scalar_one_or_none()
    
    intake_row = await fetch_latest_intake(session, user_id)
    intake = intake_to_dict(intake_row) if intake_row else None
    
    from core.bureau_mock import check_bureau_score
    bureau_score = check_bureau_score(pan)
    if bureau_score > 0:
        if account:
            account.cibil_score = bureau_score
            await session.commit()
            
        req_amount = intake.get("requested_amount", 100000.0) if intake else 100000.0
        loan_purpose = intake.get("loan_purpose", "personal") if intake else "personal"
        cohort = intake.get("cohort", "Salaried") if intake else "Salaried"
        emi_val = round(_emi(req_amount, 11.0, 36), 2)
        payload = {
            "user_id": str(user_id),
            "credit_score": bureau_score,
            "probability_of_default": 0.01,
            "decision": "TRADITIONAL_BUREAU_HIT",
            "approval_likelihood": "High",
            "actionable_insights": ["Tip: Maintain your traditional credit record."],
            "auto_reject": False,
            "reject_reason": None,
            "feature_drivers": [],
            "reason_codes": [f"Pre-Screening Gate: Fast-track approved via prime bureau history (Score: {bureau_score})"],
            "reason_codes_text": f"Pre-Screening Gate: Fast-track approved via prime bureau history (Score: {bureau_score})",
            "base_points": float(bureau_score),
            "factor_points": {},
            "feature_trace": {},
            "facet_scores": [],
            "confidence": {"confidence_pct": 100.0, "thin_file": False, "missing_sources": [], "cohort": cohort},
            "confidence_pct": 100.0,
            "thin_file": False,
            "lending": {
                "eligible": True,
                "max_loan_amount": req_amount,
                "interest_rate_pct": 11.0,
                "tenure_months": 36,
                "monthly_emi": emi_val,
                "borrower_type": "individual",
                "rationale": f"Fast-track approved via traditional bureau history (Score: {bureau_score}).",
            },
            "requested_amount": req_amount,
            "loan_purpose": loan_purpose,
            "loan_purpose_other_text": intake.get("loan_purpose_other_text") if intake else None,
            "purpose_consistent": True,
            "final_outcome": "TRADITIONAL_BUREAU_HIT",
            "funding_gap": {"gated": False},
            "panel": {"hard_conflict": False, "unanimous": True},
            "conformal": {"prediction_set": ["APPROVE"], "abstain": False, "enabled": False},
            "ood": {"ood": False, "enabled": False},
            "explanation_method": "bureau-gating",
            "model_version": "bureau-gate-1.0",
            "is_simulated": True,
            "cohort": cohort,
        }
        if persist:
            await _persist_decision(session, payload)
        return payload

    if account and account.cibil_score is not None and account.cibil_score != -1:
        cibil = account.cibil_score
        
        if cibil >= 750:
            # Prime fast-track
            req_amount = intake.get("requested_amount", 100000.0) if intake else 100000.0
            loan_purpose = intake.get("loan_purpose", "personal") if intake else "personal"
            emi_val = round(_emi(req_amount, 11.0, 36), 2)
            payload = {
                "user_id": str(user_id),
                "credit_score": cibil,
                "probability_of_default": 0.01,
                "decision": "APPROVE",
                "approval_likelihood": "High",
                "actionable_insights": ["Tip: Maintain your excellent credit record by continuing to pay all bills on time."],
                "auto_reject": False,
                "reject_reason": None,
                "feature_drivers": [],
                "reason_codes": [f"Fast-track approved via prime bureau history (CIBIL score: {cibil})"],
                "reason_codes_text": f"Fast-track approved via prime bureau history (CIBIL score: {cibil})",
                "base_points": 750.0,
                "factor_points": {},
                "feature_trace": {},
                "facet_scores": [],
                "confidence": {"confidence_pct": 100.0, "thin_file": False, "missing_sources": [], "cohort": "Salaried"},
                "confidence_pct": 100.0,
                "thin_file": False,
                "lending": {
                    "eligible": True,
                    "max_loan_amount": req_amount,
                    "interest_rate_pct": 11.0,
                    "tenure_months": 36,
                    "monthly_emi": emi_val,
                    "borrower_type": "individual",
                    "rationale": f"Fast-track approved via prime bureau history (CIBIL score: {cibil}). Risk-priced at 11.0% p.a. over 36 months.",
                },
                "requested_amount": req_amount,
                "loan_purpose": loan_purpose,
                "loan_purpose_other_text": intake.get("loan_purpose_other_text") if intake else None,
                "purpose_consistent": True,
                "final_outcome": "APPROVE",
                "funding_gap": {"gated": False},
                "panel": {"hard_conflict": False, "unanimous": True},
                "conformal": {"prediction_set": ["APPROVE"], "abstain": False, "enabled": False},
                "ood": {"ood": False, "enabled": False},
                "explanation_method": "bureau-gating",
                "model_version": "bureau-gate-1.0",
                "is_simulated": True,
            }
            if persist:
                await _persist_decision(session, payload)
            return payload
        elif cibil < 600:
            # Subprime reject
            req_amount = intake.get("requested_amount", 0.0) if intake else 0.0
            loan_purpose = intake.get("loan_purpose") if intake else None
            payload = {
                "user_id": str(user_id),
                "credit_score": cibil,
                "probability_of_default": 0.99,
                "decision": "REJECT",
                "approval_likelihood": "Needs Review",
                "actionable_insights": ["Tip: Address past defaults and outstanding dues to rebuild your credit history."],
                "auto_reject": True,
                "reject_reason": f"Rejected due to poor bureau history (CIBIL score: {cibil})",
                "feature_drivers": [],
                "reason_codes": [f"Rejected due to poor bureau history (CIBIL score: {cibil})"],
                "reason_codes_text": f"Rejected due to poor bureau history (CIBIL score: {cibil})",
                "base_points": 300.0,
                "factor_points": {},
                "feature_trace": {},
                "facet_scores": [],
                "confidence": {"confidence_pct": 100.0, "thin_file": False, "missing_sources": [], "cohort": "Salaried"},
                "confidence_pct": 100.0,
                "thin_file": False,
                "lending": {
                    "eligible": False,
                    "max_loan_amount": 0.0,
                    "interest_rate_pct": None,
                    "tenure_months": None,
                    "monthly_emi": 0.0,
                    "borrower_type": "individual",
                    "rationale": f"Rejected due to adverse bureau history (CIBIL score: {cibil}).",
                },
                "requested_amount": req_amount,
                "loan_purpose": loan_purpose,
                "loan_purpose_other_text": intake.get("loan_purpose_other_text") if intake else None,
                "purpose_consistent": None,
                "final_outcome": "REJECT",
                "funding_gap": {"gated": False},
                "panel": {"hard_conflict": False, "unanimous": True},
                "conformal": {"prediction_set": ["REJECT"], "abstain": False, "enabled": False},
                "ood": {"ood": False, "enabled": False},
                "explanation_method": "bureau-gating",
                "model_version": "bureau-gate-1.0",
                "is_simulated": True,
            }
            if persist:
                await _persist_decision(session, payload)
            return payload

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

    # Demo/red-team overlay: tamper the real feature vector before it hits the models, so
    # the gamed profile flows through the genuine PD → panel → conformal → OOD gate path.
    if feature_overrides:
        for feat, val in feature_overrides.items():
            if feat in user_wide.columns:
                user_wide[feat] = float(val)

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
    """Score every user (both alt-credit and bureau-gated) in the database."""
    # Fetch all borrower accounts to identify bureau-gated users
    accounts_res = await session.execute(select(BorrowerAccount))
    accounts = accounts_res.scalars().all()
    account_map = {str(a.user_id): a for a in accounts}

    wide = await fetch_features_wide(session)
    # Get all users who have features OR are bureau-gated
    user_ids_with_features = set(wide["user_id"].astype(str)) if not wide.empty else set()
    bureau_gated_user_ids = {uid for uid, acc in account_map.items() if acc.cibil_score is not None and acc.cibil_score != -1}
    all_target_users = user_ids_with_features.union(bureau_gated_user_ids)

    norm_stats = compute_norm_stats(wide) if not wide.empty else {}
    champion = get_cached_champion()
    challengers = get_cached_challengers()
    baseline_contrib = ebm_mean_contributions(champion, wide) if not wide.empty else {}
    intakes = await fetch_all_latest_intakes(session)
    from api.routes.consent import get_revoked_scopes

    results = []
    for user_id in all_target_users:
        try:
            # Resolve cohort early
            cohort = "Salaried"
            intake = intakes.get(user_id)
            if intake and intake.get("cohort"):
                cohort = intake.get("cohort")
            elif user_id in user_ids_with_features:
                user_wide = wide[wide["user_id"].astype(str) == user_id].copy()
                if "cohort_code" in user_wide.columns:
                    code = safe_float(user_wide.iloc[0].get("cohort_code", 0.0))
                    cohort = COHORT_CODE_MAP.get(code, BorrowerCohort.SALARIED).value
                else:
                    cohort = user_wide.iloc[0].get("cohort", "Salaried")

            account = account_map.get(user_id)
            if account and account.cibil_score is not None and account.cibil_score != -1:
                cibil = account.cibil_score
                intake = intakes.get(user_id)
                
                if cibil >= 750:
                    req_amount = intake.get("requested_amount", 100000.0) if intake else 100000.0
                    loan_purpose = intake.get("loan_purpose", "personal") if intake else "personal"
                    emi_val = round(_emi(req_amount, 11.0, 36), 2)
                    result = {
                        "user_id": str(user_id),
                        "credit_score": cibil,
                        "probability_of_default": 0.01,
                        "decision": "APPROVE",
                        "approval_likelihood": "High",
                        "actionable_insights": ["Tip: Maintain your excellent credit record by continuing to pay all bills on time."],
                        "auto_reject": False,
                        "reject_reason": None,
                        "feature_drivers": [],
                        "reason_codes": [f"Fast-track approved via prime bureau history (CIBIL score: {cibil})"],
                        "reason_codes_text": f"Fast-track approved via prime bureau history (CIBIL score: {cibil})",
                        "base_points": 750.0,
                        "factor_points": {},
                        "feature_trace": {},
                        "facet_scores": [],
                        "confidence": {"confidence_pct": 100.0, "thin_file": False, "missing_sources": [], "cohort": "Salaried"},
                        "confidence_pct": 100.0,
                        "thin_file": False,
                        "lending": {
                            "eligible": True,
                            "max_loan_amount": req_amount,
                            "interest_rate_pct": 11.0,
                            "tenure_months": 36,
                            "monthly_emi": emi_val,
                            "borrower_type": "individual",
                            "rationale": f"Fast-track approved via prime bureau history (CIBIL score: {cibil}). Risk-priced at 11.0% p.a. over 36 months.",
                        },
                        "requested_amount": req_amount,
                        "loan_purpose": loan_purpose,
                        "loan_purpose_other_text": intake.get("loan_purpose_other_text") if intake else None,
                        "purpose_consistent": True,
                        "final_outcome": "APPROVE",
                        "funding_gap": {"gated": False},
                        "panel": {"hard_conflict": False, "unanimous": True},
                        "conformal": {"prediction_set": ["APPROVE"], "abstain": False, "enabled": False},
                        "ood": {"ood": False, "enabled": False},
                        "explanation_method": "bureau-gating",
                        "model_version": "bureau-gate-1.0",
                        "is_simulated": True,
                        "cohort": cohort,
                    }
                    await _persist_decision(session, result)
                    results.append(result)
                    continue
                elif cibil < 600:
                    req_amount = intake.get("requested_amount", 0.0) if intake else 0.0
                    loan_purpose = intake.get("loan_purpose") if intake else None
                    result = {
                        "user_id": str(user_id),
                        "credit_score": cibil,
                        "probability_of_default": 0.99,
                        "decision": "REJECT",
                        "approval_likelihood": "Needs Review",
                        "actionable_insights": ["Tip: Address past defaults and outstanding dues to rebuild your credit history."],
                        "auto_reject": True,
                        "reject_reason": f"Rejected due to poor bureau history (CIBIL score: {cibil})",
                        "feature_drivers": [],
                        "reason_codes": [f"Rejected due to poor bureau history (CIBIL score: {cibil})"],
                        "reason_codes_text": f"Rejected due to poor bureau history (CIBIL score: {cibil})",
                        "base_points": 300.0,
                        "factor_points": {},
                        "feature_trace": {},
                        "facet_scores": [],
                        "confidence": {"confidence_pct": 100.0, "thin_file": False, "missing_sources": [], "cohort": "Salaried"},
                        "confidence_pct": 100.0,
                        "thin_file": False,
                        "lending": {
                            "eligible": False,
                            "max_loan_amount": 0.0,
                            "interest_rate_pct": None,
                            "tenure_months": None,
                            "monthly_emi": 0.0,
                            "borrower_type": "individual",
                            "rationale": f"Rejected due to adverse bureau history (CIBIL score: {cibil}).",
                        },
                        "requested_amount": req_amount,
                        "loan_purpose": loan_purpose,
                        "loan_purpose_other_text": intake.get("loan_purpose_other_text") if intake else None,
                        "purpose_consistent": None,
                        "final_outcome": "REJECT",
                        "funding_gap": {"gated": False},
                        "panel": {"hard_conflict": False, "unanimous": True},
                        "conformal": {"prediction_set": ["REJECT"], "abstain": False, "enabled": False},
                        "ood": {"ood": False, "enabled": False},
                        "explanation_method": "bureau-gating",
                        "model_version": "bureau-gate-1.0",
                        "is_simulated": True,
                        "cohort": cohort,
                    }
                    await _persist_decision(session, result)
                    results.append(result)
                    continue

            if user_id in user_ids_with_features:
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
            
    await trim_pending_letter_queue(session)
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
