"""Risk-based lending recommendation derived from PD + score.

Turns a credit decision into actionable terms a loan officer can act on:
an eligible principal, a risk-priced interest rate, a tenure, and the resulting
EMI. Repayment capacity uses the borrower's observed monthly income (or
cash-flow proxy for MSMEs); pricing follows standard risk-based pricing (base
rate + a premium proportional to probability of default).
"""

from __future__ import annotations

import math

import pandas as pd

from core.json_utils import safe_float

# Annual interest-rate band (percent).
BASE_RATE = 11.0
MAX_RATE = 26.0
RISK_SPREAD = 20.0  # premium fully applied as PD -> 1

# Fixed-obligation-to-income ratio: the max share of monthly income allowed to
# service the new EMI, tightened as default risk rises.
MAX_FOIR = 0.45
MIN_FOIR = 0.10

def get_msme_capacity_multiplier(cohort: str | None) -> float:
    """Calculate MSME capacity multiplier based on expected digital ratio of the cohort."""
    if not cohort:
        return 1.5
    
    if cohort == "Farmer":
        expected_digital_ratio = 0.20
    elif cohort in ("Vendor", "Homemaker"):
        expected_digital_ratio = 0.40
    elif cohort in ("GigWorker", "Student"):
        expected_digital_ratio = 0.80
    else:
        return 1.5

    # Multiplier is the inverse of the digital ratio, capped at 3.0 for safety.
    return min(1.0 / expected_digital_ratio, 3.0)



def _round_to(value: float, step: int) -> float:
    if value <= 0:
        return 0.0
    return float(int(round(value / step)) * step)


def interest_rate_for_pd(pd_value: float) -> float:
    rate = BASE_RATE + RISK_SPREAD * max(0.0, min(1.0, pd_value))
    return round(max(BASE_RATE, min(MAX_RATE, rate)), 2)


def tenure_for_score(credit_score: int) -> int:
    if credit_score >= 640:
        return 36
    if credit_score >= 580:
        return 24
    if credit_score >= 480:
        return 18
    return 12


def _emi(principal: float, annual_rate_pct: float, months: int) -> float:
    if principal <= 0 or months <= 0:
        return 0.0
    r = annual_rate_pct / 100.0 / 12.0
    if r <= 0:
        return principal / months
    factor = (1 + r) ** months
    return principal * r * factor / (factor - 1)


def _principal_from_emi(emi: float, annual_rate_pct: float, months: int) -> float:
    if emi <= 0 or months <= 0:
        return 0.0
    r = annual_rate_pct / 100.0 / 12.0
    if r <= 0:
        return emi * months
    factor = (1 + r) ** months
    return emi * (factor - 1) / (r * factor)


def recommend_terms(
    probability_of_default: float,
    credit_score: int,
    decision: str,
    row: pd.Series,
    cohort: str | None = None,
) -> dict:
    """Recommend loan terms. Returns eligible=False for rejected applicants."""
    pd_value = max(0.0, min(1.0, safe_float(probability_of_default)))
    monthly_income = safe_float(row.get("monthly_income_mean", 0.0))
    is_msme = safe_float(row.get("borrower_type", 0.0)) >= 0.5
    if is_msme:
        monthly_income *= get_msme_capacity_multiplier(cohort)


    if decision == "REJECT" or monthly_income <= 0:
        return {
            "eligible": False,
            "max_loan_amount": 0.0,
            "interest_rate_pct": None,
            "tenure_months": None,
            "monthly_emi": 0.0,
            "borrower_type": "msme" if is_msme else "individual",
            "rationale": "Does not meet minimum risk/affordability criteria for an offer.",
        }

    foir = MAX_FOIR * (1.0 - pd_value)
    foir = max(MIN_FOIR, min(MAX_FOIR, foir))
    # Reviewed (not auto-approved) applicants get a more conservative offer.
    if decision == "REVIEW":
        foir *= 0.7

    affordable_emi = monthly_income * foir
    rate = interest_rate_for_pd(pd_value)
    tenure = tenure_for_score(credit_score)
    principal = _principal_from_emi(affordable_emi, rate, tenure)
    principal = _round_to(principal, 1000)
    emi = round(_emi(principal, rate, tenure), 2)

    return {
        "eligible": True,
        "max_loan_amount": principal,
        "interest_rate_pct": rate,
        "tenure_months": tenure,
        "monthly_emi": emi,
        "borrower_type": "msme" if is_msme else "individual",
        "rationale": (
            f"Risk-priced at {rate}% p.a. over {tenure} months; EMI capped at "
            f"{round(foir * 100)}% of assessed monthly income (₹{round(monthly_income):,})."
        ),
    }


def evaluate_funding_gap(decision: str, lending: dict, intake: dict | None) -> dict:
    """Affordability gate: does the model's approval actually cover the ask?

    Pure post-decision overlay. Never touches PD, score, or the model's
    ``decision``. When the model APPROVEs but the requested amount exceeds the
    maximum serviceable principal, the application must NOT go out as an
    approval: the outcome becomes REVIEW (counter-offer at the serviceable
    amount), with an explicit borrower-facing message.
    """
    if not intake:
        return {"gated": False}

    requested = safe_float(intake.get("requested_amount", 0.0))
    if requested <= 0:
        return {"gated": False}

    max_amount = safe_float(lending.get("max_loan_amount", 0.0))
    if decision != "APPROVE" or not lending.get("eligible") or requested <= max_amount:
        return {"gated": False, "requested_amount": requested}

    return {
        "gated": True,
        "requested_amount": requested,
        "max_serviceable_amount": max_amount,
        "message": (
            f"The requested amount of ₹{round(requested):,} exceeds the maximum "
            f"serviceable amount of ₹{round(max_amount):,} for this income profile. "
            "The application cannot be approved as requested; it is routed for a "
            f"counter-offer review at up to ₹{round(max_amount):,}."
        ),
    }
