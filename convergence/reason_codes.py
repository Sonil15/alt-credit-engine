"""Plain-language adverse action reason codes from SHAP drivers."""

FEATURE_REASON_MAP: dict[str, str] = {
    "avg_days_late": "Irregular bill payment timing",
    "missed_payments_count": "Missed telecom or utility payments",
    "necessity_ratio": "Low share of essential spending",
    "avg_merchant_rating": "Low merchant quality in purchase history",
    "monthly_spend_volatility": "High spending volatility",
    "spatial_variance_score": "Geographic instability",
    "anchor_count": "Limited location anchors",
    "monthly_income_mean": "Low or unstable income",
    "monthly_expense_mean": "High recurring expenses",
    "cashflow_volatility": "Volatile cash flow patterns",
    "conscientiousness": "Low conscientiousness in financial habits",
    "locus_of_control": "External locus of control regarding finances",
    "financial_self_efficacy": "Low confidence managing personal finances",
    "present_bias": "High impulsive or present-biased spending",
    "debt_attitude": "Weak commitment to timely debt repayment",
    "response_validity": "Inconsistent psychometric responses",
    "resilience_coefficient": "Low financial resilience",
    "adf_statistic": "Non-stationary income/expense pattern",
    "adf_pvalue": "Unstable long-run cash flow equilibrium",
    "is_stationary": "Unstable long-run cash flow equilibrium",
    "historical_spatial_variance": "Inconsistent delivery locations",
    "distinct_pin_codes": "Multiple delivery addresses",
}


def shap_to_reason_codes(shap_drivers: list[dict[str, float]], top_n: int = 3) -> list[str]:
    """Convert SHAP drivers to human-readable adverse action reason codes."""
    reasons: list[str] = []
    for driver in shap_drivers[:top_n]:
        feature = driver.get("feature", "")
        shap_value = float(driver.get("shap_value", 0.0))
        if shap_value <= 0:
            continue
        label = FEATURE_REASON_MAP.get(feature, feature.replace("_", " ").title())
        reasons.append(label)
    return reasons


def format_reason_codes(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "No adverse factors identified."
    return "; ".join(reason_codes)
