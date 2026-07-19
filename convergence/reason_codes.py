"""Plain-language adverse action reason codes from EBM drivers."""

FEATURE_REASON_MAP: dict[str, str] = {
    "avg_days_late": "Irregular bill payment timing",
    "missed_payments_count": "Missed telecom or utility payments",
    "necessity_ratio": "Low share of essential spending",
    "avg_merchant_rating": "Low merchant quality in purchase history",
    "monthly_spend_volatility": "High spending volatility",
    "spatial_variance_score": "High shipping address drift",
    "anchor_count": "Multiple shipping destinations",
    "monthly_income_mean": "Low or unstable income",
    "monthly_expense_mean": "High recurring expenses",
    "cashflow_volatility": "Volatile cash flow patterns",
    "cash_burn_rate": "Rapid post-payday cash depletion",
    "conscientiousness": "Signs of less careful financial planning",
    "locus_of_control": "Feels finances are outside their control",
    "financial_self_efficacy": "Low confidence managing personal finances",
    "present_bias": "Tends to spend impulsively rather than save",
    "debt_attitude": "Weak commitment to timely debt repayment",
    "response_validity": "Inconsistent answers in the assessment",
    "resilience_coefficient": "Low financial resilience",
    "adf_statistic": "Non-stationary income/expense pattern",
    "adf_pvalue": "Unstable long-run cash flow equilibrium",
    "is_stationary": "Unstable long-run cash flow equilibrium",
    "trend_slope": "Declining income trend",
    "historical_spatial_variance": "Inconsistent delivery locations",
    "distinct_pin_codes": "Multiple delivery addresses",
    "business_vintage_years": "Limited business operating history",
    "turnover_income_consistency": "Declared income does not match observed cash flow",
    # Cohort-specific features reason codes
    "upi_spend_consistency": "Inconsistent UPI spending patterns",
    "small_dues_payment_promptness": "Delays in small dues repayment",
    "e_wallet_topup_frequency": "Infrequent digital wallet usage",
    "daily_transaction_count": "Low business transaction volume",
    "average_ticket_size": "Small average ticket size",
    "harvest_income_spike": "Weak harvest income cycles",
    "input_purchase_consistency": "Irregular agricultural input purchases",
    "utility_payment_consistency": "Inconsistent utility bill payments",
    "grocery_spend_stability": "Unstable household spending patterns",
    # Granular sub-scope features (2026-07)
    "upi_lite_txn_count": "Low UPI Lite transaction volume",
    "upi_lite_average_ticket": "Small UPI Lite average transaction size",
    "dbt_income_consistency": "Irregular direct benefit transfer receipts",
    "sms_spend_total": "Low SMS-parsed transaction volume",
    "sms_bill_delay": "High delay in paying bills after SMS alerts",
    "enam_receipt_volume": "Low e-NAM verified Mandi sales volume",
}


def drivers_to_reason_codes(drivers: list[dict[str, float]], top_n: int = 3) -> list[str]:
    """Convert EBM drivers to human-readable adverse action reason codes."""
    reasons: list[str] = []
    for driver in drivers[:top_n]:
        feature = driver.get("feature", "")
        contribution_value = float(driver.get("contribution_value", 0.0))
        if contribution_value <= 0:
            continue
        label = FEATURE_REASON_MAP.get(feature, feature.replace("_", " ").title())
        reasons.append(label)
    return reasons


def format_reason_codes(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "No adverse factors identified."
    return "; ".join(reason_codes)
