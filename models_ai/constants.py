"""Shared model constants and feature helpers."""

import math

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "avg_days_late",
    "missed_payments_count",
    "necessity_ratio",
    "avg_merchant_rating",
    "monthly_spend_volatility",
    "spatial_variance_score",
    "anchor_count",
    "monthly_income_mean",
    "monthly_expense_mean",
    "cashflow_volatility",
    "conscientiousness",
    "locus_of_control",
    "financial_self_efficacy",
    "present_bias",
    "debt_attitude",
    "response_validity",
    "resilience_coefficient",
    "adf_statistic",
    "adf_pvalue",
    "is_stationary",
    "trend_slope",
    # Borrower onboarding, self-declared business profile (Vendor/Farmer).
    # Absent for individuals; fill_missing_features maps absent -> 0.0.
    "business_vintage_years",
    "is_new_business",
    "turnover_income_consistency",
    "has_udyam_registration",
    "years_informal",
    # Cohort-specific facet features
    "upi_spend_consistency",
    "small_dues_payment_promptness",
    "e_wallet_topup_frequency",
    "daily_transaction_count",
    "average_ticket_size",
    "harvest_income_spike",
    "input_purchase_consistency",
    "utility_payment_consistency",
    "grocery_spend_stability",
]

LABEL_COLUMN = "default_label"


def prior_correction_log_odds(labels: pd.Series) -> float:
    """Log-odds shift that undoes balanced class weighting.

    All panel models train with balanced class weights so the minority default class
    carries signal, but that re-weighting recenters the fitted intercept at ~50/50
    coin-flip odds instead of the portfolio's real base rate. Adding
    ``log(n_default / n_good)`` to the model's intercept (raw-score bias) restores
    honest PD levels: the population-average borrower's PD lands back at the observed
    base rate, which is the scale the PDO scorecard and the conformal gate expect.
    """
    y = pd.Series(labels).astype(int)
    n_default = int(y.sum())
    n_good = int(len(y) - n_default)
    if n_default == 0 or n_good == 0:
        return 0.0
    return math.log(n_default / n_good)


def calculate_prior_correction_shift(p_raw: np.ndarray, y: pd.Series) -> float:
    """Compute the intercept shift required to align predictions to the true prior log-odds."""
    p_clipped = np.clip(p_raw, 1e-9, 1.0 - 1e-9)
    logits = np.log(p_clipped / (1.0 - p_clipped))
    target = float(y.mean())
    if target <= 0 or target >= 1:
        return 0.0
    
    # Binary search to solve mean(1 / (1 + exp(-(logits + shift)))) == target
    low, high = -20.0, 20.0
    for _ in range(50):
        mid = (low + high) / 2.0
        p_pred = 1.0 / (1.0 + np.exp(-(logits + mid)))
        if np.mean(p_pred) < target:
            low = mid
        else:
            high = mid
    return float(low)


def fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature matrix with absent values imputed.

    Missing or non-finite cells are filled with the *typical applicant* value for the
    borrower's cohort (see ``models_ai.imputation``) rather than ``0.0``, so an absent
    data source resolves to "unknown → typical" instead of a directionally biased
    extreme. Any column the imputation profile cannot cover (e.g. a feature that is
    structurally not applicable to the cohort, or before the profile artifact exists)
    falls back to ``0.0``.
    """
    # Lazy import keeps this module free of a circular dependency: imputation imports
    # FEATURE_COLUMNS from here.
    from models_ai.imputation import imputation_fill_frame

    fill = imputation_fill_frame(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    features = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    # Per-row, per-cohort typical-applicant fill; anything still absent -> 0.0.
    return features.fillna(fill).fillna(0.0)
