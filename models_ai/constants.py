"""Shared model constants and feature helpers."""

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
    "turnover_income_consistency",
]

LABEL_COLUMN = "default_label"


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
