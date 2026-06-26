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
]

LABEL_COLUMN = "default_label"


def fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0)
