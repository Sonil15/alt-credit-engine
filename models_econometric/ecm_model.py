"""ADF stationarity test and Error Correction Model for resilience scoring."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from statsmodels.tsa.stattools import adfuller

from core.feature_store import fetch_all_series, fetch_features_wide, upsert_features_batch

logger = logging.getLogger(__name__)

MIN_OBSERVATIONS = 4
PRIMARY_SERIES = "monthly_net_cashflow"
FALLBACK_SERIES = "telecom_payment_rate"


def _run_adf(series: pd.Series) -> dict[str, float]:
    if len(series) < MIN_OBSERVATIONS:
        return {"adf_statistic": 0.0, "adf_pvalue": 1.0, "is_stationary": 0.0}

    try:
        adf_stat, pvalue, *_ = adfuller(series.values, autolag="AIC")
        return {
            "adf_statistic": float(adf_stat),
            "adf_pvalue": float(pvalue),
            "is_stationary": 1.0 if pvalue < 0.05 else 0.0,
        }
    except Exception:
        return {"adf_statistic": 0.0, "adf_pvalue": 1.0, "is_stationary": 0.0}


def _run_ecm(series: pd.Series) -> float:
    """
    Estimate a single-equation ECM:
    delta_y_t = alpha + gamma * (y_{t-1} - y_bar) + epsilon_t

    Resilience coefficient = clamp(-gamma, 0, 1).
    """
    if len(series) < MIN_OBSERVATIONS:
        return 0.5

    y = series.astype(float)
    equilibrium = float(y.mean())
    y_lag = y.shift(1)
    delta_y = y.diff()
    error_term = y_lag - equilibrium

    valid = pd.concat([delta_y, error_term], axis=1).dropna()
    if len(valid) < 3:
        return 0.5

    delta = valid.iloc[:, 0].values
    ect = valid.iloc[:, 1].values
    x = np.column_stack([np.ones(len(ect)), ect])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x, delta, rcond=None)
        gamma = float(coeffs[1])
    except Exception:
        return 0.5

    resilience = float(np.clip(-gamma, 0.0, 1.0))
    adf = _run_adf(series)
    if adf["is_stationary"]:
        resilience = min(1.0, resilience + 0.1)
    return resilience


def compute_resilience_from_series(values: list[float]) -> dict[str, float]:
    """Run ADF + ECM on a real stored time series."""
    if not values:
        return {
            "resilience_coefficient": 0.5,
            "adf_statistic": 0.0,
            "adf_pvalue": 1.0,
            "is_stationary": 0.0,
        }

    series = pd.Series(values, dtype=float)
    adf_metrics = _run_adf(series)
    resilience = _run_ecm(series)
    return {
        "resilience_coefficient": resilience,
        "adf_statistic": adf_metrics["adf_statistic"],
        "adf_pvalue": adf_metrics["adf_pvalue"],
        "is_stationary": adf_metrics["is_stationary"],
    }


async def run_ecm_pipeline(session: AsyncSession) -> dict[str, Any]:
    """Compute ECM resilience coefficients from real stored time series."""
    wide = await fetch_features_wide(session)
    if wide.empty:
        logger.warning("No ml_features found; skipping ECM pipeline")
        return {"users_processed": 0, "features_written": 0}

    cashflow_series = await fetch_all_series(session, PRIMARY_SERIES)
    telecom_series = await fetch_all_series(session, FALLBACK_SERIES)

    user_features: dict[str, dict[str, float]] = {}
    for user_id in wide["user_id"].astype(str):
        values = cashflow_series.get(user_id) or telecom_series.get(user_id) or []
        user_features[user_id] = compute_resilience_from_series(values)

    await upsert_features_batch(session, user_features)
    feature_count = sum(len(v) for v in user_features.values())
    logger.info("ECM pipeline complete: %d users, %d features", len(user_features), feature_count)
    return {"users_processed": len(user_features), "features_written": feature_count}
