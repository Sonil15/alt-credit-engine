"""ADF stationarity test and Error Correction Model for resilience scoring."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from statsmodels.tsa.stattools import adfuller

from core.feature_store import fetch_features_wide, upsert_features_batch

logger = logging.getLogger(__name__)

MIN_OBSERVATIONS = 6
DEFAULT_PERIODS = 12


def _build_cashflow_proxy_series(
    income_mean: float,
    expense_mean: float,
    volatility: float,
    user_id: str,
    periods: int = DEFAULT_PERIODS,
) -> pd.Series:
    """Construct a monthly net-cashflow proxy series from aggregated ml_features."""
    seed = abs(hash(user_id)) % (2**32)
    rng = np.random.default_rng(seed)
    base_net = income_mean - expense_mean
    noise_scale = max(volatility, abs(base_net) * 0.05, 1.0)
    values = base_net + rng.normal(0.0, noise_scale, periods)
    index = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq="ME")
    return pd.Series(values, index=index, name="net_cashflow")


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
    Higher values indicate faster reversion to equilibrium (more resilient).
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

    # OLS: delta = alpha + gamma * ECT
    x = np.column_stack([np.ones(len(ect)), ect])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x, delta, rcond=None)
        gamma = float(coeffs[1])
    except Exception:
        return 0.5

    # Negative gamma => mean reversion; map to [0, 1]
    resilience = float(np.clip(-gamma, 0.0, 1.0))

    # Boost score for stationary series (ADF passed)
    adf = _run_adf(series)
    if adf["is_stationary"]:
        resilience = min(1.0, resilience + 0.1)

    return resilience


def compute_resilience_for_user(row: pd.Series) -> dict[str, float]:
    user_id = str(row["user_id"])
    income = float(row.get("monthly_income_mean", 0.0) or 0.0)
    expense = float(row.get("monthly_expense_mean", 0.0) or 0.0)
    volatility = float(row.get("cashflow_volatility", 0.0) or 0.0)

    series = _build_cashflow_proxy_series(income, expense, volatility, user_id)
    adf_metrics = _run_adf(series)
    resilience = _run_ecm(series)

    return {
        "resilience_coefficient": resilience,
        "adf_statistic": adf_metrics["adf_statistic"],
        "adf_pvalue": adf_metrics["adf_pvalue"],
        "is_stationary": adf_metrics["is_stationary"],
    }


async def run_ecm_pipeline(session: AsyncSession) -> dict[str, Any]:
    """Compute ECM resilience coefficients for all users and persist to ml_features."""
    wide = await fetch_features_wide(session)
    if wide.empty:
        logger.warning("No ml_features found; skipping ECM pipeline")
        return {"users_processed": 0, "features_written": 0}

    required = {"monthly_income_mean", "monthly_expense_mean", "cashflow_volatility"}
    if not required.issubset(set(wide.columns)):
        logger.warning("Missing cashflow features for ECM: %s", required - set(wide.columns))
        return {"users_processed": 0, "features_written": 0}

    user_features: dict[str, dict[str, float]] = {}
    for _, row in wide.iterrows():
        metrics = compute_resilience_for_user(row)
        user_features[str(row["user_id"])] = metrics

    await upsert_features_batch(session, user_features)
    feature_count = sum(len(v) for v in user_features.values())
    logger.info("ECM pipeline complete: %d users, %d features", len(user_features), feature_count)
    return {"users_processed": len(user_features), "features_written": feature_count}
