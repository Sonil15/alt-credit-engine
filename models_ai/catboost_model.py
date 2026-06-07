"""CatBoost classifier for probability of default from ml_features."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sqlalchemy.ext.asyncio import AsyncSession

from core.feature_store import fetch_features_wide

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = MODEL_DIR / "catboost_model.cbm"

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
    "risk_appetite",
    "savings_freq",
    "financial_stress_score",
    "intent_label_score",
    "resilience_coefficient",
    "adf_statistic",
    "adf_pvalue",
    "is_stationary",
]

LABEL_COLUMN = "default_label"


def _fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLUMNS].fillna(0.0)


def generate_synthetic_labels(df: pd.DataFrame) -> pd.Series:
    """
    Hackathon-only heuristic labels for training when no ground truth exists.
    High risk signals => default_label = 1.
    """
    labels = pd.Series(0, index=df.index, dtype=int)

    high_late = df.get("avg_days_late", pd.Series(0, index=df.index)).fillna(0) > 10
    high_missed = df.get("missed_payments_count", pd.Series(0, index=df.index)).fillna(0) >= 3
    high_volatility = df.get("cashflow_volatility", pd.Series(0, index=df.index)).fillna(0) > 5000
    low_resilience = df.get("resilience_coefficient", pd.Series(0.5, index=df.index)).fillna(0.5) < 0.2
    high_stress = df.get("financial_stress_score", pd.Series(0, index=df.index)).fillna(0) > 0.7
    zero_income = df.get("monthly_income_mean", pd.Series(0, index=df.index)).fillna(0) <= 0

    risk_score = (
        high_late.astype(int)
        + high_missed.astype(int)
        + high_volatility.astype(int)
        + low_resilience.astype(int)
        + high_stress.astype(int)
        + zero_income.astype(int)
    )
    labels = (risk_score >= 2).astype(int)
    return labels


def train_catboost(df: pd.DataFrame) -> CatBoostClassifier:
    """Train CatBoost on wide feature matrix with synthetic labels."""
    if df.empty or len(df) < 5:
        raise ValueError("Need at least 5 users with features to train CatBoost")

    df = _fill_missing_features(df.copy())
    y = generate_synthetic_labels(df)

    if y.nunique() < 2:
        # Ensure both classes exist for training
        y.iloc[0] = 0
        y.iloc[-1] = 1

    model = CatBoostClassifier(
        iterations=100,
        depth=4,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced",
    )
    model.fit(df[FEATURE_COLUMNS], y)
    return model


def save_model(model: CatBoostClassifier, path: Path | None = None) -> Path:
    target = path or MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(target))
    logger.info("Saved CatBoost model to %s", target)
    return target


def load_model(path: Path | None = None) -> CatBoostClassifier:
    target = path or MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(f"CatBoost model not found at {target}. Run models_ai/train.py first.")
    model = CatBoostClassifier()
    model.load_model(str(target))
    return model


def predict_pd(model: CatBoostClassifier, df: pd.DataFrame) -> pd.DataFrame:
    """Return user_id and probability_of_default for each row."""
    features = _fill_missing_features(df.copy())
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "user_id": df["user_id"].astype(str).values,
            "probability_of_default": probabilities.astype(float),
        }
    )


def get_feature_matrix_for_user(df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    row = df[df["user_id"].astype(str) == str(user_id)]
    if row.empty:
        raise ValueError(f"No features found for user {user_id}")
    return _fill_missing_features(row.copy())


async def train_from_db(session: AsyncSession) -> dict[str, Any]:
    """Load features from DB, train CatBoost, save artifact."""
    wide = await fetch_features_wide(session)
    if wide.empty:
        raise ValueError("No ml_features available for training")

    model = train_catboost(wide)
    path = save_model(model)
    labels = generate_synthetic_labels(_fill_missing_features(wide.copy()))
    return {
        "users_trained": len(wide),
        "default_rate": float(labels.mean()),
        "model_path": str(path),
        "feature_count": len(FEATURE_COLUMNS),
    }
