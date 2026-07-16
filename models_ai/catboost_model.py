"""CatBoost classifier for probability of default from ml_features."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from core.seeds import CATBOOST_RANDOM_SEED
from models_ai.constants import (
    LABEL_COLUMN,
    fill_missing_features,
    calculate_prior_correction_shift,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = MODEL_DIR / "catboost_model.cbm"


def extract_labels(df: pd.DataFrame) -> pd.Series:
    """Read ground-truth default labels stored during mock data ingestion."""
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Missing {LABEL_COLUMN} in ml_features. "
            "Reload mock data with ground_truth ingestion enabled."
        )
    return df[LABEL_COLUMN].fillna(0).astype(int)


def train_catboost(
    df: pd.DataFrame,
    label_series: pd.Series | None = None,
) -> CatBoostClassifier:
    """Train CatBoost on wide feature matrix using generative ground-truth labels."""
    if df.empty or len(df) < 5:
        raise ValueError("Need at least 5 users with features to train CatBoost")

    features = fill_missing_features(df.copy())
    y = label_series if label_series is not None else extract_labels(df)

    if y.nunique() < 2:
        raise ValueError("Need both default and non-default labels for training")

    model = CatBoostClassifier(
        iterations=150,
        depth=4,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=CATBOOST_RANDOM_SEED,
        verbose=False,
        auto_class_weights="Balanced",
    )
    model.fit(features, y)
    # Correct scale and bias/intercept to match target prior log-odds exactly
    p_raw = model.predict_proba(features)[:, 1]
    shift = calculate_prior_correction_shift(p_raw, y)
    scale, bias = model.get_scale_and_bias()
    model.set_scale_and_bias(scale, bias + shift)
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
    features = fill_missing_features(df.copy())
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
    return fill_missing_features(row.copy())


