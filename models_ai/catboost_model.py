"""CatBoost classifier for probability of default from ml_features."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from sqlalchemy.ext.asyncio import AsyncSession

from core.feature_store import fetch_features_wide
from core.seeds import CATBOOST_RANDOM_SEED
from models_ai.constants import FEATURE_COLUMNS, LABEL_COLUMN, fill_missing_features
from models_ai.validation import (
    build_model_card,
    cross_validate_metrics,
    evaluate_model,
    save_model_card,
    train_test_split_data,
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


async def train_from_db(session: AsyncSession) -> dict[str, Any]:
    """Load features from DB, train CatBoost with validation, save artifact + model card."""
    wide = await fetch_features_wide(session)
    if wide.empty:
        raise ValueError("No ml_features available for training")

    labels = extract_labels(wide)
    features = fill_missing_features(wide.copy())

    X_train, X_test, y_train, y_test = train_test_split_data(features, labels)
    model = train_catboost(
        X_train.reset_index(drop=True),
        label_series=y_train.reset_index(drop=True),
    )

    holdout_metrics = evaluate_model(model, X_test, y_test)
    cv_metrics = cross_validate_metrics(features, labels)

    path = save_model(model)
    card = build_model_card(
        holdout_metrics,
        users_trained=len(wide),
        feature_columns=FEATURE_COLUMNS,
        cv_metrics=cv_metrics,
    )
    save_model_card(card)

    return {
        "users_trained": len(wide),
        "default_rate": float(labels.mean()),
        "model_path": str(path),
        "feature_count": len(FEATURE_COLUMNS),
        "model_version": card["model_version"],
        "metrics": holdout_metrics,
        "cv_metrics": cv_metrics,
    }
