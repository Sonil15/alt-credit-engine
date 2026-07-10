"""Logistic-regression CHALLENGER, a structurally different family.

The panel's value comes from diversity: agreement between two boosted-tree models
is nearly tautological, but agreement between an additive GAM (EBM champion), a
boosted-tree model (CatBoost) and a *linear* model (this) is informative. This
model exists only to vote on agreement; it does not drive the published score.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.seeds import CATBOOST_RANDOM_SEED
from models_ai.constants import fill_missing_features, prior_correction_log_odds

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = MODEL_DIR / "logistic_challenger.pkl"


def train_logistic(df: pd.DataFrame, label_series: pd.Series) -> Pipeline:
    """Train a standardized, class-balanced logistic-regression challenger."""
    if df.empty or len(df) < 5:
        raise ValueError("Need at least 5 users with features to train logistic challenger")
    features = fill_missing_features(df.copy())
    y = label_series.astype(int)
    if y.nunique() < 2:
        raise ValueError("Need both default and non-default labels for training")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=CATBOOST_RANDOM_SEED,
                ),
            ),
        ]
    )
    model.fit(features, y)
    # Undo the balanced-weight recentering (see prior_correction_log_odds): shift the
    # intercept so the challenger's PD scale matches the champion's honest prior.
    model.named_steps["clf"].intercept_ = (
        model.named_steps["clf"].intercept_ + prior_correction_log_odds(y)
    )
    return model


def save_logistic(model: Pipeline, path: Path | None = None) -> Path:
    target = path or MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Saved logistic challenger to %s", target)
    return target


def load_logistic(path: Path | None = None) -> Pipeline:
    target = path or MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(f"Logistic challenger not found at {target}. Run models_ai/train.py first.")
    with open(target, "rb") as fh:
        model = pickle.load(fh)
        
    # Patch for sklearn version mismatch
    if hasattr(model, "steps"):
        for name, step in model.steps:
            if hasattr(step, "predict_proba") and not hasattr(step, "multi_class"):
                step.multi_class = "auto"
                
    return model


def predict_pd(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Return user_id and probability_of_default for each row."""
    features = fill_missing_features(df.copy())
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "user_id": df["user_id"].astype(str).values,
            "probability_of_default": probabilities.astype(float),
        }
    )
