"""Explainable Boosting Machine — the glass-box CHAMPION (model of record).

Unlike the CatBoost+SHAP path, an EBM is intrinsically interpretable: its
prediction is an additive sum of one shape function per feature,

    logit(PD) = intercept + Σ fᵢ(xᵢ)

so the per-feature contributions are not a post-hoc *approximation* (as SHAP is
for a black box) — they ARE the model's arithmetic. ``eval_terms`` returns those
contributions directly, and ``sigmoid(intercept + Σ terms) == predict_proba``
to machine precision. We train with ``interactions=0`` so every term maps to a
single feature, which keeps the contributions one-to-one with FEATURE_COLUMNS and
lets us publish a stable points table.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from core.seeds import CATBOOST_RANDOM_SEED
from models_ai.constants import FEATURE_COLUMNS, fill_missing_features

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = MODEL_DIR / "ebm_champion.pkl"


def train_ebm(df: pd.DataFrame, label_series: pd.Series) -> ExplainableBoostingClassifier:
    """Train the additive glass-box champion on the wide feature matrix."""
    if df.empty or len(df) < 5:
        raise ValueError("Need at least 5 users with features to train EBM")
    features = fill_missing_features(df.copy())
    y = label_series.astype(int)
    if y.nunique() < 2:
        raise ValueError("Need both default and non-default labels for training")

    model = ExplainableBoostingClassifier(
        feature_names=FEATURE_COLUMNS,
        interactions=0,  # pure additive: one term per feature -> publishable points
        random_state=CATBOOST_RANDOM_SEED,
    )
    model.fit(features, y, sample_weight=compute_sample_weight("balanced", y))
    return model


def save_ebm(model: ExplainableBoostingClassifier, path: Path | None = None) -> Path:
    target = path or MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Saved EBM champion to %s", target)
    return target


def load_ebm(path: Path | None = None) -> ExplainableBoostingClassifier:
    target = path or MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(f"EBM champion not found at {target}. Run models_ai/train.py first.")
    with open(target, "rb") as fh:
        return pickle.load(fh)


def predict_pd(model: ExplainableBoostingClassifier, df: pd.DataFrame) -> pd.DataFrame:
    """Return user_id and probability_of_default for each row."""
    features = fill_missing_features(df.copy())
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "user_id": df["user_id"].astype(str).values,
            "probability_of_default": probabilities.astype(float),
        }
    )


def ebm_contributions(
    model: ExplainableBoostingClassifier, feature_row: pd.DataFrame
) -> tuple[dict[str, float], float]:
    """Per-feature log-odds contributions for one borrower, plus the intercept.

    Returns ``({feature: contribution}, intercept)`` in log-odds space. Because the
    model is additive, ``intercept + Σ contributions == logit(predict_proba)`` exactly
    — the same reconciliation the SHAP path relied on, but with no approximation.
    """
    features = fill_missing_features(feature_row.copy())
    terms = np.asarray(model.eval_terms(features))[0]  # (n_terms,) for this row
    intercept = float(np.ravel(model.intercept_)[0])

    contributions: dict[str, float] = {feat: 0.0 for feat in FEATURE_COLUMNS}
    for term_idx, term_features in enumerate(model.term_features_):
        if len(term_features) != 1:
            continue  # interactions=0, but guard against pair terms regardless
        feat_name = model.term_names_[term_idx]
        if feat_name in contributions:
            contributions[feat_name] = float(terms[term_idx])
    return contributions, intercept


def _finite(value: float, fallback: float) -> float:
    v = float(value)
    return v if np.isfinite(v) else fallback


def ebm_shape_functions(model: ExplainableBoostingClassifier) -> list[dict]:
    """Global per-feature shape functions — the curves that ARE the model.

    Returns, for each main-effect feature, the bin edges ``x`` (length N+1) and the
    log-odds contribution ``logodds`` (length N, a step value per interval). This is
    the model's own decision surface, not a post-hoc explanation: reading a curve at
    a borrower's feature value gives exactly the contribution the model used.
    """
    glob = model.explain_global()
    out: list[dict] = []
    for term_idx, term_features in enumerate(model.term_features_):
        if len(term_features) != 1:
            continue
        data = glob.data(term_idx)
        names = [_finite(v, 0.0) for v in data["names"]]
        scores = [_finite(v, 0.0) for v in data["scores"]]
        out.append({"feature": model.term_names_[term_idx], "x": names, "logodds": scores})
    return out
