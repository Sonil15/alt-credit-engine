"""Cached CatBoost model and SHAP explainer for fast scoring."""

import logging
from typing import Any

from models_ai.catboost_model import load_model
from models_ai.validation import load_model_card

logger = logging.getLogger(__name__)

_model = None
_explainer = None
_model_card: dict[str, Any] | None = None


def init_model_cache() -> None:
    """Load model and model card once at startup. SHAP explainer is lazy-initialized."""
    global _model, _model_card
    try:
        _model = load_model()
        _model_card = load_model_card()
        logger.info("Model cache initialized (version=%s)", get_model_version())
    except FileNotFoundError:
        logger.warning("Model not found at startup; train before scoring")
        _model = None
        _model_card = None


def get_cached_model():
    if _model is None:
        raise FileNotFoundError("Model not trained. Run `python -m models_ai.train` first.")
    return _model


def get_cached_explainer():
    global _explainer
    if _explainer is None:
        import shap
        model = get_cached_model()
        _explainer = shap.TreeExplainer(model)
    return _explainer


def get_model_version() -> str:
    if _model_card:
        return str(_model_card.get("model_version", "unknown"))
    return "unknown"


def get_model_card() -> dict[str, Any] | None:
    return _model_card


def reload_model_cache() -> None:
    init_model_cache()
