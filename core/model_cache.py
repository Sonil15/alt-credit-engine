"""Cached champion (EBM) + challenger panel (CatBoost, logistic) for fast scoring.

The champion is the model of record: it drives the published score and supplies
the (intrinsic, no-SHAP) explanation. Challengers are loaded only to vote on
agreement. All are loaded once at startup.
"""

import logging
from typing import Any

from models_ai.catboost_model import load_model as load_catboost
from models_ai.conformal import load_calibration
from models_ai.ood import load_calibration as load_ood_calibration
from models_ai.ebm_model import load_ebm
from models_ai.logistic_model import load_logistic
from models_ai.validation import load_model_card

logger = logging.getLogger(__name__)

_champion = None
_challengers: dict[str, Any] = {}
_model_card: dict[str, Any] | None = None
_conformal_calibration: dict[str, Any] | None = None
_ood_calibration: dict[str, Any] | None = None


def init_model_cache() -> None:
    """Load champion + challengers + model card once at startup."""
    global _champion, _challengers, _model_card, _conformal_calibration, _ood_calibration
    try:
        _champion = load_ebm()
        _challengers = {"catboost": load_catboost(), "logistic": load_logistic()}
        _model_card = load_model_card()
        _conformal_calibration = load_calibration()
        _ood_calibration = load_ood_calibration()
        logger.info("Model cache initialized (champion=ebm, version=%s)", get_model_version())
    except FileNotFoundError as exc:
        logger.warning("Model panel not fully trained at startup (%s); train before scoring", exc)
        _champion = None
        _challengers = {}
        _model_card = None
        _conformal_calibration = None
        _ood_calibration = None


def get_cached_champion():
    if _champion is None:
        raise FileNotFoundError("Champion model not trained. Run `python -m models_ai.train` first.")
    return _champion


def get_cached_challengers() -> dict[str, Any]:
    if not _challengers:
        raise FileNotFoundError("Challenger models not trained. Run `python -m models_ai.train` first.")
    return _challengers


def get_model_version() -> str:
    if _model_card:
        return str(_model_card.get("model_version", "unknown"))
    return "unknown"


def get_model_card() -> dict[str, Any] | None:
    return _model_card


def get_cached_conformal_calibration() -> dict[str, Any] | None:
    return _conformal_calibration


def get_cached_ood_calibration() -> dict[str, Any] | None:
    return _ood_calibration


def reload_model_cache() -> None:
    init_model_cache()
