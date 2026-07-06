import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from convergence.feature_meta import FEATURE_META
from convergence.scorecard import shap_to_points
from convergence.score_engine import portfolio_summary, score_all_users, score_user
from core.auth import get_session_user_id, require_api_key, require_own_session
from core.database import AsyncSessionLocal, get_db
from core.model_cache import get_cached_champion, get_model_card, get_model_version, reload_model_cache
from models_ai.ebm_model import ebm_shape_functions
from models.pydantic_schemas import (
    CreditScoreResponse,
    ModelCardResponse,
    ModelMetrics,
    PortfolioSummaryResponse,
    TrainResponse,
)
from models_ai.ensemble import train_all_from_db
from models_econometric.ecm_model import run_ecm_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/score", tags=["scoring"])


@router.get("/me", response_model=CreditScoreResponse)
async def get_my_credit_score(
    user_id: str = Depends(get_session_user_id),
    db: AsyncSession = Depends(get_db),
) -> CreditScoreResponse:
    """Return the authenticated borrower's own credit score."""
    try:
        result = await score_user(db, user_id)
        return CreditScoreResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python -m models_ai.train` first.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Scoring failed for user %s", user_id)
        raise HTTPException(status_code=500, detail="Scoring failed") from exc


@router.get("/", response_model=list[CreditScoreResponse], dependencies=[Depends(require_api_key)])
async def list_credit_scores(
    db: AsyncSession = Depends(get_db),
) -> list[CreditScoreResponse]:
    """Return credit scores for all users with features."""
    try:
        results = await score_all_users(db)
        return [CreditScoreResponse(**item) for item in results]
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python -m models_ai.train` first.",
        ) from exc


@router.get("/portfolio/summary", response_model=PortfolioSummaryResponse, dependencies=[Depends(require_api_key)])
async def get_portfolio_summary(
    db: AsyncSession = Depends(get_db),
) -> PortfolioSummaryResponse:
    """Portfolio-level metrics for bank dashboard."""
    try:
        summary = await portfolio_summary(db)
        return PortfolioSummaryResponse(**summary)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python -m models_ai.train` first.",
        ) from exc


@router.get("/model/card", response_model=ModelCardResponse)
async def get_model_card_endpoint() -> ModelCardResponse:
    """Return persisted model card with validation metrics."""
    card = get_model_card()
    if not card:
        raise HTTPException(status_code=404, detail="Model card not found. Train the model first.")
    return ModelCardResponse(
        model_version=card.get("model_version", get_model_version()),
        trained_at=card.get("trained_at"),
        users_trained=card.get("users_trained", 0),
        metrics=card.get("metrics", {}),
        cv_metrics=card.get("cv_metrics", {}),
        feature_columns=card.get("feature_columns", []),
    )


_shape_cache: dict = {}


@router.get("/model/explanations")
async def get_model_explanations() -> dict:
    """EBM champion's global shape functions. The model's own decision curves.

    Each feature's contribution is read directly off its shape function; there is no
    SHAP approximation. ``points`` are the log-odds contributions expressed on the
    same credit-score scale used everywhere else (positive = raises the score).
    """
    version = get_model_version()
    if _shape_cache.get("version") != version:
        try:
            champion = get_cached_champion()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="Model not trained.") from exc
        features = []
        for fn in ebm_shape_functions(champion):
            meta = FEATURE_META.get(fn["feature"], {})
            features.append(
                {
                    "feature": fn["feature"],
                    "label": meta.get("label", fn["feature"].replace("_", " ").title()),
                    "source": meta.get("source", ""),
                    "fmt": meta.get("fmt", "number"),
                    "direction": meta.get("direction", ""),
                    "x": fn["x"],
                    "points": [round(shap_to_points(v), 2) for v in fn["logodds"]],
                }
            )
        _shape_cache.update(
            {
                "version": version,
                "payload": {
                    "model": "ebm",
                    "explanation_method": "ebm-additive-terms",
                    "note": "These curves are the model itself, not a SHAP approximation.",
                    "features": features,
                },
            }
        )
    return _shape_cache["payload"]


@router.post("/train", response_model=TrainResponse, dependencies=[Depends(require_api_key)])
async def train_models() -> TrainResponse:
    """Run ECM pipeline and train CatBoost (admin endpoint)."""
    try:
        async with AsyncSessionLocal() as session:
            ecm_result = await run_ecm_pipeline(session)
            train_result = await train_all_from_db(session)
        reload_model_cache()
        metrics = train_result.get("metrics", {})
        return TrainResponse(
            users_trained=train_result["users_trained"],
            default_rate=train_result["default_rate"],
            model_path=train_result["model_path"],
            ecm_users_processed=ecm_result["users_processed"],
            model_version=train_result.get("model_version", get_model_version()),
            metrics=ModelMetrics(**metrics) if metrics else None,
            cv_metrics=train_result.get("cv_metrics", {}),
        )
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{user_id}", response_model=CreditScoreResponse)
async def get_credit_score(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_own_session),
) -> CreditScoreResponse:
    """Return alternate credit score, PD, decision, SHAP drivers, and reason codes."""
    try:
        result = await score_user(db, user_id)
        return CreditScoreResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python -m models_ai.train` first.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Scoring failed for user %s", user_id)
        raise HTTPException(status_code=500, detail="Scoring failed") from exc

from pydantic import BaseModel
import uuid
from models.db_models import AuditLog

class OverrideRequest(BaseModel):
    user_id: str
    officer_id: str
    original_decision: str
    new_decision: str
    justification: str

@router.post("/audit/override", dependencies=[Depends(require_api_key)])
async def log_decision_override(
    req: OverrideRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        user_uuid = uuid.UUID(req.user_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid user_id: {req.user_id!r}"
        ) from exc

    if not req.justification.strip():
        raise HTTPException(status_code=400, detail="Justification is required")

    try:
        log_entry = AuditLog(
            user_id=user_uuid,
            officer_id=req.officer_id,
            original_decision=req.original_decision,
            new_decision=req.new_decision,
            justification=req.justification
        )
        db.add(log_entry)
        await db.commit()
        return {"status": "success", "message": "Override logged successfully"}
    except Exception as exc:
        await db.rollback()
        logger.exception("Failed to log override")
        raise HTTPException(status_code=500, detail="Failed to log override") from exc
