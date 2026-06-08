import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from convergence.score_engine import portfolio_summary, score_all_users, score_user
from core.auth import require_api_key
from core.database import AsyncSessionLocal, get_db
from core.model_cache import get_model_card, get_model_version, reload_model_cache
from models.pydantic_schemas import (
    CreditScoreResponse,
    ModelCardResponse,
    ModelMetrics,
    PortfolioSummaryResponse,
    TrainResponse,
)
from models_ai.catboost_model import train_from_db
from models_econometric.ecm_model import run_ecm_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/score", tags=["scoring"])


@router.get("/", response_model=list[CreditScoreResponse])
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


@router.get("/portfolio/summary", response_model=PortfolioSummaryResponse)
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


@router.post("/train", response_model=TrainResponse, dependencies=[Depends(require_api_key)])
async def train_models() -> TrainResponse:
    """Run ECM pipeline and train CatBoost (admin endpoint)."""
    try:
        async with AsyncSessionLocal() as session:
            ecm_result = await run_ecm_pipeline(session)
            train_result = await train_from_db(session)
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
