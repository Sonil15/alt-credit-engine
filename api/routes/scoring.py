import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from convergence.score_engine import score_all_users, score_user
from core.database import AsyncSessionLocal, get_db
from models.pydantic_schemas import CreditScoreResponse, TrainResponse
from models_ai.catboost_model import train_from_db
from models_econometric.ecm_model import run_ecm_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/score", tags=["scoring"])


@router.get("/{user_id}", response_model=CreditScoreResponse)
async def get_credit_score(
    user_id: str,
    db: AsyncSession = Depends(get_db),
) -> CreditScoreResponse:
    """Return alternate credit score, PD, decision, and SHAP drivers for a user."""
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


@router.post("/train", response_model=TrainResponse)
async def train_models() -> TrainResponse:
    """Run ECM pipeline and train CatBoost (admin endpoint)."""
    try:
        async with AsyncSessionLocal() as session:
            ecm_result = await run_ecm_pipeline(session)
            train_result = await train_from_db(session)
        return TrainResponse(
            users_trained=train_result["users_trained"],
            default_rate=train_result["default_rate"],
            model_path=train_result["model_path"],
            ecm_users_processed=ecm_result["users_processed"],
        )
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
