import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from convergence.score_engine import portfolio_summary, score_all_users, score_user
from core.auth import get_session_user_id, require_api_key, require_own_session
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


from pydantic import BaseModel, Field
from typing import Any
from uuid import UUID
import uuid

class SimulationRequest(BaseModel):
    user_id: str | None = None
    theta: float = Field(ge=0.0, le=1.0, default=0.5)
    cohort: str | None = None
    feature_overrides: dict[str, float] | None = None

class SimulationResponse(BaseModel):
    score: CreditScoreResponse
    raw_data: dict[str, Any]


@router.post("/simulate", response_model=SimulationResponse)
async def simulate_scoring(
    req: SimulationRequest,
    db: AsyncSession = Depends(get_db),
) -> SimulationResponse:
    """Generate mock data in-memory based on theta, run cleaners, write to ml_features (marked as simulated), and score user."""
    # 1. Determine user ID
    user_id_str = req.user_id or str(uuid.uuid4())
    user_uuid = UUID(user_id_str)

    # 2. Clear old MLFeature and FeatureSeries rows for this user to start fresh
    from sqlalchemy import delete as sa_delete
    from models.db_models import MLFeature, FeatureSeries, ScoreDecision, SecureVault
    await db.execute(sa_delete(MLFeature).where(MLFeature.user_id == user_uuid))
    await db.execute(sa_delete(FeatureSeries).where(FeatureSeries.user_id == user_uuid))
    await db.execute(sa_delete(ScoreDecision).where(ScoreDecision.user_id == user_uuid))
    await db.execute(sa_delete(SecureVault).where(SecureVault.user_id == user_uuid))
    await db.commit()

    # 3. Generate mock records in-memory (bypassing secure_vault completely)
    from synthetic_data.generate_raw_mock import (
        generate_telecom_invoices,
        generate_ecommerce_orders,
        generate_geo_locations,
        generate_cashflow_transactions,
        generate_survey,
    )
    telecom_raw = generate_telecom_invoices(user_id_str, req.theta)
    ecommerce_raw = generate_ecommerce_orders(user_id_str, req.theta)
    geo_raw = generate_geo_locations(user_id_str, req.theta)
    cashflow_raw = generate_cashflow_transactions(user_id_str, req.theta)

    cohort = req.cohort or "Salaried"
    cohort_codes = {
        "Salaried": 0.0,
        "GigWorker": 1.0,
        "Student": 2.0,
        "Vendor": 3.0,
        "Farmer": 4.0,
        "Homemaker": 5.0,
    }
    cohort_code = cohort_codes.get(cohort, 0.0)
    extra_features = {
        "cohort_code": cohort_code,
    }

    if cohort == "Student":
        extra_features["upi_spend_consistency"] = round(0.5 + req.theta * 0.5, 2)
        extra_features["small_dues_payment_promptness"] = round(0.6 + req.theta * 0.4, 2)
        extra_features["e_wallet_topup_frequency"] = round(0.3 + req.theta * 0.7, 2)
    elif cohort == "Vendor":
        extra_features["daily_transaction_count"] = round(10.0 + req.theta * 50.0, 2)
        extra_features["average_ticket_size"] = round(50.0 + req.theta * 450.0, 2)
    elif cohort == "Farmer":
        extra_features["harvest_income_spike"] = round(1.0 + req.theta * 9.0, 2)
        extra_features["input_purchase_consistency"] = round(0.5 + req.theta * 0.5, 2)
    elif cohort == "Homemaker":
        extra_features["utility_payment_consistency"] = round(0.5 + req.theta * 0.5, 2)
        extra_features["grocery_spend_stability"] = round(0.5 + req.theta * 0.5, 2)

    survey_raw = generate_survey(user_id_str, req.theta, extra_features)

    # 4. Extract features directly in-memory using cleaners
    from preprocessing.clean_telecom import clean_telecom
    from preprocessing.clean_ecommerce import clean_ecommerce
    from preprocessing.clean_geo import clean_geo
    from preprocessing.clean_cashflow import clean_cashflow
    from preprocessing.clean_survey import clean_survey
    
    telecom_feats, telecom_series = clean_telecom(telecom_raw)
    ecommerce_feats = clean_ecommerce(ecommerce_raw)
    geo_feats = clean_geo(geo_raw)
    cashflow_feats, cashflow_series = clean_cashflow(cashflow_raw)
    survey_feats = await clean_survey(survey_raw)

    # Combine all mock features and mark user as simulated
    sim_features = {}
    sim_features.update(telecom_feats)
    sim_features.update(ecommerce_feats)
    sim_features.update(geo_feats)
    sim_features.update(cashflow_feats)
    sim_features.update(survey_feats)
    sim_features["is_simulated"] = 1.0

    # 5. Upsert features and series to database
    from core.feature_store import upsert_feature, upsert_series
    for f_name, f_val in sim_features.items():
        await upsert_feature(db, user_uuid, f_name, f_val)

    if telecom_series:
        await upsert_series(db, user_uuid, "telecom_payment_rate", telecom_series)
    if cashflow_series:
        await upsert_series(db, user_uuid, "monthly_net_cashflow", cashflow_series)
    await db.commit()

    # 6. Run econometric pipeline to calculate Cashflow Resilience features
    await run_ecm_pipeline(db)

    # 7. Apply feature overrides if provided
    if req.feature_overrides:
        for f_name, val in req.feature_overrides.items():
            await upsert_feature(db, user_uuid, f_name, val)
        await db.commit()

    # 8. Score user
    result = await score_user(db, user_id_str, persist=True)

    # Assemble response
    raw_data = {
        "telecom": {"user_id": user_id_str, "invoices": telecom_raw},
        "ecommerce": {"user_id": user_id_str, "orders": ecommerce_raw},
        "geo": {"user_id": user_id_str, "locations": geo_raw},
        "cashflow": {"user_id": user_id_str, "transactions": cashflow_raw},
        "survey": {"user_id": user_id_str, **survey_raw},
    }

    return SimulationResponse(
        score=CreditScoreResponse(**result),
        raw_data=raw_data,
    )
