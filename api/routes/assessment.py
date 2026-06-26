import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_encryptor
from models.db_models import SecureVault
from models.pydantic_schemas import (
    AssessmentAnswerRequest,
    AssessmentAnswerResponse,
    AssessmentStartRequest,
    AssessmentStartResponse,
    AssessmentTimeoutRequest,
    AssessmentBeginRequest,
    IngestionResponse,
)
from preprocessing.pipeline import process_vault_record
from psychometric.session import (
    create_session,
    get_session,
    start_response,
    submit_answer,
    force_timeout_session,
    extend_session,
    begin_timer,
)

router = APIRouter(prefix="/assessment", tags=["assessment"])


@router.post("/start", response_model=AssessmentStartResponse)
async def start_assessment(request: AssessmentStartRequest) -> AssessmentStartResponse:
    """Start a multilingual agentic psychometric session."""
    session = create_session(user_id=request.user_id, language=request.language, cohort=request.cohort)
    payload = start_response(session)
    from core.config import get_settings
    settings = get_settings()
    payload["time_limit_seconds"] = settings.PSYCHOMETRIC_TIME_LIMIT_SECONDS
    payload["extension_seconds"] = settings.PSYCHOMETRIC_EXTENSION_SECONDS
    return AssessmentStartResponse(**payload)


@router.post("/begin", response_model=AssessmentAnswerResponse)
async def begin_assessment_timer(request: AssessmentBeginRequest) -> AssessmentAnswerResponse:
    """Start the assessment timer and fetch the first item."""
    try:
        result = begin_timer(request.session_id)
        return AssessmentAnswerResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/answer", response_model=AssessmentAnswerResponse)
async def answer_assessment(
    request: AssessmentAnswerRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> AssessmentAnswerResponse:
    """Submit an answer; agent may clarify or serve the next item."""
    try:
        result = await submit_answer(request.session_id, request.item_id, request.answer)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if result.get("completed") and result.get("survey_payload"):
        await _ingest_assessment(result["survey_payload"], background_tasks, db)

    return AssessmentAnswerResponse(**result)


async def _ingest_assessment(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
    db: AsyncSession,
) -> IngestionResponse:
    """Encrypt full assessment transcript and schedule feature extraction."""
    user_id = UUID(str(payload["user_id"]))
    encryptor = get_encryptor()
    encrypted = encryptor.encrypt(json.dumps(payload, default=str))

    vault_record = SecureVault(
        user_id=user_id,
        data_type="survey",
        encrypted_payload=encrypted,
    )
    db.add(vault_record)
    await db.commit()
    await db.refresh(vault_record)

    from core.config import get_settings
    if get_settings().simulate_all_pillars_enabled:
        # Process the survey synchronously to prevent redirect race conditions
        await process_vault_record(vault_record.id)

        # Clear any existing MLFeature and FeatureSeries rows for this user to start fresh
        from sqlalchemy import delete as sa_delete
        from models.db_models import MLFeature, FeatureSeries
        await db.execute(sa_delete(MLFeature).where(MLFeature.user_id == user_id).where(MLFeature.feature_name != "is_simulated"))
        await db.execute(sa_delete(FeatureSeries).where(FeatureSeries.user_id == user_id))
        await db.commit()

        # Estimate latent creditworthiness (theta) from survey traits
        conscientiousness = float(payload.get("conscientiousness", 0.5))
        locus_of_control = float(payload.get("locus_of_control", 0.5))
        financial_self_efficacy = float(payload.get("financial_self_efficacy", 0.5))
        present_bias = float(payload.get("present_bias", 0.5))
        debt_attitude = float(payload.get("debt_attitude", 0.5))
        theta = (conscientiousness + locus_of_control + financial_self_efficacy + (1.0 - present_bias) + debt_attitude) / 5.0

        # Generate mock records in-memory (bypassing secure_vault for safety)
        from synthetic_data.generate_raw_mock import (
            generate_telecom_invoices,
            generate_ecommerce_orders,
            generate_geo_locations,
            generate_cashflow_transactions,
        )
        telecom_raw = generate_telecom_invoices(str(user_id), theta)
        ecommerce_raw = generate_ecommerce_orders(str(user_id), theta)
        geo_raw = generate_geo_locations(str(user_id), theta)
        cashflow_raw = generate_cashflow_transactions(str(user_id), theta)

        # Extract features directly in-memory using cleaners
        from preprocessing.clean_telecom import clean_telecom
        from preprocessing.clean_ecommerce import clean_ecommerce
        from preprocessing.clean_geo import clean_geo
        from preprocessing.clean_cashflow import clean_cashflow
        
        telecom_feats, telecom_series = clean_telecom(telecom_raw)
        ecommerce_feats = clean_ecommerce(ecommerce_raw)
        geo_feats = clean_geo(geo_raw)
        cashflow_feats, cashflow_series = clean_cashflow(cashflow_raw)

        # Combine all mock features and mark user as simulated
        sim_features = {}
        sim_features.update(telecom_feats)
        sim_features.update(ecommerce_feats)
        sim_features.update(geo_feats)
        sim_features.update(cashflow_feats)
        sim_features["is_simulated"] = 1.0

        cohort = payload.get("cohort", "Salaried")
        cohort_codes = {
            "Salaried": 0.0,
            "GigWorker": 1.0,
            "Student": 2.0,
            "Vendor": 3.0,
            "Farmer": 4.0,
            "Homemaker": 5.0,
        }
        sim_features["cohort_code"] = cohort_codes.get(cohort, 0.0)

        if cohort == "Student":
            sim_features["upi_spend_consistency"] = round(0.5 + theta * 0.5, 2)
            sim_features["small_dues_payment_promptness"] = round(0.6 + theta * 0.4, 2)
            sim_features["e_wallet_topup_frequency"] = round(0.3 + theta * 0.7, 2)
        elif cohort == "Vendor":
            sim_features["daily_transaction_count"] = round(10.0 + theta * 50.0, 2)
            sim_features["average_ticket_size"] = round(50.0 + theta * 450.0, 2)
        elif cohort == "Farmer":
            sim_features["harvest_income_spike"] = round(1.0 + theta * 9.0, 2)
            sim_features["input_purchase_consistency"] = round(0.5 + theta * 0.5, 2)
        elif cohort == "Homemaker":
            sim_features["utility_payment_consistency"] = round(0.5 + theta * 0.5, 2)
            sim_features["grocery_spend_stability"] = round(0.5 + theta * 0.5, 2)

        # Upsert features and series to database
        from core.feature_store import upsert_feature, upsert_series
        for f_name, f_val in sim_features.items():
            await upsert_feature(db, user_id, f_name, f_val)

        if telecom_series:
            await upsert_series(db, user_id, "telecom_payment_rate", telecom_series)
        if cashflow_series:
            await upsert_series(db, user_id, "monthly_net_cashflow", cashflow_series)
        await db.commit()

        # Run econometric pipeline to calculate Cashflow Resilience features
        from models_econometric.ecm_model import run_ecm_pipeline
        await run_ecm_pipeline(db)

    else:
        background_tasks.add_task(process_vault_record, vault_record.id)

    return IngestionResponse(
        vault_id=vault_record.id,
        user_id=user_id,
        data_type="survey",
        message="Assessment encrypted and stored. Trait extraction scheduled.",
    )


@router.get("/session/{session_id}")
async def get_assessment_session(session_id: str) -> dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "language": session.language,
        "progress": session.progress,
        "completed": session.completed,
        "traits": session.traits,
        "transcript_length": len(session.transcript),
    }


@router.post("/timeout", response_model=AssessmentAnswerResponse)
async def timeout_assessment(
    request: AssessmentTimeoutRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> AssessmentAnswerResponse:
    """Forcibly finalize the assessment session when client timer expires."""
    try:
        result = await force_timeout_session(request.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if result.get("completed") and result.get("survey_payload"):
        await _ingest_assessment(result["survey_payload"], background_tasks, db)

    return AssessmentAnswerResponse(**result)


@router.post("/extend")
async def extend_assessment_timer(request: AssessmentTimeoutRequest) -> dict[str, Any]:
    """Mark the assessment session as extended on the backend."""
    try:
        result = extend_session(request.session_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
