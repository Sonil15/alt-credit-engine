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
    IngestionResponse,
)
from preprocessing.pipeline import process_vault_record
from psychometric.session import (
    create_session,
    get_session,
    start_response,
    submit_answer,
)

router = APIRouter(prefix="/assessment", tags=["assessment"])


@router.post("/start", response_model=AssessmentStartResponse)
async def start_assessment(request: AssessmentStartRequest) -> AssessmentStartResponse:
    """Start a multilingual agentic psychometric session."""
    session = create_session(user_id=request.user_id, language=request.language)
    payload = start_response(session)
    return AssessmentStartResponse(**payload)


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
