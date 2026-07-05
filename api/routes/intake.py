"""Borrower onboarding intake: loan intent + LLM-assisted business profile.

Runs before consent in the borrower journey. The raw free-text business
description is encrypted into the vault (data_type="intake"); only the
borrower-confirmed structured fields land in ``application_intake``.
"""

import json
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.borrower_auth import resolve_bearer_user_id
from core.business_profile import (
    BUSINESS_COHORTS,
    PURPOSES_BY_COHORT,
    extract_business_profile,
    fetch_latest_intake,
)
from core.config import get_settings
from core.database import get_db
from core.security import get_encryptor
from models.db_models import ApplicationIntake, SecureVault
from models.pydantic_schemas import (
    BusinessExtractRequest,
    BusinessExtractResponse,
    BusinessProfile,
    IntakeSubmitRequest,
    IntakeSubmitResponse,
)

router = APIRouter(prefix="/intake", tags=["intake"])


async def _require_borrower(
    db: AsyncSession,
    authorization: str | None,
) -> str:
    user_id = await resolve_bearer_user_id(db, authorization)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


@router.get("/purposes")
async def list_purposes() -> dict:
    """Purpose options per borrower category (for the onboarding form)."""
    return {"purposes_by_cohort": PURPOSES_BY_COHORT, "business_cohorts": list(BUSINESS_COHORTS)}


@router.post("/business-profile", response_model=BusinessExtractResponse)
async def analyze_business_description(
    request: BusinessExtractRequest,
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> BusinessExtractResponse:
    """Extract structured business fields for the borrower to confirm or edit."""
    await _require_borrower(db, authorization)
    profile, confidence, method = await extract_business_profile(request.text, request.language)
    return BusinessExtractResponse(
        profile=BusinessProfile(**profile),
        confidence=confidence,
        method=method,
    )


@router.post("/submit", response_model=IntakeSubmitResponse)
async def submit_intake(
    request: IntakeSubmitRequest,
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> IntakeSubmitResponse:
    """Record the borrower's loan intent (latest submission wins per borrower)."""
    session_user_id = await _require_borrower(db, authorization)
    if session_user_id != request.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    allowed = PURPOSES_BY_COHORT.get(request.cohort)
    if allowed is None:
        raise HTTPException(status_code=422, detail=f"Unknown borrower category: {request.cohort}")
    if request.loan_purpose not in allowed:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Loan purpose '{request.loan_purpose}' is not available for the "
                f"{request.cohort} category. Allowed: {', '.join(allowed)}."
            ),
        )

    # Raw description is personal data: encrypted vault only, never plaintext.
    if request.business_description and request.business_description.strip():
        encryptor = get_encryptor()
        db.add(
            SecureVault(
                user_id=UUID(request.user_id),
                data_type="intake",
                encrypted_payload=encryptor.encrypt(
                    json.dumps(
                        {
                            "business_description": request.business_description,
                            "cohort": request.cohort,
                        }
                    )
                ),
            )
        )

    profile_json: str | None = None
    if request.business_profile is not None:
        confirmed = request.business_profile.model_dump()
        if any(value is not None for value in confirmed.values()):
            profile_json = json.dumps(confirmed)

    intake = ApplicationIntake(
        user_id=UUID(request.user_id),
        cohort=request.cohort,
        loan_purpose=request.loan_purpose,
        requested_amount=float(request.requested_amount),
        business_profile_json=profile_json,
        extraction_method=request.extraction_method
        if request.extraction_method in {"llm", "fallback", "none"}
        else "none",
        extraction_confidence=request.extraction_confidence,
    )
    db.add(intake)
    await db.commit()
    await db.refresh(intake)

    return IntakeSubmitResponse(
        intake_id=str(intake.id),
        user_id=request.user_id,
        cohort=request.cohort,
        loan_purpose=request.loan_purpose,
        requested_amount=float(request.requested_amount),
    )


@router.get("/{user_id}")
async def get_latest_intake(
    user_id: str,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Latest intake for a borrower — the borrower themselves or a bank officer."""
    settings = get_settings()
    expected_key = settings.API_KEY.strip()
    if not (expected_key and x_api_key == expected_key):
        session_user_id = await resolve_bearer_user_id(db, authorization)
        if session_user_id != user_id:
            raise HTTPException(status_code=401, detail="Authentication required")

    intake = await fetch_latest_intake(db, user_id)
    if intake is None:
        raise HTTPException(status_code=404, detail="No intake found for this borrower")

    return {
        "intake_id": str(intake.id),
        "user_id": str(intake.user_id),
        "cohort": intake.cohort,
        "loan_purpose": intake.loan_purpose,
        "requested_amount": intake.requested_amount,
        "business_profile": json.loads(intake.business_profile_json)
        if intake.business_profile_json
        else None,
        "extraction_method": intake.extraction_method,
        "extraction_confidence": intake.extraction_confidence,
        "created_at": intake.created_at.isoformat() if intake.created_at else None,
    }
