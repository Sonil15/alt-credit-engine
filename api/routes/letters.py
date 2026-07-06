"""Decision-letter endpoints: officer review queue, sign-off, and borrower retrieval.

The letter drafting itself happens at scoring time (see convergence.letter_store); these
routes expose the queue to the loan officer, let a human review and sign a
rejection/review letter, and let a borrower fetch their issued letter, asynchronously,
whenever each party next visits.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from convergence.decision_letter import render_letter
from convergence.letter_store import (
    fetch_letter,
    fetch_pending_letters,
    letter_to_dict,
    sign_letter,
)
from core.auth import get_session_user_id, require_api_key
from core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/letters", tags=["letters"])


class SignRequest(BaseModel):
    officer_id: str = "officer-dashboard"


@router.get("/pending", dependencies=[Depends(require_api_key)])
async def list_pending_letters(db: AsyncSession = Depends(get_db)) -> list[dict]:
    """Officer review queue, every decision letter awaiting human sign-off."""
    rows = await fetch_pending_letters(db)
    return [
        {
            "user_id": str(r.user_id),
            "outcome": r.outcome,
            "credit_score": r.credit_score,
            "model_version": r.model_version,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.get("/me")
async def get_my_letter(
    lang: str = "en",
    user_id: str = Depends(get_session_user_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """The authenticated borrower's own decision letter (rendered in ``lang``).

    Returns ``available: false`` while the letter is still pending officer review so
    the borrower sees a "under review" state rather than the formal notice.
    """
    letter = await fetch_letter(db, user_id)
    if letter is None:
        raise HTTPException(status_code=404, detail="No decision letter yet for this borrower.")
    rendered = render_letter(letter_to_dict(letter), lang)
    rendered["available"] = letter.status == "issued"
    return rendered


@router.get("/{user_id}", dependencies=[Depends(require_api_key)])
async def get_letter_for_officer(
    user_id: str,
    lang: str = "en",
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Officer view of a borrower's drafted or issued letter."""
    letter = await fetch_letter(db, user_id)
    if letter is None:
        raise HTTPException(status_code=404, detail="No decision letter for this borrower.")
    return render_letter(letter_to_dict(letter), lang)


@router.post("/{user_id}/sign", dependencies=[Depends(require_api_key)])
async def sign_borrower_letter(
    user_id: str,
    req: SignRequest,
    lang: str = "en",
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Loan officer reviews and signs, issues the letter to the borrower."""
    if not req.officer_id.strip():
        raise HTTPException(status_code=400, detail="Officer identity is required to sign.")
    letter = await sign_letter(db, user_id, req.officer_id.strip())
    if letter is None:
        raise HTTPException(status_code=404, detail="No decision letter for this borrower.")
    rendered = render_letter(letter_to_dict(letter), lang)
    rendered["available"] = True
    return rendered
