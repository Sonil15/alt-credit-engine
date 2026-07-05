"""Persistence for borrower decision letters and their sign-off state.

One active ``DecisionLetter`` per borrower, upserted whenever a decision is persisted:
an APPROVE outcome is issued automatically; a REJECT/REVIEW outcome is drafted as
``pending_review`` for a loan officer to review and sign. A letter already finalised
for the *same* outcome is left untouched, so a borrower re-scoring their unchanged
result never resets a signed letter or spams the officer queue with duplicates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.db_models import DecisionLetter

PENDING = "pending_review"
ISSUED = "issued"


def _target_status(outcome: str) -> str:
    """APPROVE issues automatically; REJECT/REVIEW need an officer signature."""
    return ISSUED if (outcome or "").upper() == "APPROVE" else PENDING


def letter_to_dict(row: DecisionLetter) -> dict[str, Any]:
    return {
        "user_id": str(row.user_id),
        "status": row.status,
        "outcome": row.outcome,
        "credit_score": row.credit_score,
        "model_version": row.model_version,
        "reason_codes": json.loads(row.reason_codes_json) if row.reason_codes_json else [],
        "officer_id": row.officer_id,
        "signed_at": row.signed_at,
        "created_at": row.created_at,
    }


async def upsert_letter_for_decision(session: AsyncSession, payload: dict[str, Any]) -> None:
    """Create or refresh the borrower's letter from a scoring payload.

    Does not commit — the caller commits alongside the ScoreDecision write so the
    audit row and the letter persist together.
    """
    user_id = UUID(str(payload["user_id"]))
    outcome = str(payload.get("final_outcome") or payload.get("decision") or "REVIEW")
    target = _target_status(outcome)
    reasons_json = json.dumps(payload.get("reason_codes", []))

    existing = (
        await session.execute(select(DecisionLetter).where(DecisionLetter.user_id == user_id))
    ).scalar_one_or_none()

    if existing is None:
        session.add(
            DecisionLetter(
                user_id=user_id,
                status=target,
                outcome=outcome,
                credit_score=int(payload.get("credit_score", 0)),
                model_version=str(payload.get("model_version", "unknown")),
                reason_codes_json=reasons_json,
            )
        )
        return

    # Already finalised for this same outcome — leave the signed/auto-issued letter alone.
    if existing.status == ISSUED and existing.outcome == outcome:
        return

    # A materially different (or still-pending) decision → refresh the draft. A change
    # of outcome re-opens review, so drop any prior signature.
    existing.outcome = outcome
    existing.credit_score = int(payload.get("credit_score", 0))
    existing.model_version = str(payload.get("model_version", "unknown"))
    existing.reason_codes_json = reasons_json
    existing.status = target
    if target == PENDING:
        existing.officer_id = None
        existing.signed_at = None


async def fetch_letter(session: AsyncSession, user_id: str) -> DecisionLetter | None:
    try:
        uid = UUID(str(user_id))
    except (ValueError, AttributeError, TypeError):
        return None
    return (
        await session.execute(select(DecisionLetter).where(DecisionLetter.user_id == uid))
    ).scalar_one_or_none()


async def fetch_pending_letters(session: AsyncSession) -> list[DecisionLetter]:
    rows = (
        await session.execute(
            select(DecisionLetter)
            .where(DecisionLetter.status == PENDING)
            .order_by(DecisionLetter.created_at)
        )
    ).scalars().all()
    return list(rows)


async def sign_letter(session: AsyncSession, user_id: str, officer_id: str) -> DecisionLetter | None:
    """Officer sign-off: stamp identity + timestamp and issue the letter to the borrower."""
    letter = await fetch_letter(session, user_id)
    if letter is None:
        return None
    letter.status = ISSUED
    letter.officer_id = officer_id
    letter.signed_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(letter)
    return letter
