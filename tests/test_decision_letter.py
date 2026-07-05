"""Decision-letter rendering + sign-off state machine.

Covers: deterministic multilingual rendering (reasons translated), the auto-issue vs
pending-review split, idempotent re-scoring, and officer sign-off issuing the letter.
"""

import uuid

import pytest
from sqlalchemy import delete

from convergence.decision_letter import render_letter
from convergence.letter_store import (
    ISSUED,
    PENDING,
    fetch_letter,
    sign_letter,
    upsert_letter_for_decision,
)
from core.database import AsyncSessionLocal
from models.db_models import DecisionLetter


def _reject_letter(reasons=None):
    return {
        "user_id": str(uuid.uuid4()),
        "status": PENDING,
        "outcome": "REJECT",
        "credit_score": 420,
        "model_version": "ebm-champion-v3",
        "reason_codes": reasons if reasons is not None else ["High spending volatility", "Low or unstable income"],
    }


def test_render_translates_reasons_and_title():
    en = render_letter(_reject_letter(), "en")
    hi = render_letter(_reject_letter(), "hi")
    bn = render_letter(_reject_letter(), "bn")
    assert en["letter"]["reasons"][0] == "High spending volatility"
    assert hi["letter"]["reasons"][0] == "उच्च खर्च अस्थिरता"
    assert bn["letter"]["reasons"][0] == "উচ্চ ব্যয় অস্থিরতা"
    # Title differs by language and reflects a rejection.
    assert en["letter"]["title"] != hi["letter"]["title"]
    assert en["letter"]["outcome_text"] == "Application declined"


def test_unknown_reason_falls_back_to_english():
    letter = _reject_letter(reasons=["Thin-file applicant: limited alternative data (telecom)"])
    hi = render_letter(letter, "hi")
    assert hi["letter"]["reasons"][0].startswith("Thin-file applicant")


def test_plain_text_contains_reasons_and_pending_note():
    rendered = render_letter(_reject_letter(), "en")
    assert "PRINCIPAL REASONS" in rendered["plain_text"].upper() or "Principal reasons" in rendered["plain_text"]
    assert "High spending volatility" in rendered["plain_text"]
    # Pending letters carry the preliminary-result note, not a signature line.
    assert rendered["letter"].get("status_note")


def test_approve_letter_kind():
    letter = _reject_letter()
    letter["outcome"] = "APPROVE"
    rendered = render_letter(letter, "en")
    assert rendered["letter"]["outcome_text"] == "Application approved"


@pytest.mark.asyncio
async def test_upsert_and_sign_flow():
    uid = str(uuid.uuid4())
    reject_payload = {
        "user_id": uid,
        "decision": "REJECT",
        "final_outcome": "REJECT",
        "credit_score": 410,
        "model_version": "ebm-champion-v3",
        "reason_codes": ["Low or unstable income"],
    }
    try:
        async with AsyncSessionLocal() as s:
            await upsert_letter_for_decision(s, reject_payload)
            await s.commit()
        async with AsyncSessionLocal() as s:
            letter = await fetch_letter(s, uid)
            assert letter is not None and letter.status == PENDING and letter.officer_id is None

        # Re-scoring the same unchanged outcome must not duplicate or reset the draft.
        async with AsyncSessionLocal() as s:
            await upsert_letter_for_decision(s, reject_payload)
            await s.commit()
        async with AsyncSessionLocal() as s:
            rows = (await s.execute(
                delete(DecisionLetter).where(DecisionLetter.user_id == uuid.UUID(uid)).returning(DecisionLetter.id)
            )).fetchall()
            # exactly one row existed before deletion
            assert len(rows) == 1
            await s.commit()

        # Recreate and sign.
        async with AsyncSessionLocal() as s:
            await upsert_letter_for_decision(s, reject_payload)
            await s.commit()
        async with AsyncSessionLocal() as s:
            signed = await sign_letter(s, uid, "officer-7")
            assert signed.status == ISSUED
            assert signed.officer_id == "officer-7"
            assert signed.signed_at is not None
    finally:
        async with AsyncSessionLocal() as s:
            await s.execute(delete(DecisionLetter).where(DecisionLetter.user_id == uuid.UUID(uid)))
            await s.commit()


@pytest.mark.asyncio
async def test_approve_auto_issues_without_officer():
    uid = str(uuid.uuid4())
    approve_payload = {
        "user_id": uid,
        "decision": "APPROVE",
        "final_outcome": "APPROVE",
        "credit_score": 720,
        "model_version": "ebm-champion-v3",
        "reason_codes": [],
    }
    try:
        async with AsyncSessionLocal() as s:
            await upsert_letter_for_decision(s, approve_payload)
            await s.commit()
        async with AsyncSessionLocal() as s:
            letter = await fetch_letter(s, uid)
            assert letter.status == ISSUED and letter.officer_id is None
    finally:
        async with AsyncSessionLocal() as s:
            await s.execute(delete(DecisionLetter).where(DecisionLetter.user_id == uuid.UUID(uid)))
            await s.commit()
