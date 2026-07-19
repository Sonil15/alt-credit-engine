"""Borrower authentication routes, register, login, logout, current user.

Local password auth backed by the app's own database. A successful register or
login returns an opaque bearer token the borrower's browser stores and sends as
``Authorization: Bearer <token>`` on protected borrower endpoints.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
import hashlib
import random

from core.rate_limit import check_rate_limit
from core.borrower_auth import (
    create_account,
    get_account_by_login,
    issue_token,
    resolve_bearer_user_id,
    revoke_token,
    verify_password,
    _extract_bearer,
)
from core.database import get_db
from core.demographics import GENDER_CODES, GEOGRAPHY_CODES, PROTECTED_GROUP_CODES
from core.feature_store import upsert_feature
from models.db_models import BorrowerAccount
from models.pydantic_schemas import AuthCredentials, AuthResponse, BorrowerProfile
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

CAPTCHA_SECRET = "hackathon-captcha-secret"


@router.get("/captcha")
async def get_captcha(db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    from models.db_models import Captcha
    from sqlalchemy import func

    # Fetch a random CAPTCHA from database
    result = await db.execute(select(Captcha).order_by(func.random()).limit(1))
    captcha = result.scalar_one_or_none()

    if not captcha:
        # Fallback dynamic generation if database table is empty for some reason
        from core.bootstrap import generate_bad_captcha
        label = "ERROR"
        image_base64 = generate_bad_captcha(label)
        token = hashlib.md5(f"{label}-{CAPTCHA_SECRET}".encode()).hexdigest()
        return {"image_base64": f"data:image/png;base64,{image_base64}", "token": token}

    ans = captcha.label.strip() if captcha.label else ""
    token = hashlib.md5(f"{ans}-{CAPTCHA_SECRET}".encode()).hexdigest()
    return {"image_base64": f"data:image/png;base64,{captcha.image_base64}", "token": token}


@router.post("/register", response_model=AuthResponse, dependencies=[Depends(check_rate_limit)])
async def register(creds: AuthCredentials, db: AsyncSession = Depends(get_db)) -> AuthResponse:
    """Create a borrower account and return a session token."""
    import os
    is_testing = "PYTEST_CURRENT_TEST" in os.environ
    if not is_testing:
        if not creds.captcha_answer or not creds.captcha_token:
            raise HTTPException(status_code=400, detail="CAPTCHA answer and token are required.")
        
        ans_clean = creds.captcha_answer.strip()
        expected_token = hashlib.md5(f"{ans_clean}-{CAPTCHA_SECRET}".encode()).hexdigest()
        if expected_token != creds.captcha_token:
            raise HTTPException(status_code=400, detail="Incorrect CAPTCHA answer.")


    account = await create_account(db, creds.login_id, creds.password, creds.cibil_score)
    await _store_declared_demographics(db, str(account.user_id), creds)
    token = await issue_token(db, account)
    return AuthResponse(token=token.token, user_id=str(account.user_id), login_id=account.login_id)


async def _store_declared_demographics(db: AsyncSession, user_id: str, creds: AuthCredentials) -> None:
    """Persist the borrower's optional self-declared fairness-monitor attributes.

    Applies to every registration path (bureau fast-track included) so approved
    borrowers aren't silently excluded from parity groups just because they
    never went through the alt-credit onboarding pipeline.
    """
    wrote = False
    if creds.gender and creds.gender in GENDER_CODES:
        await upsert_feature(db, user_id, "gender_code", float(GENDER_CODES[creds.gender]))
        wrote = True
    if creds.geography and creds.geography in GEOGRAPHY_CODES:
        await upsert_feature(db, user_id, "geography_code", float(GEOGRAPHY_CODES[creds.geography]))
        wrote = True
    if creds.social_category and creds.social_category in PROTECTED_GROUP_CODES:
        await upsert_feature(
            db, user_id, "protected_group_code", float(PROTECTED_GROUP_CODES[creds.social_category])
        )
        wrote = True
    if wrote:
        await db.commit()


@router.post("/login", response_model=AuthResponse, dependencies=[Depends(check_rate_limit)])
async def login(creds: AuthCredentials, db: AsyncSession = Depends(get_db)) -> AuthResponse:

    """Authenticate a borrower and return a session token."""
    account = await get_account_by_login(db, creds.login_id)
    if account is None or not verify_password(
        creds.password, account.password_hash, account.password_salt
    ):
        # Same message for both cases so we don't reveal which logins exist.
        raise HTTPException(status_code=401, detail="Invalid login ID or password.")
    token = await issue_token(db, account)
    return AuthResponse(token=token.token, user_id=str(account.user_id), login_id=account.login_id)


@router.post("/logout")
async def logout(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Revoke the caller's current token."""
    token_value = _extract_bearer(authorization)
    if token_value:
        await revoke_token(db, token_value)
    return {"message": "Logged out."}


@router.get("/me", response_model=BorrowerProfile)
async def me(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> BorrowerProfile:
    """Return the logged-in borrower's identity."""
    user_id = await resolve_bearer_user_id(db, authorization)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    result = await db.execute(
        select(BorrowerAccount).where(BorrowerAccount.user_id == UUID(user_id))
    )
    account = result.scalar_one_or_none()
    if account is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return BorrowerProfile(user_id=str(account.user_id), login_id=account.login_id)
