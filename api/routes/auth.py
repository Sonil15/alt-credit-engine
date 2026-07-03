"""Borrower authentication routes — register, login, logout, current user.

Local password auth backed by the app's own database. A successful register or
login returns an opaque bearer token the borrower's browser stores and sends as
``Authorization: Bearer <token>`` on protected borrower endpoints.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

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
from models.db_models import BorrowerAccount
from models.pydantic_schemas import AuthCredentials, AuthResponse, BorrowerProfile
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
async def register(creds: AuthCredentials, db: AsyncSession = Depends(get_db)) -> AuthResponse:
    """Create a borrower account and return a session token."""
    account = await create_account(db, creds.login_id, creds.password)
    token = await issue_token(db, account)
    return AuthResponse(token=token.token, user_id=str(account.user_id), login_id=account.login_id)


@router.post("/login", response_model=AuthResponse)
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
