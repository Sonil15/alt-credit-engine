"""Local, zero-dependency borrower authentication.

Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib ``hashlib``) and a random
per-account salt; login issues an opaque random bearer token stored server-side
in the ``auth_tokens`` table. No external auth service, no paid dependency —
everything runs against the same local SQLite/Postgres the rest of the app uses.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db  # noqa: F401  (re-exported for route convenience)
from models.db_models import AuthToken, BorrowerAccount

# PBKDF2 cost. 200k iterations is plenty for a local demo while staying instant.
_PBKDF2_ITERATIONS = 200_000
_TOKEN_TTL_DAYS = 30


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return (hex_hash, hex_salt) for a password, generating a salt if needed."""
    if salt is None:
        salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return derived.hex(), salt


def verify_password(password: str, expected_hash: str, salt: str) -> bool:
    """Constant-time check of a password against a stored hash + salt."""
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def get_account_by_login(db: AsyncSession, login_id: str) -> BorrowerAccount | None:
    result = await db.execute(
        select(BorrowerAccount).where(BorrowerAccount.login_id == login_id)
    )
    return result.scalar_one_or_none()


async def create_account(db: AsyncSession, login_id: str, password: str) -> BorrowerAccount:
    """Create a new borrower account, raising 409 if the login is taken."""
    if await get_account_by_login(db, login_id) is not None:
        raise HTTPException(status_code=409, detail="That login ID is already taken.")
    pw_hash, salt = hash_password(password)
    account = BorrowerAccount(login_id=login_id, password_hash=pw_hash, password_salt=salt)
    db.add(account)
    await db.commit()
    await db.refresh(account)
    return account


async def issue_token(db: AsyncSession, account: BorrowerAccount) -> AuthToken:
    """Mint a fresh bearer token for an account."""
    token = AuthToken(
        token=secrets.token_urlsafe(32),
        user_id=account.user_id,
        expires_at=_now() + timedelta(days=_TOKEN_TTL_DAYS),
    )
    db.add(token)
    await db.commit()
    await db.refresh(token)
    return token


async def revoke_token(db: AsyncSession, token_value: str) -> None:
    row = await db.get(AuthToken, token_value)
    if row is not None:
        await db.delete(row)
        await db.commit()


def _to_aware_utc(dt: datetime) -> datetime:
    """SQLite returns naive UTC; normalise so expiry comparison is correct."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


async def resolve_token(db: AsyncSession, token_value: str) -> str | None:
    """Return the account's user_id for a valid, unexpired token, else None."""
    row = await db.get(AuthToken, token_value)
    if row is None:
        return None
    if _to_aware_utc(row.expires_at) < _now():
        await db.delete(row)
        await db.commit()
        return None
    return str(row.user_id)


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip() or None


async def resolve_bearer_user_id(
    db: AsyncSession, authorization: str | None
) -> str | None:
    """Resolve an ``Authorization: Bearer <token>`` header to a user_id, or None."""
    token_value = _extract_bearer(authorization)
    if not token_value:
        return None
    return await resolve_token(db, token_value)
