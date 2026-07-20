"""Authentication dependencies for protected endpoints."""

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.database import get_db


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    settings = get_settings()
    expected = settings.API_KEY.strip()
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def require_own_session(
    user_id: str,
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Allow access if caller is a bank officer (API key) OR the borrower who owns this session."""
    settings = get_settings()
    expected_key = settings.API_KEY.strip()

    # No API key configured -> keyless/demo deployment: the officer dashboard is open
    # to anyone, mirroring require_api_key's early return. Set API_KEY to re-lock
    # officer access and fall back to the per-borrower session check below.
    if not expected_key:
        return

    if expected_key and x_api_key == expected_key:
        return

    if not x_session_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    from psychometric.session import get_session
    session = get_session(x_session_id)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")


async def get_session_user_id(
    authorization: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    db: AsyncSession = Depends(get_db),
) -> str:
    """Resolve the borrower's user_id for /score/me.

    Prefers a persistent account bearer token (``Authorization: Bearer <token>``)
    so a logged-in borrower sees their own data across sessions and devices, and
    falls back to the ephemeral in-memory assessment session id for the anonymous
    apply flow.
    """
    from core.borrower_auth import resolve_bearer_user_id

    user_id = await resolve_bearer_user_id(db, authorization)
    if user_id is not None:
        return user_id

    if not x_session_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    from psychometric.session import get_session
    session = get_session(x_session_id)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return session.user_id
