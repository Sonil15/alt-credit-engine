"""Authentication dependencies for protected endpoints."""

from fastapi import Header, HTTPException

from core.config import get_settings


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
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> str:
    """Resolve the user_id from a borrower session token. Used for /score/me."""
    if not x_session_id:
        raise HTTPException(status_code=401, detail="X-Session-Id header required")

    from psychometric.session import get_session
    session = get_session(x_session_id)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return session.user_id
