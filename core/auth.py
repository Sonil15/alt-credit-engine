"""API key authentication for protected endpoints."""

from fastapi import Header, HTTPException

from core.config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    settings = get_settings()
    expected = settings.API_KEY.strip()
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
