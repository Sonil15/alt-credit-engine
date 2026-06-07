import secrets
from uuid import uuid4

from fastapi import APIRouter, Query

from models.pydantic_schemas import ConsentAuthorizeResponse, ConsentTokenRequest, ConsentTokenResponse

router = APIRouter(prefix="/consent", tags=["consent"])

CONSENT_SCOPES = ["telecom", "ecommerce", "geo", "cashflow", "survey"]


@router.get("/authorize", response_model=ConsentAuthorizeResponse)
async def authorize_consent(
    redirect_uri: str = Query(default="http://localhost:8000/consent/callback"),
    state: str | None = Query(default=None),
) -> ConsentAuthorizeResponse:
    """Mock OAuth 2.0 authorization endpoint."""
    auth_state = state or secrets.token_urlsafe(16)
    authorization_url = f"{redirect_uri}?code={secrets.token_urlsafe(24)}&state={auth_state}"
    return ConsentAuthorizeResponse(
        authorization_url=authorization_url,
        state=auth_state,
        scopes=CONSENT_SCOPES,
    )


@router.post("/token", response_model=ConsentTokenResponse)
async def issue_token(request: ConsentTokenRequest) -> ConsentTokenResponse:
    """Mock OAuth 2.0 token endpoint."""
    access_token = f"mock_{uuid4().hex}"
    return ConsentTokenResponse(
        access_token=access_token,
        scope=" ".join(CONSENT_SCOPES),
    )
