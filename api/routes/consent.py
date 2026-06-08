import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from models.pydantic_schemas import (
    ConsentAuthorizeResponse,
    ConsentRevokeRequest,
    ConsentRevokeResponse,
    ConsentTokenRequest,
    ConsentTokenResponse,
    LiveLocationRequest,
    LiveLocationResponse,
)
from preprocessing.clean_geo import latlong_to_pincode

router = APIRouter(prefix="/consent", tags=["consent"])
geo_router = APIRouter(prefix="/api", tags=["geo"])

MOCK_DATA_PATH = Path(__file__).resolve().parents[2] / "synthetic_data" / "mock_data_100_users.json"
_mock_profiles_cache: list[dict] | None = None
_active_consents: dict[str, dict] = {}
_revoked_consents: set[str] = set()

CONSENT_SCOPES = ["telecom", "ecommerce", "geo", "cashflow", "survey"]
DATA_FIDUCIARY = "UCO Bank — Alt-Credit Engine (Demo AA)"
CONSENT_PURPOSE = "Alternate creditworthiness assessment for thin-file loan origination"
CONSENT_TTL_HOURS = 24


def _consent_expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=CONSENT_TTL_HOURS)).isoformat()


@router.get("/authorize", response_model=ConsentAuthorizeResponse)
async def authorize_consent(
    redirect_uri: str = Query(default="http://localhost:8000/consent/callback"),
    state: str | None = Query(default=None),
) -> ConsentAuthorizeResponse:
    """RBI Account Aggregator-style consent authorization endpoint (demo)."""
    auth_state = state or secrets.token_urlsafe(16)
    consent_id = f"AA-CONSENT-{uuid4().hex[:12].upper()}"
    authorization_url = (
        f"{redirect_uri}?code={secrets.token_urlsafe(24)}&state={auth_state}&consent_id={consent_id}"
    )

    _active_consents[consent_id] = {
        "scopes": CONSENT_SCOPES,
        "purpose": CONSENT_PURPOSE,
        "data_fiduciary": DATA_FIDUCIARY,
        "expires_at": _consent_expiry(),
        "status": "active",
    }

    return ConsentAuthorizeResponse(
        authorization_url=authorization_url,
        state=auth_state,
        scopes=CONSENT_SCOPES,
        consent_id=consent_id,
        purpose=CONSENT_PURPOSE,
        data_fiduciary=DATA_FIDUCIARY,
        expires_at=_consent_expiry(),
    )


@router.post("/token", response_model=ConsentTokenResponse)
async def issue_token(request: ConsentTokenRequest) -> ConsentTokenResponse:
    """RBI Account Aggregator-style token exchange (demo)."""
    consent_id = request.consent_id or f"AA-CONSENT-{uuid4().hex[:12].upper()}"
    if consent_id in _revoked_consents:
        raise HTTPException(status_code=403, detail="Consent has been revoked by the data principal")

    access_token = f"aa_{uuid4().hex}"
    return ConsentTokenResponse(
        access_token=access_token,
        scope=" ".join(CONSENT_SCOPES),
        consent_id=consent_id,
        purpose=CONSENT_PURPOSE,
    )


@router.post("/revoke", response_model=ConsentRevokeResponse)
async def revoke_consent(request: ConsentRevokeRequest) -> ConsentRevokeResponse:
    """Allow data principal to revoke AA consent (DPDP right to withdraw consent)."""
    _revoked_consents.add(request.consent_id)
    if request.consent_id in _active_consents:
        _active_consents[request.consent_id]["status"] = "revoked"
    return ConsentRevokeResponse(consent_id=request.consent_id)


@router.get("/compliance")
async def compliance_summary() -> dict:
    """Regulatory compliance summary for bank stakeholders."""
    return {
        "frameworks": [
            "RBI Account Aggregator Framework (consent artifact with purpose, expiry, revocation)",
            "DPDP Act 2023 (data minimization, consent, encryption at rest, right to erasure)",
            "RBI Digital Lending Guidelines 2022 (Key Fact Statement, cooling-off period)",
        ],
        "data_localization": "Production deployment targets India-region storage (Mumbai) with AES-256-GCM encryption.",
        "privacy": "Raw PII encrypted in secure_vault; models consume tokenized ml_features only.",
        "documentation": "/docs/COMPLIANCE.md",
    }


def _load_mock_profiles() -> list[dict]:
    global _mock_profiles_cache
    if _mock_profiles_cache is None:
        if not MOCK_DATA_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Mock data not found at {MOCK_DATA_PATH}. Run generate_raw_mock.py first.",
            )
        import json

        _mock_profiles_cache = json.loads(MOCK_DATA_PATH.read_text(encoding="utf-8"))
    return _mock_profiles_cache


def _load_most_frequent_pin(user_id: str) -> str | None:
    from collections import Counter

    profiles = _load_mock_profiles()
    profile = next((p for p in profiles if p.get("user_id") == user_id), None)
    if profile is None:
        return None

    pins = [
        order.get("delivery_pin_code")
        for order in profile.get("ecommerce", {}).get("orders", [])
        if order.get("delivery_pin_code")
    ]
    if not pins:
        return None

    return Counter(pins).most_common(1)[0][0]


@geo_router.post("/verify-live-location", response_model=LiveLocationResponse)
async def verify_live_location(req: LiveLocationRequest) -> LiveLocationResponse:
    """Compare HTML5 live check-in coordinates against the user's most frequent e-commerce pin."""
    most_frequent = _load_most_frequent_pin(req.user_id)
    if most_frequent is None:
        raise HTTPException(
            status_code=404,
            detail=f"No delivery pin history found for user {req.user_id}",
        )

    submitted = latlong_to_pincode(req.lat, req.long)
    match = submitted == most_frequent
    verdict = "verified" if match else "location_mismatch"

    return LiveLocationResponse(
        user_id=req.user_id,
        submitted_pin=submitted,
        expected_pin=most_frequent,
        match=match,
        verdict=verdict,
    )
