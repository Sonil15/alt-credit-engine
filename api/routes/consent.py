import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete as sa_delete, func, select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models.db_models import FeatureSeries, MLFeature, ScoreDecision, SecureVault
from models.pydantic_schemas import (
    ConsentAuthorizeResponse,
    ConsentRevokeRequest,
    ConsentRevokeResponse,
    ConsentStatusResponse,
    ConsentTokenRequest,
    ConsentTokenResponse,
    ErasureRequest,
    ErasureResponse,
    LiveLocationRequest,
    LiveLocationResponse,
)
from preprocessing.clean_geo import latlong_to_pincode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consent", tags=["consent"])
geo_router = APIRouter(prefix="/api", tags=["geo"])

MOCK_DATA_PATH = Path(__file__).resolve().parents[2] / "synthetic_data" / "mock_data_100_users.json"
_mock_profiles_cache: list[dict] | None = None

# In-memory state (demo: production uses a database)
_active_consents: dict[str, dict] = {}       # consent_id -> consent record
_revoked_consents: set[str] = set()           # consent_id
_revoked_users: set[str] = set()              # user_id (for user-id-based revoke)
_user_consent_map: dict[str, str] = {}        # user_id -> consent_id
_erasure_requests: dict[str, dict] = {}       # user_id -> erasure record

CONSENT_SCOPES = ["telecom", "ecommerce", "geo", "cashflow", "survey", "upi_lite", "dbt_logs", "sms_parsing", "enam_receipts"]
DATA_FIDUCIARY = "Alt-Credit Engine (Demo AA)"
CONSENT_PURPOSE = "Alternate creditworthiness assessment for thin-file loan origination"
CONSENT_TTL_HOURS = 24


def _consent_expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=CONSENT_TTL_HOURS)).isoformat()


@router.get("/authorize", response_model=ConsentAuthorizeResponse)
async def authorize_consent(
    redirect_uri: str = Query(default="http://localhost:8000/consent/callback"),
    state: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
) -> ConsentAuthorizeResponse:
    """RBI Account Aggregator-style consent authorization endpoint (demo)."""
    auth_state = state or secrets.token_urlsafe(16)
    consent_id = f"AA-CONSENT-{uuid4().hex[:12].upper()}"
    authorization_url = (
        f"{redirect_uri}?code={secrets.token_urlsafe(24)}&state={auth_state}&consent_id={consent_id}"
    )

    _active_consents[consent_id] = {
        "scopes": CONSENT_SCOPES.copy(),
        "purpose": CONSENT_PURPOSE,
        "data_fiduciary": DATA_FIDUCIARY,
        "expires_at": _consent_expiry(),
        "status": "active",
        "user_id": user_id,
        "revoked_scopes": [],
    }

    if user_id:
        _user_consent_map[user_id] = consent_id

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
    """Allow data principal to revoke AA consent (DPDP right to withdraw consent).
    Accepts either consent_id or user_id."""
    if not request.consent_id and not request.user_id:
        raise HTTPException(status_code=400, detail="Provide consent_id or user_id")

    now = datetime.now(timezone.utc).isoformat()

    # Resolve consent_id from user_id if only user_id was provided
    consent_id = request.consent_id
    user_id = request.user_id
    if not consent_id and user_id:
        consent_id = _user_consent_map.get(user_id, f"implied-{user_id[:8]}")
    if not user_id and consent_id:
        # Try reverse-lookup
        record = _active_consents.get(consent_id, {})
        user_id = record.get("user_id")

    if request.scopes:
        if consent_id not in _active_consents:
            _active_consents[consent_id] = {
                "scopes": CONSENT_SCOPES.copy(),
                "purpose": CONSENT_PURPOSE,
                "data_fiduciary": DATA_FIDUCIARY,
                "expires_at": _consent_expiry(),
                "status": "active",
                "user_id": user_id,
                "revoked_scopes": [],
            }
        record = _active_consents[consent_id]
        if "revoked_scopes" not in record:
            record["revoked_scopes"] = []
        if "scopes" not in record:
            record["scopes"] = CONSENT_SCOPES.copy()

        for sc in request.scopes:
            if sc in record["scopes"]:
                record["scopes"].remove(sc)
            if sc not in record["revoked_scopes"]:
                record["revoked_scopes"].append(sc)

        if not record["scopes"]:
            _revoked_consents.add(consent_id)
            record["status"] = "revoked"
            if user_id:
                _revoked_users.add(user_id)
    else:
        _revoked_consents.add(consent_id)
        if consent_id in _active_consents:
            record = _active_consents[consent_id]
            record["status"] = "revoked"
            record["revoked_scopes"] = CONSENT_SCOPES.copy()
            record["scopes"] = []
        else:
            _active_consents[consent_id] = {
                "scopes": [],
                "purpose": CONSENT_PURPOSE,
                "data_fiduciary": DATA_FIDUCIARY,
                "expires_at": _consent_expiry(),
                "status": "revoked",
                "user_id": user_id,
                "revoked_scopes": CONSENT_SCOPES.copy(),
            }
        if user_id:
            _revoked_users.add(user_id)

    if user_id:
        _user_consent_map[user_id] = consent_id

    # Determine response status
    is_fully_revoked = (
        consent_id in _revoked_consents or
        not request.scopes or
        (consent_id in _active_consents and not _active_consents[consent_id]["scopes"])
    )
    status_str = "revoked" if is_fully_revoked else "partial_revoked"

    return ConsentRevokeResponse(
        consent_id=consent_id,
        user_id=user_id,
        status=status_str,
        effective_from=now,
    )


@router.post("/erasure", response_model=ErasureResponse)
async def request_erasure(
    request: ErasureRequest,
    db: AsyncSession = Depends(get_db),
) -> ErasureResponse:
    """DPDP Act 2023 right to erasure. Deletes raw PII and derived features;
    retains anonymised ScoreDecision records for RBI regulatory audit."""
    try:
        uid = UUID(request.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid user_id format") from exc

    # Count audit records that will be retained
    count_res = await db.execute(
        sa_select(func.count()).select_from(ScoreDecision).where(ScoreDecision.user_id == uid)
    )
    audit_retained = count_res.scalar() or 0

    # Delete raw encrypted payloads
    vault_res = await db.execute(
        sa_delete(SecureVault).where(SecureVault.user_id == uid)
    )
    # Delete ML feature rows
    feat_res = await db.execute(
        sa_delete(MLFeature).where(MLFeature.user_id == uid)
    )
    # Delete feature time-series
    series_res = await db.execute(
        sa_delete(FeatureSeries).where(FeatureSeries.user_id == uid)
    )
    await db.commit()

    vault_deleted = vault_res.rowcount
    feature_deleted = (feat_res.rowcount or 0) + (series_res.rowcount or 0)

    _erasure_requests[request.user_id] = {
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vault_deleted": vault_deleted,
        "features_deleted": feature_deleted,
        "audit_retained": audit_retained,
    }

    logger.info("Erasure completed for user %s: vault=%d features=%d audit_retained=%d",
                request.user_id, vault_deleted, feature_deleted, audit_retained)

    return ErasureResponse(
        user_id=request.user_id,
        vault_records_deleted=vault_deleted,
        feature_records_deleted=feature_deleted,
        audit_records_retained=audit_retained,
    )


def get_revoked_scopes(user_id: str) -> list[str]:
    """Return the list of revoked scopes for a given user_id."""
    if user_id in _revoked_users:
        return CONSENT_SCOPES.copy()
    
    consent_id = _user_consent_map.get(user_id)
    if consent_id:
        if consent_id in _revoked_consents:
            return CONSENT_SCOPES.copy()
        record = _active_consents.get(consent_id)
        if record:
            revoked = list(record.get("revoked_scopes", []))
            # Cascade parent scope revocations to sub-scopes
            if "cashflow" in revoked:
                if "upi_lite" not in revoked:
                    revoked.append("upi_lite")
                if "dbt_logs" not in revoked:
                    revoked.append("dbt_logs")
            if "telecom" in revoked:
                if "sms_parsing" not in revoked:
                    revoked.append("sms_parsing")
            if "farmer" in revoked:
                if "enam_receipts" not in revoked:
                    revoked.append("enam_receipts")
            return revoked
    return []


@router.get("/status/{user_id}", response_model=ConsentStatusResponse)
async def consent_status(
    user_id: str,
    db: AsyncSession = Depends(get_db),
) -> ConsentStatusResponse:
    """Return consent and erasure status for a borrower (for privacy dashboard)."""
    consent_id = _user_consent_map.get(user_id)
    active_scopes = []
    revoked_scopes = []

    if user_id in _revoked_users:
        c_status = "revoked"
        revoked_scopes = CONSENT_SCOPES.copy()
    elif consent_id and consent_id in _revoked_consents:
        c_status = "revoked"
        revoked_scopes = CONSENT_SCOPES.copy()
    elif consent_id and consent_id in _active_consents:
        record = _active_consents[consent_id]
        active_scopes = list(record.get("scopes", []))
        revoked_scopes = list(record.get("revoked_scopes", []))
        if revoked_scopes:
            c_status = "partial" if active_scopes else "revoked"
        else:
            c_status = "active"
    else:
        c_status = "unknown"

    erasure = _erasure_requests.get(user_id)

    # Check whether any vault data still exists
    try:
        uid = UUID(user_id)
        count_res = await db.execute(
            sa_select(func.count()).select_from(SecureVault).where(SecureVault.user_id == uid)
        )
        vault_count = count_res.scalar() or 0
        data_present = vault_count > 0
    except ValueError:
        data_present = False

    return ConsentStatusResponse(
        user_id=user_id,
        consent_id=consent_id,
        consent_status=c_status,
        data_present=data_present,
        erasure_requested=bool(erasure),
        erasure_status=erasure.get("status") if erasure else None,
        erasure_timestamp=erasure.get("timestamp") if erasure else None,
        active_scopes=active_scopes,
        revoked_scopes=revoked_scopes,
    )


@router.get("/compliance")
async def compliance_summary() -> dict:
    """Regulatory compliance summary for bank stakeholders."""
    return {
        "frameworks": [
            "RBI Account Aggregator Framework (consent artifact with purpose, expiry, revocation)",
            "DPDP Act 2023 (data minimization, consent, encryption at rest, right to erasure)",
            "RBI Digital Lending Guidelines 2022 (Key Fact Statement, cooling-off period)",
        ],
        "borrower_rights": {
            "revoke_consent": "POST /consent/revoke: stops future data sharing immediately",
            "request_erasure": "POST /consent/erasure: deletes raw PII and ML features",
            "retained_after_erasure": "Anonymised credit decision records (score, decision, date) retained 5 years per RBI",
            "bureau_caveat": "Scores already submitted to credit bureaus cannot be recalled, governed by bureau rules",
            "check_status": "GET /consent/status/{user_id}",
        },
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
