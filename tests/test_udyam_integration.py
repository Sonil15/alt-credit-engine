import uuid
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from api.main import app
from core.database import AsyncSessionLocal
from models.db_models import ApplicationIntake, MLFeature
from core.business_profile import upsert_intake_features

async def _register(ac: AsyncClient) -> dict:
    resp = await ac.post(
        "/auth/register",
        json={"login_id": f"udyam-{uuid.uuid4().hex[:10]}", "password": "secret123"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()

def _headers(account: dict) -> dict:
    return {"Authorization": f"Bearer {account['token']}"}

@pytest.mark.asyncio
async def test_udyam_api_and_feature_extraction():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        account = await _register(ac)
        
        # 1. Submit intake with Udyam info
        unique_udyam = f"UDYAM-DL-01-{uuid.uuid4().hex[:7].upper()}"
        body = {
            "user_id": account["user_id"],
            "cohort": "Vendor",
            "loan_purpose": "working_capital",
            "requested_amount": 50000,
            "business_description": "retail store for 10 years",
            "business_profile": {
                "sector": "retail",
                "years_in_business": 10.0,
                "monthly_turnover": 40000.0,
                "udyam_number": unique_udyam,
                "udyam_vintage_years": 3.0,
                "years_informal": 7.0
            },
            "extraction_method": "fallback"
        }
        resp = await ac.post("/intake/submit", json=body, headers=_headers(account))
        assert resp.status_code == 200, resp.text
        
        # Check intake API retrieve Udyam details
        latest = await ac.get(f"/intake/{account['user_id']}", headers=_headers(account))
        assert latest.status_code == 200
        data = latest.json()
        assert data["business_profile"]["udyam_number"] == unique_udyam
        assert data["business_profile"]["udyam_vintage_years"] == 3.0
        assert data["business_profile"]["years_informal"] == 7.0

    # 2. Materialize features
    async with AsyncSessionLocal() as db:
        await upsert_intake_features(db, account["user_id"])
        await db.commit()
        
        # 3. Check ML Features table
        feats_result = await db.execute(
            select(MLFeature).where(MLFeature.user_id == uuid.UUID(account["user_id"]))
        )
        feats = {f.feature_name: f.feature_value for f in feats_result.scalars().all()}
        
        assert "has_udyam_registration" in feats
        assert feats["has_udyam_registration"] == 1.0
        assert "years_informal" in feats
        assert feats["years_informal"] == 7.0
        assert feats["business_vintage_years"] == 10.0


@pytest.mark.asyncio
async def test_udyam_velocity_check_collision():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user1 = await _register(ac)
        user2 = await _register(ac)
        
        collision_udyam = f"UDYAM-COLL-{uuid.uuid4().hex[:7].upper()}"
        body1 = {
            "user_id": user1["user_id"],
            "cohort": "Vendor",
            "loan_purpose": "working_capital",
            "requested_amount": 50000,
            "business_description": "retail store",
            "business_profile": {
                "sector": "retail",
                "years_in_business": 10.0,
                "monthly_turnover": 40000.0,
                "udyam_number": collision_udyam,
                "udyam_vintage_years": 3.0,
                "years_informal": 7.0
            },
            "extraction_method": "fallback"
        }
        resp1 = await ac.post("/intake/submit", json=body1, headers=_headers(user1))
        assert resp1.status_code == 200, resp1.text
        
        body2 = {
            "user_id": user2["user_id"],
            "cohort": "Vendor",
            "loan_purpose": "working_capital",
            "requested_amount": 30000,
            "business_description": "another retail store",
            "business_profile": {
                "sector": "retail",
                "years_in_business": 5.0,
                "monthly_turnover": 20000.0,
                "udyam_number": collision_udyam,
                "udyam_vintage_years": 2.0,
                "years_informal": 3.0
            },
            "extraction_method": "fallback"
        }
        resp2 = await ac.post("/intake/submit", json=body2, headers=_headers(user2))
        assert resp2.status_code == 400, resp2.text
        assert "Velocity Check Failed" in resp2.json()["detail"]

