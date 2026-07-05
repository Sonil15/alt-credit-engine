"""Intake API: submit validation, latest-wins, vault encryption of raw text."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from api.main import app
from core.database import AsyncSessionLocal
from models.db_models import ApplicationIntake, SecureVault


async def _register(ac: AsyncClient) -> dict:
    resp = await ac.post(
        "/auth/register",
        json={"login_id": f"intake-{uuid.uuid4().hex[:10]}", "password": "secret123"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _headers(account: dict) -> dict:
    return {"Authorization": f"Bearer {account['token']}"}


@pytest.mark.asyncio
async def test_submit_happy_path_and_latest_wins():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        account = await _register(ac)
        body = {
            "user_id": account["user_id"],
            "cohort": "Vendor",
            "loan_purpose": "inventory",
            "requested_amount": 200000,
            "business_description": "sabzi stall for 8 years, earn 40,000 a month",
            "business_profile": {
                "sector": "retail",
                "years_in_business": 8,
                "monthly_turnover": 40000,
            },
            "extraction_method": "fallback",
        }
        first = await ac.post("/intake/submit", json=body, headers=_headers(account))
        assert first.status_code == 200, first.text

        body["requested_amount"] = 50000
        body["loan_purpose"] = "working_capital"
        second = await ac.post("/intake/submit", json=body, headers=_headers(account))
        assert second.status_code == 200

        latest = await ac.get(f"/intake/{account['user_id']}", headers=_headers(account))
        assert latest.status_code == 200
        data = latest.json()
        assert data["requested_amount"] == 50000
        assert data["loan_purpose"] == "working_capital"
        assert data["business_profile"]["years_in_business"] == 8

    # Raw description went to the encrypted vault, not the intake table.
    async with AsyncSessionLocal() as db:
        vault_rows = (
            await db.execute(
                select(SecureVault)
                .where(SecureVault.user_id == uuid.UUID(account["user_id"]))
                .where(SecureVault.data_type == "intake")
            )
        ).scalars().all()
        assert len(vault_rows) == 2
        assert b"sabzi" not in vault_rows[0].encrypted_payload

        intake_rows = (
            await db.execute(
                select(ApplicationIntake).where(
                    ApplicationIntake.user_id == uuid.UUID(account["user_id"])
                )
            )
        ).scalars().all()
        assert len(intake_rows) == 2


@pytest.mark.asyncio
async def test_purpose_cohort_mismatch_is_422():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        account = await _register(ac)
        resp = await ac.post(
            "/intake/submit",
            json={
                "user_id": account["user_id"],
                "cohort": "Student",
                "loan_purpose": "inventory",  # a Vendor purpose
                "requested_amount": 10000,
            },
            headers=_headers(account),
        )
        assert resp.status_code == 422
        assert "not available" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_submit_requires_auth_and_matching_user():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        body = {
            "user_id": str(uuid.uuid4()),
            "cohort": "Salaried",
            "loan_purpose": "personal",
            "requested_amount": 10000,
        }
        anonymous = await ac.post("/intake/submit", json=body)
        assert anonymous.status_code == 401

        account = await _register(ac)
        other = await ac.post("/intake/submit", json=body, headers=_headers(account))
        assert other.status_code == 403  # token does not own that user_id


@pytest.mark.asyncio
async def test_business_profile_endpoint_falls_back_without_llm():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        account = await _register(ac)
        resp = await ac.post(
            "/intake/business-profile",
            json={"text": "kirana shop for 6 years, kamai 25,000 monthly", "language": "en"},
            headers=_headers(account),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] in {"llm", "fallback"}
        assert data["profile"]["sector"] == "retail"
        assert data["profile"]["years_in_business"] == 6
