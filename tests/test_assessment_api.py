import uuid
from datetime import datetime, timedelta, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from core.database import AsyncSessionLocal
from models.db_models import SecureVault
from psychometric.session import create_session, start_response, submit_answer


@pytest.mark.asyncio
async def test_assessment_session_flow():
    session = create_session(user_id="test-user-123", language="hi")
    start = start_response(session)
    assert start["session_id"]
    assert start["language"] == "hi"
    assert start["item"] is not None

    item_id = start["item"]["item_id"]
    result = await submit_answer(session.session_id, item_id, "A")
    assert result["completed"] is False or result["completed"] is True


async def _add_survey_record(user_id: str, when: datetime) -> None:
    async with AsyncSessionLocal() as db:
        db.add(
            SecureVault(
                user_id=uuid.UUID(user_id),
                data_type="survey",
                encrypted_payload=b"x",
                timestamp=when,
            )
        )
        await db.commit()


@pytest.mark.asyncio
async def test_application_rate_limit_blocks_repeat_borrower():
    user_id = str(uuid.uuid4())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # A fresh borrower's first application is allowed.
        first = await ac.post("/assessment/start", json={"user_id": user_id, "language": "en"})
        assert first.status_code == 200

        # Simulate a just-completed application (writes the survey vault record).
        await _add_survey_record(user_id, _utcnow())

        # Within the 30-day window the borrower is now throttled.
        blocked = await ac.post("/assessment/start", json={"user_id": user_id, "language": "en"})
        assert blocked.status_code == 429
        assert "Application limit" in blocked.json()["detail"]


@pytest.mark.asyncio
async def test_application_rate_limit_allows_after_window():
    # An application older than the 30-day window must not block a new one.
    user_id = str(uuid.uuid4())
    await _add_survey_record(user_id, _utcnow() - timedelta(days=60))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/assessment/start", json={"user_id": user_id, "language": "en"})
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_application_rate_limit_skips_anonymous():
    # No user_id => nothing to key the throttle on; must not be blocked.
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/assessment/start", json={"language": "en"})
        assert resp.status_code == 200
