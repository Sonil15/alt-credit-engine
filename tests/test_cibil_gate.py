import pytest
import uuid
from core.database import AsyncSessionLocal
from models.db_models import BorrowerAccount
from convergence.score_engine import score_user, score_all_users

@pytest.mark.asyncio
async def test_cibil_gating_prime():
    user_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as session:
        # Create borrower account with Prime CIBIL score
        acc = BorrowerAccount(
            user_id=uuid.UUID(user_id),
            login_id=f"prime_test_{user_id[:8]}",
            password_hash=b"xyz",
            password_salt="salt",
            cibil_score=780
        )
        session.add(acc)
        await session.commit()

        # Score the user (should bypass feature store fetch)
        payload = await score_user(session, user_id, persist=False)
        assert payload["decision"] == "APPROVE"
        assert payload["credit_score"] == 780
        assert payload["lending"]["eligible"] is True
        assert payload["lending"]["interest_rate_pct"] == 11.0
        assert payload["lending"]["tenure_months"] == 36
        assert payload["explanation_method"] == "bureau-gating"

@pytest.mark.asyncio
async def test_cibil_gating_subprime():
    user_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as session:
        # Create borrower account with Subprime CIBIL score
        acc = BorrowerAccount(
            user_id=uuid.UUID(user_id),
            login_id=f"subprime_test_{user_id[:8]}",
            password_hash=b"xyz",
            password_salt="salt",
            cibil_score=520
        )
        session.add(acc)
        await session.commit()

        # Score the user (should bypass feature store fetch)
        payload = await score_user(session, user_id, persist=False)
        assert payload["decision"] == "REJECT"
        assert payload["credit_score"] == 520
        assert payload["lending"]["eligible"] is False
        assert payload["explanation_method"] == "bureau-gating"

@pytest.mark.asyncio
async def test_cibil_gating_thin_file():
    user_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as session:
        # Create borrower account with Thin File CIBIL score
        acc = BorrowerAccount(
            user_id=uuid.UUID(user_id),
            login_id=f"thin_test_{user_id[:8]}",
            password_hash=b"xyz",
            password_salt="salt",
            cibil_score=-1
        )
        session.add(acc)
        await session.commit()

        # Score the user (should try alt-credit but fail because no features exist in the DB for this new UUID)
        with pytest.raises(ValueError, match="not found in ml_features"):
            await score_user(session, user_id, persist=False)
