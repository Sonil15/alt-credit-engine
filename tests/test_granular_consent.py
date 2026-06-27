import pytest
from httpx import ASGITransport, AsyncClient
from api.main import app
from api.routes.consent import get_revoked_scopes, _active_consents, _revoked_users, _revoked_consents, _user_consent_map

@pytest.mark.asyncio
async def test_granular_consent_flow():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user_id = "test-user-consent-123"
        
        # 1. Authorize
        auth_resp = await ac.get(f"/consent/authorize?user_id={user_id}")
        assert auth_resp.status_code == 200
        auth_data = auth_resp.json()
        assert auth_data["consent_id"]
        consent_id = auth_data["consent_id"]
        
        # Check initial status
        status_resp = await ac.get(f"/consent/status/{user_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["consent_status"] == "active"
        assert "telecom" in status_data["active_scopes"]
        assert "telecom" not in status_data["revoked_scopes"]
        
        # 2. Granular Revoke: Revoke "telecom" scope
        revoke_resp = await ac.post("/consent/revoke", json={"user_id": user_id, "scopes": ["telecom"]})
        assert revoke_resp.status_code == 200
        revoke_data = revoke_resp.json()
        assert revoke_data["status"] == "partial_revoked"
        
        # Check updated status
        status_resp2 = await ac.get(f"/consent/status/{user_id}")
        assert status_resp2.status_code == 200
        status_data2 = status_resp2.json()
        assert status_data2["consent_status"] == "partial"
        assert "telecom" not in status_data2["active_scopes"]
        assert "telecom" in status_data2["revoked_scopes"]
        
        # Expose and test get_revoked_scopes
        revoked = get_revoked_scopes(user_id)
        assert "telecom" in revoked
        assert "ecommerce" not in revoked

        # Clean up in-memory mappings
        if consent_id in _active_consents:
            del _active_consents[consent_id]
        if user_id in _user_consent_map:
            del _user_consent_map[user_id]
        _revoked_users.discard(user_id)
        _revoked_consents.discard(consent_id)
