import pytest
from httpx import ASGITransport, AsyncClient
from api.main import app
from core.model_cache import init_model_cache

@pytest.mark.asyncio
async def test_simulate_endpoint():
    # Initialize the model cache manually for the test environment
    init_model_cache()
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Run a simulated score evaluation
        response = await ac.post("/score/simulate", json={"theta": 0.8})
        assert response.status_code == 200
        data = response.json()
        
        # Verify the structure and contents of the simulation response
        assert "score" in data
        assert "raw_data" in data
        
        score = data["score"]
        assert score["is_simulated"] is True
        assert score["credit_score"] >= 300
        assert score["credit_score"] <= 900
        
        # Verify that all 5 sources of raw data are returned
        raw = data["raw_data"]
        assert "telecom" in raw
        assert "ecommerce" in raw
        assert "geo" in raw
        assert "cashflow" in raw
        assert "survey" in raw

        # Verify that features and overrides work
        response_override = await ac.post("/score/simulate", json={
            "theta": 0.5,
            "feature_overrides": {
                "missed_payments_count": 6.0
            }
        })
        assert response_override.status_code == 200
        data_override = response_override.json()
        assert data_override["score"]["is_simulated"] is True
        assert data_override["score"]["factor_points"] is not None
