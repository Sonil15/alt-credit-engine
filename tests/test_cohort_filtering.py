import pytest
from sqlalchemy import select
from core.database import AsyncSessionLocal
from models.db_models import MLFeature
from convergence.score_engine import score_user, SCOPE_TO_FEATURES
from core.model_cache import init_model_cache

# Initialize the model cache so that score_user can access the EBM champion
init_model_cache()

@pytest.mark.asyncio
async def test_student_cohort_filtering():
    async with AsyncSessionLocal() as session:
        # Find a seeded student (cohort_code = 2.0)
        stmt = select(MLFeature.user_id).where(
            MLFeature.feature_name == "cohort_code",
            MLFeature.feature_value == 2.0
        ).limit(1)
        result = await session.execute(stmt)
        user_uuid = result.scalar_one_or_none()
        
        assert user_uuid is not None, "Please seed the DB. Run test with AUTO_SEED_ON_STARTUP=true"
        
        # Get score payload
        payload = await score_user(session, str(user_uuid), persist=False)
        
        # Allowed scopes for Student: geo, survey, campus
        allowed_scopes = {"geo", "survey", "campus"}
        allowed_features = set()
        for scope in allowed_scopes:
            allowed_features.update(SCOPE_TO_FEATURES[scope])
            
        # Get all features in the trace
        by_source = payload["feature_trace"].get("by_source", [])
        for src_group in by_source:
            for sig in src_group.get("signals", []):
                feature_name = sig["feature"]
                # Verify that no irrelevant features are shown
                assert feature_name in allowed_features, f"Student has irrelevant feature: {feature_name}"
                
        # Also check top_drivers in feature_trace
        top_drivers = payload["feature_trace"].get("top_drivers", [])
        for sig in top_drivers:
            feature_name = sig["feature"]
            assert feature_name in allowed_features, f"Student top_drivers has irrelevant feature: {feature_name}"

@pytest.mark.asyncio
async def test_farmer_cohort_filtering():
    async with AsyncSessionLocal() as session:
        # Find a seeded farmer (cohort_code = 4.0)
        stmt = select(MLFeature.user_id).where(
            MLFeature.feature_name == "cohort_code",
            MLFeature.feature_value == 4.0
        ).limit(1)
        result = await session.execute(stmt)
        user_uuid = result.scalar_one_or_none()
        
        assert user_uuid is not None, "Please seed the DB. Run test with AUTO_SEED_ON_STARTUP=true"
        
        # Get score payload
        payload = await score_user(session, str(user_uuid), persist=False)
        
        # Allowed scopes for Farmer: geo, survey, farmer
        # Note: Business credentials features (vintage, udyam, etc.) are handled separately,
        # but farmer features must be present while student/telecom/ecommerce/household are hidden.
        allowed_scopes = {"geo", "survey", "farmer"}
        allowed_features = set()
        for scope in allowed_scopes:
            allowed_features.update(SCOPE_TO_FEATURES[scope])
        
        # We also allow BUSINESS_MODEL_FEATURES for Farmer (business_credentials)
        from core.business_profile import BUSINESS_MODEL_FEATURES
        allowed_features.update(BUSINESS_MODEL_FEATURES)
            
        by_source = payload["feature_trace"].get("by_source", [])
        for src_group in by_source:
            for sig in src_group.get("signals", []):
                feature_name = sig["feature"]
                assert feature_name in allowed_features, f"Farmer has irrelevant feature: {feature_name}"
