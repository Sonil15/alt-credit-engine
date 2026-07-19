import asyncio
import json
import os
from sqlalchemy import select

os.environ['USE_SQLITE'] = 'true'

from core.database import AsyncSessionLocal
from models.db_models import ApplicationIntake, MLFeature
from convergence.score_engine import score_user
from core.model_cache import init_model_cache

async def main():
    init_model_cache()
    async with AsyncSessionLocal() as session:
        cohort_code_val = 3.0
        
        stmt = select(MLFeature.user_id).where(
            MLFeature.feature_name == "cohort_code",
            MLFeature.feature_value == cohort_code_val
        ).limit(1)
        result = await session.execute(stmt)
        user_uuid = result.scalar_one_or_none()
        
        score_payload = await score_user(session, str(user_uuid), persist=False)
        with open("vendor_payload.json", "w") as f:
            json.dump(score_payload, f, indent=2)

asyncio.run(main())
