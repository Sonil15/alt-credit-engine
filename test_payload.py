import asyncio
import json
import os
from sqlalchemy import select

os.environ['USE_SQLITE'] = 'true'

from core.database import AsyncSessionLocal
from models.db_models import ApplicationIntake, MLFeature
from convergence.score_engine import score_user
from api.routes.scoring import DEMO_INTAKE_PROFILES

async def main():
    async with AsyncSessionLocal() as session:
        cohort = "Vendor"
        cohort_code_val = 3.0
        
        # 1. Find a seeded user of this cohort
        stmt = select(MLFeature.user_id).where(
            MLFeature.feature_name == "cohort_code",
            MLFeature.feature_value == cohort_code_val
        ).limit(1)
        result = await session.execute(stmt)
        user_uuid = result.scalar_one_or_none()
        
        print("user_uuid:", user_uuid)
        
        # 2. Check mock ApplicationIntake
        intake_stmt = select(ApplicationIntake).where(ApplicationIntake.user_id == user_uuid)
        intake_result = await session.execute(intake_stmt)
        intake_row = intake_result.scalar_one_or_none()
        
        if not intake_row:
            mock_data = DEMO_INTAKE_PROFILES[cohort]
            intake_row = ApplicationIntake(
                user_id=user_uuid,
                cohort=mock_data["cohort"],
                loan_purpose=mock_data["loan_purpose"],
                loan_purpose_other_text=mock_data["loan_purpose_other_text"],
                requested_amount=mock_data["requested_amount"],
                business_profile_json=json.dumps(mock_data["business_profile"]),
                extraction_method=mock_data["extraction_method"],
                extraction_confidence=mock_data["extraction_confidence"]
            )
            session.add(intake_row)
            await session.commit()
            
        score_payload = await score_user(session, str(user_uuid), persist=False)
        with open("vendor_payload.json", "w") as f:
            json.dump(score_payload, f, indent=2)

asyncio.run(main())
