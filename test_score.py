import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from api.routes.scoring import DEMO_INTAKE_PROFILES
from convergence.score_engine import score_user
from core.model_cache import init_model_cache
from models.db_models import MLFeature
from sqlalchemy import select

async def main():
    engine = create_async_engine("sqlite+aiosqlite:///alt_credit.db")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    init_model_cache()
    
    async with async_session() as db:
        # Vendor cohort_code is 4.0 in feature_meta.py, let's just query ApplicationIntake
        from models.db_models import ApplicationIntake
        stmt = select(ApplicationIntake.user_id).where(ApplicationIntake.cohort == "Vendor").limit(1)
        result = await db.execute(stmt)
        user_uuid = result.scalar_one_or_none()
        
        if not user_uuid:
            print("No vendor user in ApplicationIntake")
            return
            
        payload = await score_user(db, str(user_uuid), persist=False)
        print("credit_score:", payload.get("credit_score"))
        print("requested_amount:", payload.get("requested_amount"))
        print("final_outcome:", payload.get("final_outcome"))
        print("funding_gap:", payload.get("funding_gap"))

if __name__ == "__main__":
    asyncio.run(main())
