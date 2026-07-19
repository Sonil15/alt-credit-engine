import asyncio
import os
os.environ["USE_SQLITE"] = "true"
os.environ["AUTO_SEED_ON_STARTUP"] = "true"

from core.database import AsyncSessionLocal
from api.routes.scoring import get_demo_audit_trail
from core.model_cache import init_model_cache

async def run():
    init_model_cache()
    async with AsyncSessionLocal() as session:
        result = await get_demo_audit_trail("Vendor", session)
        score = result["score_result"]
        print(f"Vendor score={score['credit_score']}, requested_amount={score['requested_amount']}, max_loan={score['lending'].get('max_loan_amount')}")

asyncio.run(run())
