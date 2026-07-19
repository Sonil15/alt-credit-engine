import asyncio
import os
os.environ["USE_SQLITE"] = "true"
os.environ["AUTO_SEED_ON_STARTUP"] = "true"

from core.database import AsyncSessionLocal
from convergence.score_engine import score_all_users
from core.model_cache import init_model_cache

async def run():
    init_model_cache()
    async with AsyncSessionLocal() as session:
        scores = await score_all_users(session)
        for s in scores:
            if s.get("requested_amount"):
                print(f"User {s['user_id']}: requested={s['requested_amount']}, score={s['credit_score']}, gap={s.get('funding_gap')}")

asyncio.run(run())
