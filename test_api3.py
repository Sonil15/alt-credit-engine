import asyncio
import os
os.environ['USE_SQLITE'] = 'true'
from core.database import AsyncSessionLocal
from api.routes.scoring import get_demo_audit_trail
from fastapi import HTTPException
from core.model_cache import init_model_cache
import json

async def main():
    init_model_cache()
    async with AsyncSessionLocal() as session:
        try:
            res = await get_demo_audit_trail(cohort="Vendor", db=session)
            print(json.dumps(res, indent=2))
        except Exception as e:
            print(f"Error: {e}")

asyncio.run(main())
