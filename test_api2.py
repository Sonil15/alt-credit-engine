import asyncio
from core.database import get_db_session_context
from api.routes.scoring import get_demo_audit_trail
from convergence.score_engine import score_user
import json

async def main():
    async with get_db_session_context() as session:
        # Get demo user
        audit = await get_demo_audit_trail("Vendor", session)
        user_id = audit["user_id"]
        
        # Get score
        score = await score_user(session, user_id)
        print("Score payload:")
        print(json.dumps(score, indent=2))

asyncio.run(main())
