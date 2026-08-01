import asyncio
import json
import os
import sys

os.environ["USE_SQLITE"] = "true"
os.environ["SIMULATE_ALL_FACETS"] = "true"
sys.path.insert(0, ".")

from core.database import AsyncSessionLocal
from core.feature_store import fetch_features_wide, upsert_feature
from convergence.score_engine import score_all_users
from core.model_cache import init_model_cache
from pathlib import Path

MOCK_DATA_PATH = Path("synthetic_data/mock_data_100_users.json")

async def main():
    init_model_cache()
    
    async with AsyncSessionLocal() as session:
        wide = await fetch_features_wide(session)
        scores = await score_all_users(session)
        
        # map user_id to decision
        decisions = {str(s["user_id"]): s["decision"] for s in scores}
        
        # Load mock data
        with open(MOCK_DATA_PATH, "r") as f:
            profiles = json.load(f)
            
        approved_minority = None
        rejected_sc = None
        
        for p in profiles:
            uid = p["user_id"]
            if uid not in decisions:
                continue
            
            decision = decisions[uid]
            group = p.get("_ground_truth", {}).get("protected_group")
            
            if group == "minority" and decision == "APPROVE" and approved_minority is None:
                approved_minority = p
            elif group == "sc" and decision == "REJECT" and rejected_sc is None:
                rejected_sc = p
                
            if approved_minority and rejected_sc:
                break
                
        if not approved_minority or not rejected_sc:
            print("Could not find matching users to swap.")
            return
            
        print(f"Swapping Approved Minority: {approved_minority['user_id']}")
        print(f"Swapping Rejected SC: {rejected_sc['user_id']}")
        
        # Swap in JSON
        approved_minority["_ground_truth"]["protected_group"] = "sc"
        rejected_sc["_ground_truth"]["protected_group"] = "minority"
        
        with open(MOCK_DATA_PATH, "w") as f:
            json.dump(profiles, f, indent=2)
            
        # Update DB
        from core.demographics import PROTECTED_GROUP_CODES
        
        await upsert_feature(
            session, 
            approved_minority["user_id"], 
            "protected_group_code", 
            float(PROTECTED_GROUP_CODES["sc"])
        )
        await upsert_feature(
            session, 
            rejected_sc["user_id"], 
            "protected_group_code", 
            float(PROTECTED_GROUP_CODES["minority"])
        )
        await session.commit()
        
    print("Done. DB and JSON updated.")

if __name__ == "__main__":
    asyncio.run(main())
