import asyncio
import json
import os
import sys

os.environ["USE_SQLITE"] = "true"
os.environ["SIMULATE_ALL_FACETS"] = "true"
sys.path.insert(0, ".")

from core.database import AsyncSessionLocal
from core.feature_store import fetch_features_wide
from convergence.score_engine import score_all_users
from convergence.fairness import compute_fairness_report
from core.model_cache import init_model_cache
from pathlib import Path

async def main():
    init_model_cache()
    
    async with AsyncSessionLocal() as session:
        wide = await fetch_features_wide(session)
        scores = await score_all_users(session)
    report = compute_fairness_report(scores, wide)
    
    # We want the social_category dimension specifically
    dimensions = report.get("dimensions", {})
    sc_dim = dimensions.get("social_category", {})
    
    print(f"Current DI ratio for social_category: {sc_dim.get('disparate_impact_ratio')}")
    print(f"Groups: {json.dumps(sc_dim.get('groups', {}), indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
