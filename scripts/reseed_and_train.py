import asyncio
import os
import sys
import logging

# Ensure repo root is on path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)

from core.database import init_db
from core.bootstrap import ensure_seeded
from models_ai.ensemble import train_all_from_db
from models_econometric.ecm_model import run_ecm_pipeline
from core.database import AsyncSessionLocal

async def main():
    db_file = "alt_credit.db"
    if os.path.exists(db_file):
        logging.info("Removing existing SQLite database to reset tables...")
        os.remove(db_file)
        
    logging.info("Initializing database...")
    await init_db()
    
    logging.info("Seeding database with new mock profiles...")
    seed_res = await ensure_seeded()
    logging.info("Seed result: %s", seed_res)
    
    async with AsyncSessionLocal() as session:
        logging.info("Running ECM econometric pipeline...")
        ecm_result = await run_ecm_pipeline(session)
        logging.info("ECM result: %s", ecm_result)
        
        logging.info("Training EBM champion and CatBoost/Logistic challengers...")
        train_result = await train_all_from_db(session)
        logging.info("Training complete!")
        metrics = train_result.get("metrics", {})
        logging.info("Champion Holdout AUC: %.4f", metrics.get("auc", 0))

if __name__ == "__main__":
    asyncio.run(main())
