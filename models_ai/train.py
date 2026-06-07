"""One-shot training script: ECM pipeline + CatBoost model."""

import asyncio
import logging
import sys

from core.database import AsyncSessionLocal
from models_ai.catboost_model import train_from_db
from models_econometric.ecm_model import run_ecm_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    async with AsyncSessionLocal() as session:
        logger.info("Running ECM pipeline...")
        ecm_result = await run_ecm_pipeline(session)
        logger.info("ECM result: %s", ecm_result)

        logger.info("Training CatBoost model...")
        train_result = await train_from_db(session)
        logger.info("Training result: %s", train_result)

    print("Training complete.")
    print(f"  Users (ECM): {ecm_result['users_processed']}")
    print(f"  Users (CatBoost): {train_result['users_trained']}")
    print(f"  Synthetic default rate: {train_result['default_rate']:.2%}")
    print(f"  Model saved: {train_result['model_path']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)
