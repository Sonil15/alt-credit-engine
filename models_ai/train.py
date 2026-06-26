"""One-shot training script: ECM pipeline + CatBoost model."""

import asyncio
import logging
import sys

from core.database import AsyncSessionLocal
from core.model_cache import reload_model_cache
from models_ai.ensemble import train_all_from_db
from models_econometric.ecm_model import run_ecm_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    async with AsyncSessionLocal() as session:
        logger.info("Running ECM pipeline on real time series...")
        ecm_result = await run_ecm_pipeline(session)
        logger.info("ECM result: %s", ecm_result)

        logger.info("Training champion (EBM) + challenger panel (CatBoost, logistic)...")
        train_result = await train_all_from_db(session)
        logger.info("Training result: %s", train_result)

    reload_model_cache()

    metrics = train_result.get("metrics", {})
    cv = train_result.get("cv_metrics", {})
    challengers = train_result.get("challenger_metrics", {})
    print("Training complete.")
    print(f"  Users (ECM): {ecm_result['users_processed']}")
    print(f"  Users trained: {train_result['users_trained']}")
    print(f"  Default rate: {train_result['default_rate']:.2%}")
    print(f"  Champion (EBM) Holdout AUC: {metrics.get('auc', 0):.4f}  Gini: {metrics.get('gini', 0):.4f}  KS: {metrics.get('ks', 0):.4f}")
    print(f"  Champion CV AUC: {cv.get('cv_auc_mean', 0):.4f} (+/- {cv.get('cv_auc_std', 0):.4f})")
    for name, m in challengers.items():
        print(f"  Challenger {name} Holdout AUC: {m.get('auc', 0):.4f}")
    print(f"  Champion saved: {train_result['model_path']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)
