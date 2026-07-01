"""Zero-friction startup seeding.

Populates an empty database with the bundled mock borrower cohort so the demo is
fully interactive the instant the server boots — no Docker, no manual
generate/load/train sequence. Reuses the exact same encryption + preprocessing
path as live ingestion (`process_vault_record`), then runs the ECM pipeline so
econometric features are present. Training is NOT done here: the committed
CatBoost artifact is loaded by the model cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select

from core.database import AsyncSessionLocal
from core.feature_store import upsert_feature
from core.security import get_encryptor
from models.db_models import MLFeature, SecureVault
from preprocessing.pipeline import process_vault_record

logger = logging.getLogger(__name__)

MOCK_DATA_PATH = Path(__file__).resolve().parent.parent / "synthetic_data" / "mock_data_100_users.json"
DATA_TYPES = ("telecom", "ecommerce", "geo", "cashflow", "survey")
PROTECTED_GROUP_CODES = {"general": 0, "obc": 1, "sc": 2, "st": 3, "minority": 4}
GENDER_CODES = {"male": 0, "female": 1, "other": 2}
GEOGRAPHY_CODES = {"rural": 0, "semi_urban": 1, "urban": 2}
INCOME_BRACKET_CODES = {"low": 0, "mid": 1, "high": 2}


async def _already_seeded() -> bool:
    async with AsyncSessionLocal() as session:
        count = await session.scalar(select(func.count()).select_from(MLFeature))
        return bool(count and count > 0)


async def _store_ground_truth(user_id: str, ground_truth: dict) -> None:
    async with AsyncSessionLocal() as session:
        await upsert_feature(session, user_id, "default_label", float(ground_truth.get("default_label", 0)))
        await upsert_feature(
            session,
            user_id,
            "protected_group_code",
            float(PROTECTED_GROUP_CODES.get(ground_truth.get("protected_group", "general"), 0)),
        )
        await upsert_feature(
            session,
            user_id,
            "borrower_type",
            1.0 if ground_truth.get("borrower_type") == "msme" else 0.0,
        )
        await upsert_feature(
            session,
            user_id,
            "gender_code",
            float(GENDER_CODES.get(ground_truth.get("gender", "male"), 0)),
        )
        await upsert_feature(
            session,
            user_id,
            "geography_code",
            float(GEOGRAPHY_CODES.get(ground_truth.get("geography", "rural"), 0)),
        )
        await upsert_feature(
            session,
            user_id,
            "income_bracket_code",
            float(INCOME_BRACKET_CODES.get(ground_truth.get("income_bracket", "mid"), 1)),
        )
        await session.commit()


async def _ingest_profile(user_id: str, profile: dict) -> None:
    encryptor = get_encryptor()
    for data_type in DATA_TYPES:
        payload = profile.get(data_type)
        if payload is None:
            continue
        async with AsyncSessionLocal() as session:
            record = SecureVault(
                user_id=UUID(str(user_id)),
                data_type=data_type,
                encrypted_payload=encryptor.encrypt(json.dumps(payload, default=str)),
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            vault_id = record.id
        # process_vault_record opens its own session, decrypts, cleans, persists.
        await process_vault_record(vault_id)


async def ensure_seeded() -> dict:
    """Idempotently seed the demo cohort. Returns a small summary dict."""
    if await _already_seeded():
        logger.info("Database already seeded; skipping bootstrap.")
        return {"seeded": False, "reason": "already_populated"}

    if not MOCK_DATA_PATH.exists():
        logger.warning("Mock data not found at %s; skipping bootstrap seed.", MOCK_DATA_PATH)
        return {"seeded": False, "reason": "no_mock_data"}

    profiles = json.loads(MOCK_DATA_PATH.read_text(encoding="utf-8"))
    logger.info("Seeding %d demo borrowers (encrypt -> vault -> preprocess)...", len(profiles))

    for index, profile in enumerate(profiles, start=1):
        user_id = profile["user_id"]
        try:
            await _ingest_profile(user_id, profile)
            ground_truth = profile.get("_ground_truth")
            if ground_truth:
                await _store_ground_truth(user_id, ground_truth)
        except Exception:
            logger.exception("Failed to seed user %s", user_id)
        if index % 25 == 0:
            logger.info("  seeded %d/%d borrowers", index, len(profiles))

    # Populate econometric (ADF/ECM) features now that base features exist.
    try:
        from models_econometric.ecm_model import run_ecm_pipeline

        async with AsyncSessionLocal() as session:
            ecm_result = await run_ecm_pipeline(session)
        logger.info("ECM pipeline complete during seed: %s", ecm_result)
    except Exception:
        logger.exception("ECM pipeline failed during seed (non-fatal)")

    logger.info("Bootstrap seed complete: %d borrowers ready.", len(profiles))
    return {"seeded": True, "borrowers": len(profiles)}
