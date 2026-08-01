"""Zero-friction startup seeding.

Populates an empty database with the bundled mock borrower cohort so the demo is
fully interactive the instant the server boots, no Docker, no manual
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
import random
import io
import base64
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from sqlalchemy import func, select

from core.database import AsyncSessionLocal
from core.feature_store import upsert_feature
from core.security import get_encryptor
from models.db_models import MLFeature, SecureVault, Captcha
from preprocessing.pipeline import process_vault_record

logger = logging.getLogger(__name__)


def generate_bad_captcha(text: str) -> str:
    """Generates a distorted, noisy visual CAPTCHA image and returns it as base64."""
    w, h = 180, 60
    # Dark slate background matching dashboard UI
    image = Image.new("RGB", (w, h), (30, 41, 59))
    draw = ImageDraw.Draw(image)

    # Try loading common system fonts
    font = None
    for font_path in [
        "/System/Library/Fonts/Geneva.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/System/Library/Fonts/SFNSMono.ttf",
        # Linux/container (Debian fonts-dejavu-core) - keeps the CAPTCHA looking
        # the same when deployed as it does on macOS locally.
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, 32)
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()

    # Draw grid/noise lines in background
    for _ in range(12):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = random.randint(0, w)
        y2 = random.randint(0, h)
        draw.line(
            (x1, y1, x2, y2),
            fill=(random.randint(60, 120), random.randint(60, 120), random.randint(60, 120)),
            width=2,
        )

    # Draw point noise (dots)
    for _ in range(400):
        draw.point(
            (random.randint(0, w), random.randint(0, h)),
            fill=(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)),
        )

    # Draw characters one by one with individual skew/rotation
    char_w = w / len(text)
    for i, char in enumerate(text):
        char_img = Image.new("RGBA", (40, 50), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255), 255)
        char_draw.text((5, 5), char, font=font, fill=char_color)
        char_rotated = char_img.rotate(random.randint(-30, 30), expand=True, resample=Image.Resampling.BILINEAR)
        char_x = int(10 + i * char_w + random.randint(-5, 5))
        char_y = int(random.randint(2, 10))
        image.paste(char_rotated, (char_x, char_y), char_rotated)

    # Add foreground noise lines crossing characters
    for _ in range(5):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = random.randint(0, w)
        y2 = random.randint(0, h)
        draw.line(
            (x1, y1, x2, y2),
            fill=(random.randint(200, 255), random.randint(100, 200), random.randint(100, 200)),
            width=2,
        )

    # Apply a slight blur
    image = image.filter(ImageFilter.GaussianBlur(radius=0.8))

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def ensure_captchas_seeded() -> None:
    """Ensures at least 10 visual captcha challenges are stored in the database."""
    async with AsyncSessionLocal() as session:
        count = await session.scalar(select(func.count()).select_from(Captcha))
        if count >= 10:
            return

        to_seed = 10 - count
        logger.info("Seeding %d visual captchas to database...", to_seed)

        # Exclude easily confused characters like I, O, 0, 1
        possible_chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        for _ in range(to_seed):
            label = "".join(random.choice(possible_chars) for _ in range(5))
            image_b64 = generate_bad_captcha(label)
            record = Captcha(image_base64=image_b64, label=label)
            session.add(record)

        await session.commit()


from core.demographics import GENDER_CODES, GEOGRAPHY_CODES, PROTECTED_GROUP_CODES

MOCK_DATA_PATH = Path(__file__).resolve().parent.parent / "synthetic_data" / "mock_data_100_users.json"
DATA_TYPES = ("telecom", "ecommerce", "geo", "cashflow", "survey")


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
