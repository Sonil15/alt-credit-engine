import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_api_key
from core.database import get_db
from core.feature_store import upsert_feature
from core.security import get_encryptor
from models.db_models import SecureVault
from models.pydantic_schemas import (
    DataType,
    GroundTruthPayload,
    IngestionResponse,
    validate_payload,
)
from preprocessing.pipeline import process_vault_record

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/{data_type}", response_model=IngestionResponse)
async def ingest_data(
    data_type: DataType,
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> IngestionResponse:
    """Accept raw payload, encrypt immediately, store in secure_vault, and schedule preprocessing."""
    try:
        validated = validate_payload(data_type, payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid payload for {data_type.value}: {exc}") from exc

    user_id: UUID = validated.user_id
    encryptor = get_encryptor()
    encrypted = encryptor.encrypt(json.dumps(payload, default=str))

    vault_record = SecureVault(
        user_id=user_id,
        data_type=data_type.value,
        encrypted_payload=encrypted,
    )
    db.add(vault_record)
    await db.commit()
    await db.refresh(vault_record)

    background_tasks.add_task(process_vault_record, vault_record.id)

    return IngestionResponse(
        vault_id=vault_record.id,
        user_id=user_id,
        data_type=data_type.value,
    )


@router.post("/ground_truth", response_model=IngestionResponse, dependencies=[Depends(require_api_key)])
async def ingest_ground_truth(
    payload: GroundTruthPayload,
    db: AsyncSession = Depends(get_db),
) -> IngestionResponse:
    """
    Store generative ground-truth labels and fairness metadata.
    Used only for model training — never exposed as model inputs at scoring time.
    """
    await upsert_feature(db, payload.user_id, "default_label", float(payload.default_label))

    group_map = {"general": 0, "obc": 1, "sc": 2, "st": 3, "minority": 4}
    await upsert_feature(
        db,
        payload.user_id,
        "protected_group_code",
        float(group_map.get(payload.protected_group, 0)),
    )
    await upsert_feature(
        db,
        payload.user_id,
        "borrower_type",
        1.0 if payload.borrower_type == "msme" else 0.0,
    )
    await db.commit()

    return IngestionResponse(
        user_id=payload.user_id,
        message="Ground truth labels stored for training and fairness monitoring.",
    )
