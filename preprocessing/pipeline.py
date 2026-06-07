import json
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import AsyncSessionLocal
from core.security import get_encryptor
from models.db_models import MLFeature, SecureVault
from models.pydantic_schemas import DataType
from preprocessing.clean_cashflow import clean_cashflow
from preprocessing.clean_ecommerce import clean_ecommerce
from preprocessing.clean_geo import clean_geo
from preprocessing.clean_survey import clean_survey
from preprocessing.clean_telecom import clean_telecom

logger = logging.getLogger(__name__)


async def save_features(session: AsyncSession, user_id: UUID, features: dict[str, float]) -> None:
    for feature_name, feature_value in features.items():
        session.add(
            MLFeature(
                user_id=user_id,
                feature_name=feature_name,
                feature_value=float(feature_value),
            )
        )
    await session.commit()


def _extract_records(data_type: DataType, payload: dict) -> list | dict:
    mapping = {
        DataType.TELECOM: "invoices",
        DataType.ECOMMERCE: "orders",
        DataType.GEO: "locations",
        DataType.CASHFLOW: "transactions",
    }
    if data_type == DataType.SURVEY:
        return {k: v for k, v in payload.items() if k != "user_id"}
    key = mapping[data_type]
    return payload.get(key, [])


async def process_vault_record(vault_id: UUID) -> None:
    """Decrypt a vault record, run the appropriate cleaner, and persist ML features."""
    encryptor = get_encryptor()

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(SecureVault).where(SecureVault.id == vault_id))
        record = result.scalar_one_or_none()
        if record is None:
            logger.warning("Vault record %s not found", vault_id)
            return

        try:
            decrypted = encryptor.decrypt(record.encrypted_payload)
            payload = json.loads(decrypted)
            data_type = DataType(record.data_type)
            raw_records = _extract_records(data_type, payload)

            if data_type == DataType.TELECOM:
                features = clean_telecom(raw_records)
            elif data_type == DataType.ECOMMERCE:
                features = clean_ecommerce(raw_records)
            elif data_type == DataType.GEO:
                features = clean_geo(raw_records)
            elif data_type == DataType.CASHFLOW:
                features = clean_cashflow(raw_records)
            elif data_type == DataType.SURVEY:
                features = await clean_survey(raw_records)
            else:
                logger.error("Unsupported data type: %s", record.data_type)
                return

            await save_features(session, record.user_id, features)
            logger.info(
                "Processed vault %s for user %s (%s): %d features",
                vault_id,
                record.user_id,
                record.data_type,
                len(features),
            )
        except Exception:
            logger.exception("Failed to process vault record %s", vault_id)
            await session.rollback()
