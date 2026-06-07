"""Shared helpers for reading and writing ML feature rows."""

from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.db_models import MLFeature


async def fetch_features_long(session: AsyncSession) -> pd.DataFrame:
    """Return all ml_features rows as a long-format DataFrame."""
    result = await session.execute(
        select(MLFeature.user_id, MLFeature.feature_name, MLFeature.feature_value, MLFeature.created_at)
    )
    rows = result.all()
    if not rows:
        return pd.DataFrame(columns=["user_id", "feature_name", "feature_value", "created_at"])

    df = pd.DataFrame(rows, columns=["user_id", "feature_name", "feature_value", "created_at"])
    df["user_id"] = df["user_id"].astype(str)
    return df


async def fetch_features_wide(session: AsyncSession) -> pd.DataFrame:
    """Pivot ml_features to one row per user (latest value per feature)."""
    long_df = await fetch_features_long(session)
    if long_df.empty:
        return pd.DataFrame()

    latest = (
        long_df.sort_values("created_at")
        .groupby(["user_id", "feature_name"], as_index=False)
        .last()
    )
    wide = latest.pivot(index="user_id", columns="feature_name", values="feature_value")
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


async def upsert_feature(session: AsyncSession, user_id: UUID | str, feature_name: str, value: float) -> None:
    """Append a feature row (latest wins on read)."""
    session.add(
        MLFeature(
            user_id=UUID(str(user_id)),
            feature_name=feature_name,
            feature_value=float(value),
        )
    )


async def upsert_features_batch(
    session: AsyncSession,
    user_features: dict[str, dict[str, float]],
) -> None:
    for user_id, features in user_features.items():
        for feature_name, value in features.items():
            await upsert_feature(session, user_id, feature_name, value)
    await session.commit()
