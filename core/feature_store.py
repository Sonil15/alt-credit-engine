"""Shared helpers for reading and writing ML feature rows."""

import json
from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.db_models import FeatureSeries, MLFeature


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


async def fetch_user_features_wide(session: AsyncSession, user_id: str) -> pd.DataFrame:
    """Return wide feature row for a single user."""
    wide = await fetch_features_wide(session)
    if wide.empty:
        return pd.DataFrame()
    return wide[wide["user_id"].astype(str) == str(user_id)]


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


async def upsert_series(
    session: AsyncSession,
    user_id: UUID | str,
    series_name: str,
    values: list[float],
) -> None:
    """Store or replace a named time series for a user."""
    session.add(
        FeatureSeries(
            user_id=UUID(str(user_id)),
            series_name=series_name,
            values_json=json.dumps(values),
        )
    )


async def fetch_series(session: AsyncSession, user_id: str, series_name: str) -> list[float] | None:
    """Return latest series values for a user, or None if missing."""
    result = await session.execute(
        select(FeatureSeries.values_json, FeatureSeries.created_at)
        .where(FeatureSeries.user_id == UUID(str(user_id)))
        .where(FeatureSeries.series_name == series_name)
        .order_by(FeatureSeries.created_at.desc())
        .limit(1)
    )
    row = result.first()
    if row is None:
        return None
    return json.loads(row[0])


async def fetch_all_series(session: AsyncSession, series_name: str) -> dict[str, list[float]]:
    """Return latest series per user for a given series name."""
    result = await session.execute(
        select(FeatureSeries.user_id, FeatureSeries.values_json, FeatureSeries.created_at)
        .where(FeatureSeries.series_name == series_name)
        .order_by(FeatureSeries.created_at)
    )
    rows = result.all()
    latest: dict[str, list[float]] = {}
    for user_id, values_json, _ in rows:
        latest[str(user_id)] = json.loads(values_json)
    return latest
