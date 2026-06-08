import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, LargeBinary, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class SecureVault(Base):
    __tablename__ = "secure_vault"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True, nullable=False)
    data_type: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    encrypted_payload: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class MLFeature(Base):
    __tablename__ = "ml_features"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True, nullable=False)
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class FeatureSeries(Base):
    """Time-series feature blobs (e.g. monthly net cashflow) for econometric models."""

    __tablename__ = "feature_series"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True, nullable=False)
    series_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    values_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class ScoreDecision(Base):
    """Audit trail for every credit scoring decision."""

    __tablename__ = "score_decisions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True, nullable=False)
    credit_score: Mapped[int] = mapped_column(Integer, nullable=False)
    probability_of_default: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    auto_reject: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason_codes_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
