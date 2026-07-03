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


class BorrowerAccount(Base):
    """A borrower's login account.

    The account's ``user_id`` is the borrower's stable identity across the whole
    pipeline (vault, features, decisions), so once a borrower logs in every
    application they submit ties back to the same account. Passwords are stored
    only as a PBKDF2 hash + per-account salt — never in plaintext.
    """

    __tablename__ = "borrower_accounts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), unique=True, index=True, nullable=False, default=uuid.uuid4
    )
    login_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    password_salt: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class AuthToken(Base):
    """An opaque bearer token issued to a logged-in borrower (server-side session)."""

    __tablename__ = "auth_tokens"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AuditLog(Base):
    """Audit trail for manual decision overrides by loan officers."""

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True, nullable=False)
    officer_id: Mapped[str] = mapped_column(String(100), nullable=False)
    original_decision: Mapped[str] = mapped_column(String(20), nullable=False)
    new_decision: Mapped[str] = mapped_column(String(20), nullable=False)
    justification: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
