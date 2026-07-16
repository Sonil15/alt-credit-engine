import uuid
from datetime import datetime, timezone

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
    # Borrower intent captured at onboarding + the post-gate lending outcome.
    # `decision` stays the model's call; `final_outcome` is what the borrower is told.
    requested_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    loan_purpose: Mapped[str | None] = mapped_column(String(50), nullable=True)
    final_outcome: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class ApplicationIntake(Base):
    """Borrower-declared intent captured at onboarding (before consent).

    One row per submission, latest-wins per ``user_id``. The raw free-text
    business description is NOT stored here, it goes encrypted into
    ``SecureVault`` (data_type="intake"); this table holds only the
    borrower-confirmed structured fields the pipeline may read.
    """

    __tablename__ = "application_intake"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True, nullable=False)
    cohort: Mapped[str] = mapped_column(String(30), nullable=False)
    loan_purpose: Mapped[str] = mapped_column(String(50), nullable=False)
    loan_purpose_other_text: Mapped[str | None] = mapped_column(String(300), nullable=True)
    requested_amount: Mapped[float] = mapped_column(Float, nullable=False)
    business_profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_method: Mapped[str] = mapped_column(String(20), nullable=False, default="none")
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Python-side default keeps microsecond precision so "latest wins" is
    # unambiguous even for re-submissions within the same second (SQLite's
    # server-side now() is second-granular).
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class BorrowerAccount(Base):
    """A borrower's login account.

    The account's ``user_id`` is the borrower's stable identity across the whole
    pipeline (vault, features, decisions), so once a borrower logs in every
    application they submit ties back to the same account. Passwords are stored
    only as a PBKDF2 hash + per-account salt. Never in plaintext.
    """

    __tablename__ = "borrower_accounts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), unique=True, index=True, nullable=False, default=uuid.uuid4
    )
    login_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    password_salt: Mapped[str] = mapped_column(String(64), nullable=False)
    cibil_score: Mapped[int | None] = mapped_column(Integer, nullable=True, default=-1)
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


class DecisionLetter(Base):
    """A borrower-facing decision letter and its human sign-off state.

    One active letter per borrower (latest-wins, keyed on ``user_id``). The letter
    freezes the decision *as made*, outcome, reason codes, score, model version, and
    date, so it is not silently re-rendered when the borrower re-scores later. An
    APPROVE outcome is issued automatically (``status='issued'`` with no officer). A
    REJECT/REVIEW outcome is drafted as ``status='pending_review'`` and only becomes
    ``issued`` after a loan officer reviews and signs, at which point ``officer_id``
    and ``signed_at`` are stamped (the human-accountability record; the free-text
    justification of any decision change still lives in ``audit_logs``).
    """

    __tablename__ = "decision_letters"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), unique=True, index=True, nullable=False)
    # 'pending_review' (awaiting officer) or 'issued' (available to the borrower).
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending_review")
    outcome: Mapped[str] = mapped_column(String(20), nullable=False)
    credit_score: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    # Frozen reason phrases shown to the borrower, as English labels (translated at
    # render time). JSON-encoded list of strings.
    reason_codes_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Officer sign-off, null until a human signs (always null for auto-issued approvals).
    officer_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    signed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


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


class Captcha(Base):
    """Stores generated visual CAPTCHA challenges."""

    __tablename__ = "captchas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_base64: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(String(20), nullable=True)

