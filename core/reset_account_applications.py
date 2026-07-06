"""Remove loan-application artifacts for registered borrower accounts.

Only ``user_id`` values present in ``borrower_accounts`` are touched: the
bundled synthetic demo cohort (seeded borrowers with no login account) is left
intact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.db_models import (
    ApplicationIntake,
    AuditLog,
    BorrowerAccount,
    DecisionLetter,
    FeatureSeries,
    MLFeature,
    ScoreDecision,
    SecureVault,
)

logger = logging.getLogger(__name__)

# Application pipeline tables keyed on user_id. Accounts and auth tokens are kept.
_TABLES: tuple[tuple[str, type], ...] = (
    ("application_intake", ApplicationIntake),
    ("secure_vault", SecureVault),
    ("ml_features", MLFeature),
    ("feature_series", FeatureSeries),
    ("score_decisions", ScoreDecision),
    ("decision_letters", DecisionLetter),
    ("audit_logs", AuditLog),
)


@dataclass
class ResetSummary:
    account_count: int = 0
    login_ids: list[str] = field(default_factory=list)
    deleted: dict[str, int] = field(default_factory=dict)
    dry_run: bool = False


async def fetch_account_user_ids(session: AsyncSession) -> list[tuple[UUID, str]]:
    rows = (
        await session.execute(
            select(BorrowerAccount.user_id, BorrowerAccount.login_id).order_by(
                BorrowerAccount.login_id
            )
        )
    ).all()
    return [(uid, login_id) for uid, login_id in rows]


async def reset_account_applications(
    session: AsyncSession,
    *,
    dry_run: bool = False,
) -> ResetSummary:
    """Delete application data for every registered borrower account."""
    accounts = await fetch_account_user_ids(session)
    summary = ResetSummary(
        account_count=len(accounts),
        login_ids=[login_id for _, login_id in accounts],
        dry_run=dry_run,
    )

    if not accounts:
        return summary

    user_ids = [uid for uid, _ in accounts]

    for table_name, model in _TABLES:
        if dry_run:
            count = await session.scalar(
                select(func.count()).select_from(model).where(model.user_id.in_(user_ids))
            )
            summary.deleted[table_name] = int(count or 0)
        else:
            result = await session.execute(delete(model).where(model.user_id.in_(user_ids)))
            summary.deleted[table_name] = result.rowcount or 0

    if not dry_run:
        await session.commit()
        logger.info(
            "Reset application data for %d account(s): %s",
            summary.account_count,
            ", ".join(summary.login_ids) or "(none)",
        )

    return summary
