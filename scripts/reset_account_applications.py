"""Reset loan applications for registered borrower accounts.

Removes intake, vault, features, scores, letters, and audit rows for every
``user_id`` in ``borrower_accounts``. The synthetic demo cohort (no login
account) is untouched.

Run:
  USE_SQLITE=true PYTHONPATH=. .venv/bin/python -m scripts.reset_account_applications

Dry run (counts only, no deletes):
  USE_SQLITE=true PYTHONPATH=. .venv/bin/python -m scripts.reset_account_applications --dry-run

If the API server is running, restart it afterward so in-memory consent and
assessment session state is cleared too.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from core.database import AsyncSessionLocal
from core.reset_account_applications import reset_account_applications

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _print_summary(summary) -> None:
    mode = "would delete" if summary.dry_run else "deleted"
    if summary.account_count == 0:
        print("No borrower accounts found, nothing to reset.")
        return

    print(f"Borrower accounts ({summary.account_count}): {', '.join(summary.login_ids)}")
    print(f"Rows {mode}:")
    for table, count in summary.deleted.items():
        print(f"  {table}: {count}")
    total = sum(summary.deleted.values())
    print(f"  total: {total}")

    if not summary.dry_run:
        print(
            "\nAccounts kept. Synthetic seed data unchanged. "
            "Restart the API server to clear in-memory consent/assessment state."
        )


async def _main(dry_run: bool) -> int:
    async with AsyncSessionLocal() as session:
        summary = await reset_account_applications(session, dry_run=dry_run)
    _print_summary(summary)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove loan application data for all registered borrower accounts "
            "(does not delete accounts or synthetic seed data)."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report row counts without deleting anything.",
    )
    args = parser.parse_args()

    try:
        raise SystemExit(asyncio.run(_main(args.dry_run)))
    except Exception:
        logger.exception("Reset failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
