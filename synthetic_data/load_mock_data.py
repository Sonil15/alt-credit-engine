"""Bulk-load mock user profiles into the running Alt-Credit Engine API."""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

MOCK_DATA_PATH = Path(__file__).parent / "mock_data_100_users.json"
DATA_TYPES = ("telecom", "ecommerce", "geo", "cashflow", "survey")
DEFAULT_BASE_URL = "http://localhost:8000"


async def _wait_for_api(client: httpx.AsyncClient, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = await client.get("/health")
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1.0)
    raise RuntimeError(f"API not reachable at {client.base_url} within {timeout}s")


async def _ingest_user(
    client: httpx.AsyncClient,
    profile: dict,
    *,
    delay_between_types: float,
) -> tuple[str, int, int]:
    user_id = profile["user_id"]
    success = 0
    failed = 0

    for data_type in DATA_TYPES:
        payload = profile[data_type]
        try:
            response = await client.post(f"/ingest/{data_type}", json=payload)
            response.raise_for_status()
            success += 1
        except httpx.HTTPError as exc:
            failed += 1
            logger.error("Failed ingest %s for user %s: %s", data_type, user_id, exc)
        if delay_between_types:
            await asyncio.sleep(delay_between_types)

    return user_id, success, failed


async def load_profiles(
    base_url: str,
    *,
    limit: int | None = None,
    delay_between_types: float = 0.0,
    delay_between_users: float = 0.05,
) -> dict:
    if not MOCK_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Mock data not found at {MOCK_DATA_PATH}. Run generate_raw_mock.py first."
        )

    profiles = json.loads(MOCK_DATA_PATH.read_text(encoding="utf-8"))
    if limit is not None:
        profiles = profiles[:limit]

    totals = {"users": len(profiles), "success": 0, "failed": 0, "user_ids": []}

    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        await _wait_for_api(client)

        for index, profile in enumerate(profiles, start=1):
            user_id, success, failed = await _ingest_user(
                client,
                profile,
                delay_between_types=delay_between_types,
            )
            totals["success"] += success
            totals["failed"] += failed
            totals["user_ids"].append(user_id)
            logger.info(
                "Loaded user %d/%d (%s): %d ok, %d failed",
                index,
                len(profiles),
                user_id,
                success,
                failed,
            )
            if delay_between_users:
                await asyncio.sleep(delay_between_users)

    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Load mock data into Alt-Credit Engine API")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--limit", type=int, default=None, help="Max users to load")
    parser.add_argument(
        "--delay-types",
        type=float,
        default=0.0,
        help="Seconds to wait between data types per user",
    )
    parser.add_argument(
        "--delay-users",
        type=float,
        default=0.05,
        help="Seconds to wait between users",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        totals = asyncio.run(
            load_profiles(
                args.base_url,
                limit=args.limit,
                delay_between_types=args.delay_types,
                delay_between_users=args.delay_users,
            )
        )
    except Exception as exc:
        logger.error("Bulk load failed: %s", exc)
        sys.exit(1)

    print(
        f"Done: {totals['users']} users, "
        f"{totals['success']} ingestions ok, {totals['failed']} failed"
    )
    if totals["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
