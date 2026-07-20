"""One-off: wipe stored CAPTCHAs so they regenerate with the current font.

The visual CAPTCHAs are pre-rendered to base64 and stored in the DB, which now
lives on a persistent volume. After changing the CAPTCHA font, the old images
are still cached there — run this once on the server to clear and re-render them:

    fly ssh console -a alt-credit-demo -C "python -m scripts.reseed_captchas"
"""

import asyncio

from sqlalchemy import delete

from core.bootstrap import ensure_captchas_seeded
from core.database import AsyncSessionLocal
from models.db_models import Captcha


async def main() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(delete(Captcha))
        await session.commit()
        print(f"Deleted {result.rowcount} stored CAPTCHA(s).")
    await ensure_captchas_seeded()
    print("Regenerated CAPTCHAs with the current font.")


if __name__ == "__main__":
    asyncio.run(main())
