"""Create the bootstrap administrator from environment settings.

Usage:
    python -m app.seed
"""

import asyncio
import logging

from app.core.logging import configure_logging
from app.db.session import SessionLocal, engine
from app.services.auth_service import AuthService

logger = logging.getLogger(__name__)


async def seed() -> None:
    async with SessionLocal() as session:
        await AuthService(session).ensure_seed_admin()
    await engine.dispose()


def main() -> None:
    configure_logging()
    asyncio.run(seed())
    logger.info("Seed step finished.")


if __name__ == "__main__":
    main()
