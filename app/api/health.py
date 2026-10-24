from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import DbSession
from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def health(session: DbSession) -> dict[str, str]:
    await session.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "environment": settings.environment,
        "embedding_provider": settings.embedding_provider,
    }
