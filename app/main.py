"""Application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.router import api_router
from app.core.cache import EmbeddingCache
from app.core.config import settings
from app.core.exceptions import install_exception_handlers
from app.core.logging import configure_logging
from app.db.session import SessionLocal, engine
from app.rag.embeddings import build_embedding_provider
from app.rag.llm import build_llm_client
from app.services.auth_service import AuthService
from app.storage.base import build_object_store

logger = logging.getLogger(__name__)

_INSECURE_SECRET = "dev-only-change-me"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.debug)

    if settings.environment == "production" and settings.secret_key == _INSECURE_SECRET:
        raise RuntimeError(
            "SECRET_KEY must be set to a strong random value when ENVIRONMENT=production."
        )

    # Singletons shared by every request. The embedding model in particular is
    # expensive to construct, so it is created once per process.
    store = build_object_store()
    await store.ensure_ready()
    app.state.object_store = store
    app.state.embedder = build_embedding_provider()
    app.state.embedding_cache = EmbeddingCache()
    # None when synthesis is disabled or DEEPSEEK_API_KEY is unset.
    app.state.llm = build_llm_client()

    async with SessionLocal() as session:
        await AuthService(session).ensure_seed_admin()

    logger.info(
        "TEPSIR API ready (env=%s, embedder=%s, answer_synthesis=%s)",
        settings.environment,
        app.state.embedder.name,
        app.state.llm is not None,
    )
    try:
        yield
    finally:
        if app.state.llm is not None:
            await app.state.llm.aclose()
        await app.state.embedding_cache.close()
        await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.project_name,
        description=(
            "Travel and Expense Policy Search and Information Retrieval. "
            "Administrators publish policy documents; users search them with "
            "semantic retrieval over chunked, embedded policy text."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": settings.project_name,
            "version": "0.1.0",
            "docs": "/docs",
            "api": settings.api_v1_prefix,
        }

    return app


app = create_app()
