"""Shared FastAPI dependencies: identity, authorization and app singletons."""

import uuid
from typing import Annotated

import jwt
from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import EmbeddingCache
from app.core.config import settings
from app.core.exceptions import AuthenticationError, PermissionDeniedError
from app.core.security import decode_access_token
from app.db.models import Role, User
from app.db.session import get_db
from app.rag.embeddings import EmbeddingProvider
from app.repositories.user_repository import UserRepository
from app.storage.base import ObjectStore

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api_v1_prefix}/auth/token", auto_error=False
)


def get_object_store(request: Request) -> ObjectStore:
    return request.app.state.object_store


def get_embedder(request: Request) -> EmbeddingProvider:
    return request.app.state.embedder


def get_embedding_cache(request: Request) -> EmbeddingCache:
    return request.app.state.embedding_cache


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    if not token:
        raise AuthenticationError("Missing authentication credentials.")
    try:
        payload = decode_access_token(token)
        user_id = uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError) as exc:
        raise AuthenticationError("Invalid or expired access token.") from exc

    user = await UserRepository(session).get_by_id(user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("Account not found or disabled.")
    return user


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role is not Role.ADMIN:
        raise PermissionDeniedError("Administrator role required.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(require_admin)]
DbSession = Annotated[AsyncSession, Depends(get_db)]
