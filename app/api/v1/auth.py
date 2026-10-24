from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import DbSession
from app.schemas.auth import LoginRequest, Token
from app.schemas.user import UserCreate, UserPublic
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, session: DbSession) -> UserPublic:
    """Self-service sign-up. New accounts always receive ROLE_USER."""
    user = await AuthService(session).register(payload)
    return UserPublic.model_validate(user)


@router.post("/login", response_model=Token)
async def login(payload: LoginRequest, session: DbSession) -> Token:
    return await AuthService(session).login(payload)


@router.post("/token", response_model=Token, include_in_schema=False)
async def token(
    form: Annotated[OAuth2PasswordRequestForm, Depends()], session: DbSession
) -> Token:
    """OAuth2 password flow, kept so the OpenAPI docs can authenticate."""
    user = await AuthService(session).authenticate(form.username, form.password)
    return AuthService.issue_token(user)
