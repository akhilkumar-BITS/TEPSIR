from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.api.deps import AdminUser, CurrentUser, DbSession
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserPublic

router = APIRouter(prefix="/users", tags=["users"])


class UserList(BaseModel):
    items: list[UserPublic]
    total: int
    page: int
    page_size: int


@router.get("/me", response_model=UserPublic)
async def read_me(user: CurrentUser) -> UserPublic:
    return UserPublic.model_validate(user)


@router.get("", response_model=UserList)
async def list_users(
    _: AdminUser,
    session: DbSession,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> UserList:
    users, total = await UserRepository(session).list_users(
        offset=(page - 1) * page_size, limit=page_size
    )
    return UserList(
        items=[UserPublic.model_validate(user) for user in users],
        total=total,
        page=page,
        page_size=page_size,
    )
