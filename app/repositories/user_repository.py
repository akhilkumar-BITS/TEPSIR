from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Role, User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return await self.session.get(User, user_id)

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(func.lower(User.email) == email.lower())
        return await self.session.scalar(stmt)

    async def create(
        self,
        *,
        email: str,
        full_name: str,
        hashed_password: str,
        role: Role = Role.USER,
    ) -> User:
        user = User(
            email=email.lower(),
            full_name=full_name,
            hashed_password=hashed_password,
            role=role,
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def list_users(self, *, offset: int, limit: int) -> tuple[list[User], int]:
        total = await self.session.scalar(select(func.count()).select_from(User)) or 0
        stmt = select(User).order_by(User.created_at.desc()).offset(offset).limit(limit)
        users = list((await self.session.scalars(stmt)).all())
        return users, total
