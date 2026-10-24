import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    ConflictError,
    PermissionDeniedError,
    ValidationError,
)
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import Role, User
from app.repositories.user_repository import UserRepository
from app.schemas.auth import LoginRequest, Token
from app.schemas.user import UserCreate

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.users = UserRepository(session)

    async def register(self, payload: UserCreate) -> User:
        if len(payload.password) < settings.password_min_length:
            raise ValidationError(
                f"Password must be at least {settings.password_min_length} characters."
            )
        existing = await self.users.get_by_email(payload.email)
        if existing is not None:
            raise ConflictError("An account with this email already exists.")

        user = await self.users.create(
            email=payload.email,
            full_name=payload.full_name.strip(),
            hashed_password=hash_password(payload.password),
            role=Role.USER,
        )
        await self.session.commit()
        logger.info("Registered user %s", user.email)
        return user

    async def login(self, payload: LoginRequest) -> Token:
        user = await self.users.get_by_email(payload.email)
        if user is None or not verify_password(payload.password, user.hashed_password):
            raise AuthenticationError("Incorrect email or password.")
        if not user.is_active:
            raise PermissionDeniedError("This account is disabled.")
        return self.issue_token(user)

    async def authenticate(self, email: str, password: str) -> User:
        """Credential check used by the OAuth2 form endpoint (no token issued)."""
        user = await self.users.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            raise AuthenticationError("Incorrect email or password.")
        if not user.is_active:
            raise PermissionDeniedError("This account is disabled.")
        return user

    @staticmethod
    def issue_token(user: User) -> Token:
        token = create_access_token(subject=str(user.id), role=user.role.value)
        return Token(
            access_token=token,
            expires_in=settings.access_token_expire_minutes * 60,
        )

    async def ensure_seed_admin(self) -> None:
        email = settings.seed_admin_email
        password = settings.seed_admin_password
        if not email or not password:
            return
        if await self.users.get_by_email(email) is not None:
            return
        await self.users.create(
            email=email,
            full_name="Administrator",
            hashed_password=hash_password(password),
            role=Role.ADMIN,
        )
        await self.session.commit()
        logger.info("Seeded administrator account %s", email)
