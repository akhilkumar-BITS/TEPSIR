"""Password hashing and JSON Web Token helpers."""

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from app.core.config import settings

# bcrypt refuses inputs longer than 72 bytes. Passwords are truncated at that
# boundary so a long passphrase does not raise at sign-up time.
_BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    encoded = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(encoded, hashed_password.encode("utf-8"))
    except ValueError:
        # Malformed hash in the database - treat as a failed login.
        return False


def create_access_token(subject: str, role: str, expires_minutes: int | None = None) -> str:
    ttl = expires_minutes if expires_minutes is not None else settings.access_token_expire_minutes
    expire = datetime.now(UTC) + timedelta(minutes=ttl)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "iat": datetime.now(UTC),
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and verify a token. Raises ``jwt.PyJWTError`` when invalid."""
    return jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
