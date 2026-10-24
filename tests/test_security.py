import jwt
import pytest

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hash_round_trip():
    hashed = hash_password("a-strong-password")
    assert hashed != "a-strong-password"
    assert verify_password("a-strong-password", hashed)
    assert not verify_password("wrong-password", hashed)


def test_password_longer_than_bcrypt_limit_is_supported():
    password = "x" * 200
    assert verify_password(password, hash_password(password))


def test_token_carries_subject_and_role():
    token = create_access_token(subject="user-1", role="ROLE_USER")
    payload = decode_access_token(token)
    assert payload["sub"] == "user-1"
    assert payload["role"] == "ROLE_USER"


def test_tampered_token_is_rejected():
    token = create_access_token(subject="user-1", role="ROLE_USER")
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(token + "tampered")
