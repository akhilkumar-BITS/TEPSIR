import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.db.models import Role


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=200)
    password: str = Field(min_length=8, max_length=128)


class UserPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    # Plain string on the way out: stored rows may predate stricter
    # validation, and a response model must never fail on existing data.
    email: str
    full_name: str
    role: Role
    is_active: bool
    created_at: datetime
