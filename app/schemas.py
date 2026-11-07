"""Request/response models for the REST API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.prompts import PERSONAS

PersonaName = Literal[tuple(PERSONAS)]  # type: ignore[valid-type]


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be blank")
        return v.strip()

    @field_validator("password")
    @classmethod
    def _bcrypt_limit(cls, v: str) -> str:
        if len(v.encode()) > 72:  # bcrypt only uses the first 72 bytes
            raise ValueError("Password is too long (max 72 bytes)")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    email: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class SessionCreate(BaseModel):
    persona: PersonaName = "assistant"
    title: str | None = Field(default=None, max_length=120)


class SessionUpdate(BaseModel):
    persona: PersonaName | None = None
    title: str | None = Field(default=None, min_length=1, max_length=120)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    persona: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    role: str
    content: str
    created_at: datetime


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
