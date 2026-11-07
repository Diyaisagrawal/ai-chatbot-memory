"""Request/response models for the REST API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.prompts import PERSONAS

PersonaName = Literal[tuple(PERSONAS)]  # type: ignore[valid-type]


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=64, pattern=r"^[\w.\- ]+$")


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    username: str
    created_at: datetime


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
