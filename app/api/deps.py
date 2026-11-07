"""FastAPI dependencies: DB session, authenticated user, chat service, session ownership."""

from collections.abc import Iterator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session as DBSession

from app.chain import ChatService
from app.db import ChatSession, User
from app.security import TokenService

bearer = HTTPBearer(auto_error=False, description="Token from /api/auth/login or /api/auth/register")


def get_db(request: Request) -> Iterator[DBSession]:
    yield from request.app.state.db.session()


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_token_service(request: Request) -> TokenService:
    return request.app.state.tokens


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    tokens: TokenService = Depends(get_token_service),
    db: DBSession = Depends(get_db),
) -> User:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "Not signed in or session expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None:
        raise unauthorized
    user_id = tokens.user_id_from(creds.credentials)
    user = db.get(User, user_id) if user_id else None
    if user is None:
        raise unauthorized
    return user


def get_owned_session(session_id: str, user: User, db: DBSession) -> ChatSession:
    chat = db.get(ChatSession, session_id)
    # Same 404 whether the session doesn't exist or belongs to someone else,
    # so session IDs can't be probed across users.
    if chat is None or chat.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return chat
