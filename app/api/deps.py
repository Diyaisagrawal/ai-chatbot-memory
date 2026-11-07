"""FastAPI dependencies: DB session, current user, chat service, session ownership."""

from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session as DBSession

from app.chain import ChatService
from app.db import ChatSession, User


def get_db(request: Request) -> Iterator[DBSession]:
    yield from request.app.state.db.session()


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_current_user(
    x_user_id: str = Header(..., description="ID returned by POST /api/users"),
    db: DBSession = Depends(get_db),
) -> User:
    user = db.get(User, x_user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown user. Create one via POST /api/users.")
    return user


def get_owned_session(session_id: str, user: User, db: DBSession) -> ChatSession:
    chat = db.get(ChatSession, session_id)
    # Same 404 whether the session doesn't exist or belongs to someone else,
    # so session IDs can't be probed across users.
    if chat is None or chat.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return chat
