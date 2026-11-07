"""REST endpoints for users, sessions, history and chat."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.api.deps import get_chat_service, get_current_user, get_db, get_owned_session
from app.chain import ChatService
from app.db import ChatSession, Message, User
from app.prompts import PERSONAS
from app.schemas import (
    ChatRequest,
    ChatResponse,
    MessageOut,
    SessionCreate,
    SessionOut,
    SessionUpdate,
    UserCreate,
    UserOut,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# --- Users ---------------------------------------------------------------------

@router.post("/users", response_model=UserOut, tags=["users"])
def create_or_get_user(body: UserCreate, response: Response, db: DBSession = Depends(get_db)):
    """Idempotent sign-in: returns the existing user for a username or creates one."""
    username = body.username.strip()
    user = db.scalar(select(User).where(User.username == username))
    if user is None:
        user = User(username=username)
        db.add(user)
        db.commit()
        response.status_code = status.HTTP_201_CREATED
    return user


@router.get("/users/me", response_model=UserOut, tags=["users"])
def me(user: User = Depends(get_current_user)):
    return user


# --- Sessions ------------------------------------------------------------------

@router.get("/personas", tags=["sessions"])
def list_personas() -> dict[str, str]:
    return PERSONAS


@router.post("/sessions", response_model=SessionOut, status_code=201, tags=["sessions"])
def create_session(
    body: SessionCreate, user: User = Depends(get_current_user), db: DBSession = Depends(get_db)
):
    chat = ChatSession(user_id=user.id, persona=body.persona, title=body.title or "New chat")
    db.add(chat)
    db.commit()
    return chat


@router.get("/sessions", response_model=list[SessionOut], tags=["sessions"])
def list_sessions(user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    return db.scalars(
        select(ChatSession).where(ChatSession.user_id == user.id).order_by(ChatSession.updated_at.desc())
    ).all()


@router.patch("/sessions/{session_id}", response_model=SessionOut, tags=["sessions"])
def update_session(
    session_id: str,
    body: SessionUpdate,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    chat = get_owned_session(session_id, user, db)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(chat, field, value)
    db.commit()
    return chat


@router.delete("/sessions/{session_id}", status_code=204, tags=["sessions"])
def delete_session(session_id: str, user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    db.delete(get_owned_session(session_id, user, db))
    db.commit()


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut], tags=["sessions"])
def get_messages(session_id: str, user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    get_owned_session(session_id, user, db)
    return db.scalars(select(Message).where(Message.session_id == session_id).order_by(Message.id)).all()


@router.delete("/sessions/{session_id}/messages", status_code=204, tags=["sessions"])
def clear_messages(
    session_id: str,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
):
    get_owned_session(session_id, user, db)
    service.get_history(session_id).clear()


# --- Chat ----------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
):
    chat_session = get_owned_session(body.session_id, user, db)
    try:
        reply = await service.respond(chat_session.id, body.message, user.username, chat_session.persona)
    except Exception as exc:  # upstream LLM errors -> 502, not a stack trace
        log.exception("LLM call failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM error: {exc}") from exc
    return ChatResponse(session_id=chat_session.id, reply=reply)


@router.post("/chat/stream", tags=["chat"])
async def chat_stream(
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
):
    """Server-Sent Events: `data: {"token": "..."}` per chunk, then `data: {"done": true}`."""
    chat_session = get_owned_session(body.session_id, user, db)
    session_id, persona, username = chat_session.id, chat_session.persona, user.username

    async def events():
        try:
            async for token in service.stream(session_id, body.message, username, persona):
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            log.exception("LLM stream failed")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
