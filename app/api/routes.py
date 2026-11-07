"""REST endpoints for auth, sessions, history and chat."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DBSession

from app.api.deps import (
    get_chat_service,
    get_current_user,
    get_db,
    get_owned_session,
    get_token_service,
)
from app.chain import ChatService
from app.db import ChatSession, Message, User
from app.prompts import PERSONAS
from app.schemas import (
    ChatRequest,
    ChatResponse,
    LoginRequest,
    MessageOut,
    RegisterRequest,
    SessionCreate,
    SessionOut,
    SessionUpdate,
    TokenResponse,
    UserOut,
)
from app.security import TokenService, hash_password, verify_password

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# Compared against when the email doesn't exist, so response time doesn't reveal
# which emails are registered.
_DUMMY_HASH = hash_password("not-a-real-password")


def llm_error_message(exc: Exception) -> str:
    text = str(exc)
    if "not supported by any provider" in text or "model_not_supported" in text:
        return (
            "None of the configured models are available from the Hugging Face providers you "
            "have enabled. Set HF_MODEL_ID in .env to a model listed at "
            "https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation "
            "or enable more providers at https://huggingface.co/settings/inference-providers"
        )
    if "401" in text or "403" in text or "Unauthorized" in text:
        return "Hugging Face rejected the token. Check HF_TOKEN has the 'Make calls to Inference Providers' permission."
    return f"LLM error: {text}"


# --- Auth ----------------------------------------------------------------------

@router.post("/auth/register", response_model=TokenResponse, status_code=201, tags=["auth"])
def register(
    body: RegisterRequest,
    db: DBSession = Depends(get_db),
    tokens: TokenService = Depends(get_token_service),
):
    email = body.email.lower()
    if db.scalar(select(User).where(func.lower(User.email) == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user = User(email=email, name=body.name, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return TokenResponse(access_token=tokens.create(user.id), user=UserOut.model_validate(user))


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(
    body: LoginRequest,
    db: DBSession = Depends(get_db),
    tokens: TokenService = Depends(get_token_service),
):
    user = db.scalar(select(User).where(func.lower(User.email) == body.email.lower()))
    if not verify_password(body.password, user.password_hash if user else _DUMMY_HASH) or user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return TokenResponse(access_token=tokens.create(user.id), user=UserOut.model_validate(user))


@router.get("/auth/me", response_model=UserOut, tags=["auth"])
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
        reply = await service.respond(chat_session.id, body.message, user.name, chat_session.persona)
    except Exception as exc:  # upstream LLM errors -> 502, not a stack trace
        log.exception("LLM call failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, llm_error_message(exc)) from exc
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
    session_id, persona, username = chat_session.id, chat_session.persona, user.name

    async def events():
        try:
            async for token in service.stream(session_id, body.message, username, persona):
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            log.exception("LLM stream failed")
            yield f"data: {json.dumps({'error': llm_error_message(exc)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
