"""Persistent, per-session LangChain message history backed by SQLAlchemy.

Implements LangChain's `BaseChatMessageHistory`, so it plugs directly into
`RunnableWithMessageHistory`. Every message is stored durably; the chain
decides how much of it to put in the prompt (see `app/chain.py`).
"""

from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import delete, select
from sqlalchemy.orm import sessionmaker

from app.db import ChatSession, Message, _now

_TO_LC = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}


class SQLChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, session_factory: sessionmaker):
        self.session_id = session_id
        self._sf = session_factory

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        with self._sf() as db:
            rows = db.scalars(
                select(Message).where(Message.session_id == self.session_id).order_by(Message.id)
            ).all()
            return [_TO_LC[r.role](content=r.content) for r in rows if r.role in _TO_LC]

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        with self._sf() as db:
            for m in messages:
                content = m.content if isinstance(m.content, str) else str(m.content)
                db.add(Message(session_id=self.session_id, role=m.type, content=content))
            chat = db.get(ChatSession, self.session_id)
            if chat is not None:
                # Auto-title a fresh session from the first user message.
                first_human = next((m for m in messages if m.type == "human"), None)
                if chat.title == "New chat" and first_human is not None:
                    text = str(first_human.content).strip().replace("\n", " ")
                    chat.title = (text[:57] + "...") if len(text) > 60 else text
                chat.updated_at = _now()
            db.commit()

    def clear(self) -> None:
        with self._sf() as db:
            db.execute(delete(Message).where(Message.session_id == self.session_id))
            db.commit()
