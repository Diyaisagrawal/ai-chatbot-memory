"""Database engine, ORM models and session factory (SQLAlchemy 2.x)."""

import logging
import uuid
from collections.abc import Iterator
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, String, Text, create_engine, event, inspect
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session as DBSession,
)


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(64))
    password_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(default=_now)

    sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(120), default="New chat")
    persona: Mapped[str] = mapped_column(String(32), default="assistant")
    created_at: Mapped[datetime] = mapped_column(default=_now)
    updated_at: Mapped[datetime] = mapped_column(default=_now, onupdate=_now)

    user: Mapped[User] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # human | ai | system
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=_now)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class Database:
    """Owns the engine and session factory so tests can swap in a temp DB."""

    def __init__(self, url: str):
        self.url = url
        self._backup_legacy_sqlite(url)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=connect_args)
        if url.startswith("sqlite"):
            # Enforce ON DELETE CASCADE in SQLite.
            event.listen(self.engine, "connect", lambda conn, _: conn.execute("PRAGMA foreign_keys=ON"))
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    @staticmethod
    def _backup_legacy_sqlite(url: str) -> None:
        """v1 stored users by username only (no email/password). If an old SQLite
        file is found, move it aside so the new schema starts cleanly."""
        if not url.startswith("sqlite:///") or url.endswith(":memory:"):
            return
        path = Path(url.removeprefix("sqlite:///"))
        if not path.exists():
            return
        probe = create_engine(url)
        try:
            insp = inspect(probe)
            legacy = insp.has_table("users") and "password_hash" not in {
                c["name"] for c in insp.get_columns("users")
            }
        finally:
            probe.dispose()
        if legacy:
            backup = path.with_name(f"{path.stem}.v1-backup{path.suffix}")
            path.replace(backup)
            logging.getLogger(__name__).warning(
                "Old database schema found; moved it to %s and created a new database.", backup
            )

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self) -> Iterator[DBSession]:
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
