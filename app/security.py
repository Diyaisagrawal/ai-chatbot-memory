"""Password hashing (bcrypt) and login tokens (JWT, HS256)."""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import jwt

from app.config import Settings

log = logging.getLogger(__name__)
ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


def resolve_secret(settings: Settings) -> str:
    """Use JWT_SECRET if set; otherwise create one once and keep it in a local file."""
    if settings.jwt_secret:
        return settings.jwt_secret
    path = Path(settings.secret_key_file)
    if path.exists() and path.read_text().strip():
        return path.read_text().strip()
    secret = secrets.token_urlsafe(48)
    path.write_text(secret)
    log.info("Generated a new JWT secret in %s", path)
    return secret


class TokenService:
    def __init__(self, secret: str, ttl_hours: int):
        self.secret = secret
        self.ttl = timedelta(hours=ttl_hours)

    def create(self, user_id: str) -> str:
        now = datetime.now(timezone.utc)
        payload = {"sub": user_id, "iat": now, "exp": now + self.ttl}
        return jwt.encode(payload, self.secret, algorithm=ALGORITHM)

    def user_id_from(self, token: str) -> str | None:
        try:
            return jwt.decode(token, self.secret, algorithms=[ALGORITHM])["sub"]
        except jwt.PyJWTError:
            return None
