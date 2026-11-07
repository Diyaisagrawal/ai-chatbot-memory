"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.chain import ChatService
from app.config import Settings, get_settings
from app.db import Database
from app.llm.provider import build_llm
from app.security import TokenService, resolve_secret

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.database_url)
        db.create_all()
        app.state.db = db
        app.state.tokens = TokenService(resolve_secret(settings), settings.token_ttl_hours)
        app.state.chat_service = ChatService(
            llm=build_llm(settings),
            session_factory=db.SessionLocal,
            memory_window=settings.memory_window,
        )
        logging.getLogger(__name__).info(
            "Started with provider=%s model=%s",
            settings.llm_provider,
            settings.hf_model_ids if settings.llm_provider == "hf_endpoint" else settings.hf_local_model_id,
        )
        yield
        db.engine.dispose()

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "provider": settings.llm_provider}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
